#    Copyright 2018 SAS Project Authors. All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
==================================================================================
  Iterative Allocation Process(IAP) Reference Model [R2-SGN-16].
  Based on WINNF-TS-0112, Version V1.4.1, 16 January 2018.

  This model provides functions to calculate post IAP allowed interference margin for 
  the protection constraint of FSS, ESC, GWPZ and PPA. 

  The main routines are:

    performIap 
    performIapForEsc
    performIapForGwpz
    performIapForPpa
    performIapForFss
    calculateApIapRef

  The routines return a dictionary containing key as protection constraints 
  (lat, long, low_freq, high_freq, entity_type) and value as the post IAP aggregate 
  interference value from all the CBSDs managed by the SAS in mW/RBW for each 
  of the constraints.
==================================================================================
"""

from reference_models.geo import utils
from reference_models.interference import interference as interf
import numpy as np
import logging
from collections import namedtuple
from multiprocessing import Pool

# Define Post IAP Grant information, i.e., a tuple with named fields of
# 'eirp', 'interference'
PostIAPGrantInfo = namedtuple('PostIAPGrantInfo', ['eirp', 'interference'])

# Global Post IAP managed grants container
post_iap_managed_grants = {}

# Global container for aggregate interference calculated by managing SAS using
# the EIRP obtained by all CBSDs (including the CBSDs managed by other SASs)
# through application of IAP for protected entity pi.(mW/RBW)
aggregate_interference = {}

# Global IAP Allowed Margin container (mW/RBW)
iap_allowed_margin = {}

# Global IAP protection threshold per IAPBW_HZ container (dBm/IAPBW_HZ)
threshold_per_iapbw = {}

# values from WINNF-TS-0061-V1.1.0 - WG4 SAS Test and Certification Spec-Table 
# 8.4-2 Protected entity reference for IAP Protection 
# The values are in (dBm/RBW)
protection_thresholds = {'QPpa': -80, 'QGwpz': -80, 'QFssCochannel': -129, 
                         'QFssBlocking': -60, 'QEsc': -109}

# Default 1 dB pre IAP margin for all incumbent types
pre_iap_headrooms = {'MgPpa': 1, 'MgGwpz': 1, 'MgFssCochannel': 1, 
                     'MgFssBlocking': 1, 'MgEsc': 1}


def iapPointConstraint(protection_point, channels, low_freq, high_freq, 
                       grant_objects, fss_info, esc_antenna_info, region_type, 
                       threshold, protection_ent_type):
  """ Routine to compute aggregate interference(Ap) from authorized grants 
   
  Aggregate interference is calculated over all the 5MHz channels.This routine 
  is applicable for FSS Co-Channel and ESC Sensor protection points.Also for 
  protection points within PPA and GWPZ protection areas. Post IAP grant EIRP
  and interference contribution is calculated over each 5MHz channel

  Args:
    protection_point: List of protection points inside protection area
    channels: List of tuples containing (low_freq, high_freq) of each 5MHz 
              segment of the whole frequency range of protected entity
    low_freq: low frequency of protection constraint (Hz)
    high_freq: high frequency of protection constraint (Hz)
    grant_objects: a list of grant requests, each one being a tuple 
                   containing grant information
    fss_info: contains information on antenna height and 
              weights on the tangent and perpendicular components.
    esc_antenna_info: contains information on ESC antenna height, azimuth, 
                      gain and pattern gain
    threshold: protection threshold (dBm/IAPBW_HZ)
    protection_ent_type: an enum member of class ProtectedEntityType
  Returns:
    Updates global container aggregate_interference dictionary in the format of 
    {(latitude,longitude,low_freq,high_freq,entity_type):aggregate_interference}
    aggregate_interference value is in mW/RBW.
    and global container post_iap_managed_grants nested dictionary in the format of
    {grant_id:{ProtectionConstraint:PostIAPGrantInfo,ProtectionConstraint:PostIAPGrantInfo},
    where ProtectionConstraint and PostIAPGrantInfo is a namedtuple 
  """

  for channel in channels:
    # Set interference quota for protection_point, channel combination

    # Calculate Roll of Attenuation for ESC Sensor Protection entity
    if protection_ent_type is interf.ProtectedEntityType.ESC_CAT_A or\
      protection_ent_type is interf.ProtectedEntityType.ESC_CAT_B:
      if channel[0] >= 3650.e6:
        center_freq = (channel[0] + channel[1]) / 2
        interference_quota = interf.dbToLinear(interf.linearToDb(threshold) - 
                (2.5 + ((center_freq - interf.ESC_CH21_CF_HZ) / interf.ONE_MHZ)))
      else:
        interference_quota = threshold

    # Get the FSS blocking passband range
    elif protection_ent_type is interf.ProtectedEntityType.FSS_BLOCKING:
      interference_quota = threshold
      channel = [low_freq, high_freq]

    else:
      interference_quota = threshold

    # Get protection constraint over 5MHz channel range
    channel_constraint = interf.ProtectionConstraint(latitude=protection_point[0],
                           longitude=protection_point[1], low_frequency=channel[0],
                           high_frequency=channel[1], entity_type=protection_ent_type)

    # Identify CBSD grants in the neighborhood of the protection point 
    # and channel
    neighborhood_grants = interf.findOverlappingGrantsInsideNeighborhood(
                            grant_objects, channel_constraint)
    num_grants = len(neighborhood_grants)

    if num_grants is 0:
      continue

    # calculate the fair share
    fair_share = interference_quota / num_grants

    # Empty list of final IAP EIRP and interference assigned to the grant
    grants_eirp = []
    interference_list = []

    aggr_interference = 0

    # Initialize flag to False for all the grants
    grant_initialize_flag = []
    grant_initialize_flag[:num_grants] = [False] * num_grants

    # Initialize grant EIRP to maxEirp for all the grants 
    grants_eirp = [grant.max_eirp for grant in neighborhood_grants]
    for i, grant in enumerate(neighborhood_grants):
      grants_eirp.append(grant.max_eirp)

    while num_grants:
      for i, grant in enumerate(neighborhood_grants):
        if grant_initialize_flag[i] is False:
          # Compute interference grant causes to protection point over channel

          # Compute interference to FSS Co-channel protection constraint
          if protection_ent_type is interf.ProtectedEntityType.FSS_CO_CHANNEL:
            interference = interf.dbToLinear(interf.computeInterferenceFss(
                             grant, channel_constraint, fss_info, grants_eirp[i]))

          # Compute interference to FSS Blocking protection constraint
          elif protection_ent_type is interf.ProtectedEntityType.FSS_BLOCKING:
            interference = interf.dbToLinear(interf.computeInterferenceFssBlocking(
                             grant, channel_constraint, fss_info, grants_eirp[i]))

          # Compute interference to ESC protection constraint
          elif protection_ent_type is interf.ProtectedEntityType.ESC_CAT_A or\
                protection_ent_type is interf.ProtectedEntityType.ESC_CAT_B:
            interference = interf.dbToLinear(interf.computeInterferenceEsc(
                             grant, channel_constraint, esc_antenna_info, grants_eirp[i]))

          # Compute interference to GWPZ or PPA protection constraint
          else:
            interference = interf.dbToLinear(interf.computeInterferencePpaGwpzPoint(
                             grant, channel_constraint, interf.GWPZ_PPA_HEIGHT, 
                             grants_eirp[i], region_type))

          # calculated interference exceeds fair share of
          # interference to which the grants are entitled
          if interference < fair_share:
            interference_list.append(interference)
            # Grant is satisfied
            grant_initialize_flag[i] = True

      for i, grant in enumerate(neighborhood_grants):

        if grant_initialize_flag[i] is True:

          # Remove satisfied CBSD grant from consideration
          interference_quota = interference_quota - interference_list[i]
          num_grants = num_grants - 1

          # re-calculate the fair share if number of grants are not zero
          if num_grants != 0:
            fair_share = interference_quota / num_grants

          # Get aggregate interference
          aggr_interference += interference_list[i]

          # Update EIRP and Interference only if managed grant
          if grant.is_managed_grant:
            if grant.grant_index not in post_iap_managed_grants:
              post_iap_managed_grants[grant.grant_index] = {}

              # Nested dictionary - contains dictionary of protection
              # constraint over which grant extends
              post_iap_grant = PostIAPGrantInfo(eirp=grants_eirp[i], 
                                 interference=interference_list[i])
              post_iap_managed_grants[grant.grant_index][channel_constraint] =\
                                                             post_iap_grant
        else:
          grants_eirp[i] = grants_eirp[i] - 1

      aggregate_interference[channel_constraint] = aggr_interference


def performIapForEsc(protected_entity, grant_objects, iap_margin_esc):
  """ Compute post IAP interference margin for ESC 
  
  IAP algorithm is run over ESC passband 3550-3660MHz for Category A CBSDs and 
  3550-3680MHz for Category B CBSDs

  Args:
    protected_entity: List of protection points inside protection area
    grant_objects: a list of grant requests, each one being a dictionary 
                   containing grant information
    iap_margin_esc: Pre IAP threshold value for ESC incumbent type(mW/RBW)
  """

  # Get the protection point of the ESC
  esc_point = protected_entity['installationParam']
  protection_point = (esc_point['latitude'], esc_point['longitude'])

  # Get azimuth radiation pattern
  ant_pattern = esc_point['azimuthRadiationPattern']

  # Initialize array of size length of ant_pattern list
  ant_pattern_gain = np.arange(len(ant_pattern))
  for i, pattern in enumerate(ant_pattern):
    ant_pattern_gain[i] = pattern['gain']

  # Get ESC antenna information
  esc_antenna_info = interf.EscInformation(antenna_height=esc_point['height'],
                       antenna_azimuth=esc_point['antennaAzimuth'],
                       antenna_gain=esc_point['antennaGain'],
                       antenna_pattern_gain=ant_pattern_gain)

  pool = Pool(processes=min(interf.NUM_OF_PROCESSES, len(protection_point)))

  # ESC Category A CBSDs between 3550 - 3660 MHz within 40 km
  protection_channels = interf.getProtectedChannels(interf.ESC_CAT_A_LOW_FREQ_HZ, 
                          interf.ESC_CAT_A_HIGH_FREQ_HZ)

  logging.debug('$$$$ Calling ESC Category A Protection $$$$')
  # ESC algorithm
  iapPointConstraint(protection_point, protection_channels, 
    interf.ESC_CAT_A_LOW_FREQ_HZ, interf.ESC_CAT_A_HIGH_FREQ_HZ, 
    grant_objects, None, esc_antenna_info, None, 
    iap_margin_esc, interf.ProtectedEntityType.ESC_CAT_A)

  # ESC Category B CBSDs between 3550 - 3680 MHz within 80 km
  protection_channels = interf.getProtectedChannels(interf.ESC_CAT_B_LOW_FREQ_HZ, 
                          interf.ESC_CAT_B_HIGH_FREQ_HZ)

  logging.debug('$$$$ Calling ESC Category B Protection $$$$')
  iapPointConstraint(protection_point, protection_channels, 
    interf.ESC_CAT_B_LOW_FREQ_HZ, interf.ESC_CAT_B_HIGH_FREQ_HZ, 
    grant_objects, None, esc_antenna_info, None, 
    iap_margin_esc, interf.ProtectedEntityType.ESC_CAT_B)

  pool.close()
  pool.join()

  return


def performIapForGwpz(protected_entity, grant_objects, iap_margin_gwpz):
  """Compute post IAP interference margin for GWPZ incumbent type

  Routine to get protection points within GWPZ protection area and perform 
  IAP algorithm on each protection point 

  Args:
    protected_entity: List of protection points inside protection area
    grant_objects: a list of grant requests, each one being a dictionary 
                   containing grant information
    iap_margin_gwpz: Pre IAP threshold value for GWPZ incumbent type(mW/RBW)
  """

  logging.debug('$$$$ Getting GRID points for GWPZ Protection Area $$$$')
  # Get Fine Grid Points for a GWPZ protection area
  protection_points = utils.GridPolygon(protected_entity)

  gwpz_freq_range = protected_entity['deploymentParam']\
                      ['operationParam']['operationFrequencyRange']
  gwpz_low_freq = gwpz_freq_range['lowFrequency']
  gwpz_high_freq = gwpz_freq_range['highFrequency']

  gwpz_region = protected_entity['zone']['features'][0]\
                                 ['properties']['gwpzRegionType']

  # Get channels over which area incumbent needs partial/full protection
  protection_channels = interf.getProtectedChannels(gwpz_low_freq, gwpz_high_freq)

  # Apply IAP for each protection constraint with a pool of parallel processes.
  pool = Pool(processes=min(interf.NUM_OF_PROCESSES, len(protection_points)))

  logging.debug('$$$$ Calling GWPZ Protection $$$$')
  for protection_point in protection_points:
    iapPointConstraint(protection_point, protection_channels, gwpz_low_freq, 
      gwpz_high_freq, grant_objects, None, None, gwpz_region, 
      iap_margin_gwpz, interf.ProtectedEntityType.GWPZ_AREA)

  pool.close()
  pool.join()

  return


def performIapForPpa(protected_entity, grant_objects, 
                     iap_margin_ppa, pal_records):
  """Compute post IAP interference margin for PPA incumbent type

  Routine to get protection points within PPA protection area and perform 
  IAP algorithm on each protection point

  Args:
    protected_entity: List of protection points inside protection area
    grant_objects: a list of grant requests, each one being a dictionary 
                   containing grant information
    iap_margin_ppa: Pre IAP threshold value for PPA incumbent type(mW/RBW)
    pal_records: PAL records associated with a PPA protection area 
  """

  logging.debug('$$$$ Getting GRID points for PPA Protection Area $$$$')

  # Get Fine Grid Points for a PPA protection area
  protection_points = utils.GridPolygon(protected_entity)

  # Get the region type of the PPA protection area
  ppa_region = protected_entity['ppaInfo']['ppaRegionType']

  # Find the PAL records for the PAL_IDs defined in PPA records
  ppa_pal_ids = protected_entity['ppaInfo']['palId']

  for pal_record in pal_records:

    if pal_record['palId'] == ppa_pal_ids[0]:

      # Get the frequencies from the PAL records
      ppa_freq_range = pal_record['channelAssignment']['primaryAssignment']
      ppa_low_freq = ppa_freq_range['lowFrequency']
      ppa_high_freq = ppa_freq_range['highFrequency']

      # Get channels over which area incumbent needs partial/full protection
      protection_channels = interf.getProtectedChannels(ppa_low_freq, ppa_high_freq)

      # Apply IAP for each protection constraint with a pool of parallel 
      # processes.
      pool = Pool(processes=min(interf.NUM_OF_PROCESSES, 
                                   len(protection_points)))

      logging.debug('$$$$ Calling PPA Protection $$$$')
      for protection_point in protection_points:
        iapPointConstraint(protection_point, protection_channels, ppa_low_freq,
          ppa_high_freq, grant_objects, None, None, ppa_region,
          iap_margin_ppa, interf.ProtectedEntityType.PPA_AREA)
      break

  pool.close()
  pool.join()

  return


def performIapForFss(protected_entity, grant_objects, 
                     iap_margin_fss_cochannel, iap_margin_fss_blocking):
  """Compute post IAP interference margin for FSS incumbent types

  Routine to determine FSS protection criteria. Based on FSS protection 
  channel range and TT&C flag type of FSS protection is determined.
  FSS protection is provided over co-channel and blocking pass band

  Args:
    protected_entity: List of protection points inside protection area
    grant_objects: a list of grant requests, each one being a tuple
                              containing grant information
    iap_margin_fss_cochannel: Pre IAP threshold value for FSS co-channel 
                              incumbent type(mW/RBW)
    iap_margin_fss_blocking: Pre IAP threshold value for FSS Blocking 
                             incumbent type(mW/RBW)
  """

  # Get FSS T&C Flag value
  fss_ttc_flag = protected_entity['deploymentParam'][0]['ttc']

  fss_weight1 = protected_entity['deploymentParam'][0]['weight1']
  fss_weight2 = protected_entity['deploymentParam'][0]['weight2']

  # Get the protection point of the FSS
  fss_point = protected_entity['deploymentParam'][0]['installationParam']
  protection_point = (fss_point['latitude'], fss_point['longitude'])

  # Get the frequency range of the FSS
  fss_freq_range = protected_entity['deploymentParam'][0]\
      ['operationParam']['operationFrequencyRange']
  fss_low_freq = fss_freq_range['lowFrequency']
  fss_high_freq = fss_freq_range['highFrequency']

  # Get FSS information
  fss_info = interf.FssInformation(antenna_height=fss_point['height'],
               antenna_azimuth=fss_point['antennaAzimuth'],
               antenna_elevation=fss_point['antennaElevationAngle'],
               antenna_gain=fss_point['antennaGain'],
               weight1=fss_weight1, weight2=fss_weight2)

  pool = Pool(processes=min(interf.NUM_OF_PROCESSES, len(protection_point)))

  # FSS Passband is between 3600 and 4200
  if (fss_low_freq >= interf.FSS_LOW_FREQ_HZ and 
            fss_low_freq < interf.CBRS_HIGH_FREQ_HZ):
    # Get channels for co-channel CBSDs
    protection_channels = interf.getProtectedChannels(fss_low_freq, interf.CBRS_HIGH_FREQ_HZ)

    # FSS Co-channel algorithm
    logging.debug('$$$$ Calling FSS Protection $$$$')
    iapPointConstraint(protection_point, protection_channels, fss_low_freq,
      interf.CBRS_HIGH_FREQ_HZ, grant_objects, fss_info, None, 
      None, iap_margin_fss_cochannel, 
      interf.ProtectedEntityType.FSS_CO_CHANNEL)

    logging.debug('$$$$ Calling FSS In-band blocking Protection $$$$')

    # 5MHz channelization is not required for Blocking protection
    # FSS in-band blocking algorithm
    protection_channels = [interf.CBRS_LOW_FREQ_HZ, fss_low_freq]
    iapPointConstraint(protection_point, protection_channels, 
      interf.CBRS_LOW_FREQ_HZ, fss_low_freq, grant_objects, 
      fss_info, None, None, iap_margin_fss_blocking, 
      interf.ProtectedEntityType.FSS_BLOCKING)

  # FSS Passband is between 3700 and 4200 and TT&C flag is set to TRUE
  elif (fss_low_freq >= interf.FSS_TTC_LOW_FREQ_HZ and 
           fss_high_freq <= interf.FSS_TTC_HIGH_FREQ_HZ):
    if (fss_ttc_flag is True):

      # 5MHz channelization is not required for Blocking protection
      # FSS TT&C Blocking Algorithm
      logging.debug('$$$$ Calling FSS TT&C  blocking Protection $$$$')
      protection_channels = [interf.CBRS_LOW_FREQ_HZ, fss_low_freq]
      iapPointConstraint(protection_point, protection_channels, 
        interf.CBRS_LOW_FREQ_HZ, fss_low_freq, grant_objects, 
        fss_info, None, None, iap_margin_fss_blocking, 
        interf.ProtectedEntityType.FSS_BLOCKING)

    # FSS Passband is between 3700 and 4200 and TT&C flag is set to FALSE
    else:
      logging.debug('$$$$ IAP for FSS not applied for FSS Pass band 3700 to 4200' 
        'and TT&C flag set to false $$$$')

  pool.close()
  pool.join()

  return


def calculateIapQ():
  """Compute allowed interference margin 

  Routine to calculate allowed interference margin and protection 
  threshold for each 5MHz channel

  Returns:
    Updates threshold_per_iapbw(dBm/IAPBW_HZ) and iap_allowed_margin(mW/RBW) 
    global container
  """
  
  # Get threshold per IAPBW_HZ(dBm/IAPBW_HZ) and IAP allowed margin(mW/RBW)
  for key, interference_quota in protection_thresholds.iteritems():

    if key is 'QPpa':
      threshold_per_iapbw['iap_threshold_ppa'] = interference_quota + \
                    interf.linearToDb(interf.IAPBW_HZ / interf.PPA_RBW_HZ)
      iap_allowed_margin['iap_margin_ppa'] = interf.dbToLinear(
                      threshold_per_iapbw['iap_threshold_ppa'] -
                            pre_iap_headrooms.get('MgPpa'))

    elif key is 'QGwpz':
      threshold_per_iapbw['iap_threshold_gwpz'] = interference_quota + \
                       interf.linearToDb(interf.IAPBW_HZ / interf.GWPZ_RBW_HZ)
      iap_allowed_margin['iap_margin_gwpz'] = interf.dbToLinear(
                      threshold_per_iapbw['iap_threshold_gwpz'] -
                      pre_iap_headrooms.get('MgGwpz'))

    elif key is 'QFssCochannel':
      threshold_per_iapbw['iap_threshold_fss_cochannel'] = interference_quota + \
                       interf.linearToDb(interf.IAPBW_HZ / interf.ONE_MHZ)
      iap_allowed_margin['iap_margin_fss_cochannel'] = interf.dbToLinear(
                      threshold_per_iapbw['iap_threshold_fss_cochannel'] -
                      pre_iap_headrooms.get('MgFssCochannel'))

    elif key is 'QFssBlocking':
      threshold_per_iapbw['iap_threshold_fss_blocking'] = interference_quota
      iap_allowed_margin['iap_margin_fss_blocking'] = interf.dbToLinear(
                      threshold_per_iapbw['iap_threshold_fss_blocking'] -
                      pre_iap_headrooms.get('MgFssBlocking'))

    elif key is 'QEsc':
      threshold_per_iapbw['iap_threshold_esc'] = interference_quota + \
                       interf.linearToDb(interf.IAPBW_HZ / interf.ONE_MHZ)
      iap_allowed_margin['iap_margin_esc'] = interf.dbToLinear(
                      threshold_per_iapbw['iap_threshold_esc'] -
                      pre_iap_headrooms.get('MgEsc'))

  logging.debug('$$$$ IAP Allowed Margin in mW (q_p - m_g_p) : $$$$' + 
                                           str(iap_allowed_margin))

  logging.debug('$$$$ Threshold per IAPBW_HZ in dBm/IAPBW_HZ (q_p) : $$$$' + 
                                           str(threshold_per_iapbw))


def calculateApIapRef(num_sas):
  """Compute post IAP aggregate interference 

  Routine to calculate aggregate interference from all the CBSDs managed by the
  SAS at protected entity.

  Args:
    num_sas: number of SASs in the peer group 
  Returns:
    ap_iap_ref: post IAP aggregate interference dictionary in the format of 
                {(latitude,longitude,low_freq,high_freq,entity_type):ap_iap_ref}
  """

  ap_iap_ref = {}

  for p_ch_key, a_p_ch in aggregate_interference.iteritems():
    a_sas_p = 0

    # Calculate a_sas_p - Aggregate interference calculated using the
    # EIRP obtained by all CBSDs through application of IAP for protected
    # entity p. Only grants managed by SAS UUT are considered
    for grant_id, grant_info in post_iap_managed_grants.iteritems():
      for grant_protection_constraint, grant_params in grant_info.iteritems():
        if p_ch_key == grant_protection_constraint:
          # Aggregate interference if grant is inside protection constraint
          a_sas_p = a_sas_p + grant_params.interference 

    # Calculate ap_iap_ref for a particular protection point, channel
    if p_ch_key.entity_type == interf.ProtectedEntityType.ESC_CAT_A or \
         p_ch_key.entity_type == interf.ProtectedEntityType.ESC_CAT_B:
      q_p = threshold_per_iapbw['iap_threshold_esc']

    elif p_ch_key.entity_type == interf.ProtectedEntityType.GWPZ_AREA:
      q_p = threshold_per_iapbw['iap_threshold_gwpz']

    elif p_ch_key.entity_type == interf.ProtectedEntityType.PPA_AREA:
      q_p = threshold_per_iapbw['iap_threshold_ppa']

    elif p_ch_key.entity_type == interf.ProtectedEntityType.FSS_CO_CHANNEL:
      q_p = threshold_per_iapbw['iap_threshold_fss_cochannel']

    elif p_ch_key.entity_type == interf.ProtectedEntityType.FSS_BLOCKING:
      q_p = threshold_per_iapbw['iap_threshold_fss_blocking']

    ap_iap_ref[p_ch_key] = (interf.dbToLinear(q_p) - a_p_ch) / num_sas + a_sas_p
    
  return ap_iap_ref


def performIap(protected_entities, sas_uut_fad_object, sas_th_fad_objects, pal_records):
  """Main IAP Routine

  Routine to apply IAP to all existing and pending grants across all the 
  incumbent types.

  Args:
    protected_entities: a list of protected entities, containing FSS, 
                         GWPZ, PPA and ESC 
    sas_uut_fad_object: FAD object from SAS UUT
    sas_th_fad_object: a list of FAD objects from SAS Test Harness
  Returns:
    A list of post IAP aggregate interference from all the CBSDs managed by the 
    SAS at a ProtectionConstraint
  """

  # Calculate protection threshold per IAPBW_HZ
  calculateIapQ()

  # List of CBSD grant tuples extracted from FAD record
  grant_objects = []

  # Creating list of cbsds
  cbsd_list_uut = []
  cbsd_list_th = []

  for cbsds in sas_uut_fad_object.getCbsdRecords():
    cbsd_list_uut.append(cbsds)

  grant_objects = interf.getAllGrantInformationFromCbsdDataDump(cbsd_list_uut, True)

  for fad in sas_th_fad_objects:
    for cbsds in fad.getCbsdRecords():
      cbsd_list_th.append(cbsds)
      grant_objects_test_harness = interf.getAllGrantInformationFromCbsdDataDump(
                                     cbsd_list_th, False)

  grant_objects = grant_objects + grant_objects_test_harness

  # Get number of SAS 
  num_sas = len(sas_th_fad_objects) + 1

  for protected_entity in protected_entities:
    if (str(protected_entity)).find('FSS') is not -1:
      performIapForFss(protected_entity, grant_objects, 
        iap_allowed_margin['iap_margin_fss_cochannel'], 
        iap_allowed_margin['iap_margin_fss_blocking'])

    elif (str(protected_entity)).find('PART_90') is not -1:
      performIapForGwpz(protected_entity, grant_objects, 
        iap_allowed_margin['iap_margin_gwpz'])

    elif str((protected_entity)).find('PPA') is not -1:
      performIapForPpa(protected_entity, grant_objects, 
        iap_allowed_margin['iap_margin_ppa'], pal_records)

    elif (str(protected_entity)).find('esc') is not -1:
      performIapForEsc(protected_entity, grant_objects, 
        iap_allowed_margin['iap_margin_esc'])

  logging.debug('$$$$  AGGREGATE INTERFERENCE A_p (mW/RBW) : $$$$' + 
                                   str(aggregate_interference))
  logging.debug('$$$$ Post IAP Managed Grant $$$$$$\n' + 
                                   str(post_iap_managed_grants))

  # List of Post IAP Aggregate interference from all the CBSDs managed by
  # the SAS at the protected entity
  ap_iap_ref = calculateApIapRef(num_sas)

  return ap_iap_ref
