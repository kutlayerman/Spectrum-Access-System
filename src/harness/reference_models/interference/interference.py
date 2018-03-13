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
  Compute Interference caused by a grant for all the incumbent types
  APIs in this file are used by IAP and Aggregate Interference Reference Models

  The main routines are:

    computeInterferencePpaGwpzPoint 
    computeInterferenceEsc 
    computeInterferenceFss 
    computeInterferenceFssBlocking 
    calculateInterference 

  The common utility APIs are:

    getAllGrantInformationFromCbsdDataDump 
    findOverlappingGrantsInsideNeighborhood
    getProtectedChannels

  The routines return a interference caused by a grant in the neighborhood of 
  FSS/GWPZ/PPA/ESC incumbent types
==================================================================================
"""
import numpy as np
from reference_models.antenna import antenna
from reference_models.antenna import fss_pointing
from reference_models.geo import vincenty
from reference_models.propagation import wf_itm
from reference_models.propagation import wf_hybrid
from reference_models.examples import entities, example_fss_interference
from collections import namedtuple
from enum import Enum
from shapely.geometry import Point as SPoint
from shapely.geometry import Polygon as SPolygon
from shapely.geometry import MultiPolygon as MPolygon
from shapely.geometry import shape, Point, LineString

# Initialize terrain driver
# terrainDriver = terrain.TerrainDriver()
terrainDriver = wf_itm.terrainDriver

# Set constant parameters based on requirements in the WINNF-TS-0112
# [R2-SGN-16]
GWPZ_NBRHD_DIST = 40  # neighborhood distance from a CBSD to a given protection
# point (in km) in GWPZ protection area
PPA_NBRHD_DIST = 40  # neighborhood distance from a CBSD to a given protection
# point (in km) in PPA protection area

FSS_CO_CHANNEL_NBRHD_DIST = 150  # neighborhood distance from a CBSD to FSS for
# co-channel protection

FSS_BLOCKING_NBRHD_DIST = 40  # neighborhood distace from a CBSD to FSS
# blocking protection

ESC_NBRHD_DIST_A = 40  # neighborhood distance from a ESC to category A CBSD

ESC_NBRHD_DIST_B = 80  # neighborhood distance from a ESC to category B CBSD

NUM_OF_PROCESSES = 6

# Frequency used in propagation model (in MHz) [R2-SGN-04]
FREQ_PROP_MODEL = 3625.0

# CBRS Band Frequency Range (Hz)
CBRS_LOW_FREQ = 3550.e6
CBRS_HIGH_FREQ = 3700.e6

# FSS Passband low frequency range  (Hz)
FSS_LOW_FREQ = 3600.e6

# FSS Passband for TT&C (Hz)
FSS_TTC_LOW_FREQ = 3700.e6
FSS_TTC_HIGH_FREQ = 4200.e6

# ESC IAP for Out-of-Band Categoroy A CBSDs in Frequency Range (Hz)
ESC_CAT_A_LOW_FREQ = 3550.e6
ESC_CAT_A_HIGH_FREQ = 3660.e6

# ESC IAP for Out-of-Band Categoroy B CBSDs in Frequency Range (Hz)
ESC_CAT_B_LOW_FREQ = 3550.e6
ESC_CAT_B_HIGH_FREQ = 3680.e6

# ESC Channel 21 Center Frequency
ESC_CH21_CF = 36525.e5

# One Mega Herts
ONE_MHZ = 1.e6

# Channel bandwidth over which SASs execute the IAP process
IAPBW = 5.e6

# GWPZ Area Protection reference bandwidth for the IAP process
GWPZ_RBW = 10.e6

# PPA Area Protection reference bandwidth for the IAP process
PPA_RBW = 10.e6

# GWPZ and PPA height (m)
GWPZ_PPA_HEIGHT = 1.5

# Number of SASs - NSAS
NUM_SAS = 3

# Global grant counter
grant_counter = 0

# In-band insertion loss
IN_BAND_INSERTION_LOSS = 0.5

# Define an enumeration class named ProtectionEntityType with members
# 'GWPZ_AREA', 'PPA_AREA', 'FSS_CO_CHANNEL', 'FSS_BLOCKING', 'ESC_CAT_A'
# 'ESC_CAT_B'


class ProtectionEntityType(Enum):
  GWPZ_AREA = 1
  PPA_AREA = 2
  FSS_CO_CHANNEL = 3
  FSS_BLOCKING = 4
  ESC_CAT_A = 5
  ESC_CAT_B = 6


# Define CBSD grant, i.e., a tuple with named fields of 'latitude',
# 'longitude', 'height_agl', 'indoor_deployment', 'antenna_azimuth',
# 'antenna_gain', 'antenna_beamwidth', 'cbsd_category', 'grant_index',
# 'max_eirp', 'low_frequency', 'high_frequency', 'is_managed_grant'
CbsdGrantInformation = namedtuple('CbsdGrantInformation',
                       ['latitude', 'longitude', 'height_agl', 
                        'indoor_deployment', 'antenna_azimuth', 'antenna_gain',
                        'antenna_beamwidth', 'cbsd_category', 'grant_index', 
                        'max_eirp', 'low_frequency', 'high_frequency', 
                        'is_managed_grant'])

# Define protection constraint, i.e., a tuple with named fields of
# 'latitude', 'longitude', 'low_frequency', 'high_frequency'
ProtectionConstraint = namedtuple('ProtectionConstraint',
                                  ['latitude', 'longitude', 'low_frequency',
                                   'high_frequency', 'entity_type'])

# Define FSS information, i.e., a tuple with named fields of
# 'fss_entities', 'antenna_height', 'weight1', 'weight2'
FssInformation = namedtuple('FssInformation',
                                  ['antenna_height', 'antenna_azimuth',
                                   'antenna_elevation', 'antenna_gain',
                                   'weight1', 'weight2'])

# Define ESC information, i.e., a tuple with named fields of
# 'antenna_height', 'antenna_azimuth', 'antenna_gain', 'antenna_pattern_gain'
EscInformation = namedtuple('EscInformation',
                                 ['antenna_height', 'antenna_azimuth',
                                  'antenna_gain', 'antenna_pattern_gain'])

def dbToLinear(x):
  """This function returns dBm to mW converted value"""
  return 10**(x / 10)

def linearToDb(x):
  """This function returns mW to dBm converted value"""
  return 10 * np.log10(x)

def performChannelization(low_freq, high_freq):
  """This function performs 5MHz channelization """
  protection_channels = []
  while (low_freq < high_freq):
    ch_low_freq = low_freq
    ch_high_freq = low_freq + 5
    protection_channels.append((ch_low_freq * ONE_MHZ, ch_high_freq * ONE_MHZ))
    low_freq = low_freq + 5
  return protection_channels

def getProtectedChannels(low_freq, high_freq):
  """This function returns protected channels list 

  Args:
    low_freq: Low frequency of the protected entity.
    high_freq: High frequency of the protected entity
  Returns:
    An array of protected channnel frequency range tuple 
    (low_freq,high_freq).
  """
  low_freq = low_freq / ONE_MHZ 
  high_freq = high_freq / ONE_MHZ
  
  assert low_freq < high_freq, 'Low frequency is greater than high frequency'

  if low_freq >= 3550 and low_freq <= 3700:
    if high_freq <= 3700:
      return performChannelization(low_freq, high_freq)
    else:
      return performChannelization(low_freq, 3700)
  elif low_freq <= 3550:
    if high_freq <= 3700 and high_freq >= 3550:
      return performChannelization(3550, high_freq)
    elif high_freq >= 3700:
      return performChannelization(3550, 3700)
  else:
    raise ValueError('Invalid frequency range %s,%s', low_freq, high_freq)


def findOverlappingGrantsInsideNeighborhood(grants, constraint):
  """
  Identify the CBSD grants in the neighborhood of protection constraint.

  Args:
    grants: a list of grants
    constraint: protection constraint of type ProtectionConstraint
  Returns:
    grants_inside: a list of grants, each one being a namedtuple of type
                   CbsdGrantInformation, of all CBSDs inside the neighborhood 
                   of the protection constraint.
  """
  # Initialize an empty list
  grants_inside = []

  # Loop over each CBSD grant
  for grant in grants:
    # Compute distance from CBSD location to protection constraint location
    dist_km, _, _ = vincenty.GeodesicDistanceBearing(
         grant.latitude, grant.longitude,
         constraint.latitude, constraint.longitude)

    # Check if CBSD is inside the neighborhood of protection constraint
    cbsd_in_nbrhd = False
    if constraint.entity_type is ProtectionEntityType.GWPZ_AREA:
      if dist_km <= GWPZ_NBRHD_DIST:
        cbsd_in_nbrhd = True
    elif constraint.entity_type is ProtectionEntityType.PPA_AREA:
      if dist_km <= PPA_NBRHD_DIST:
        cbsd_in_nbrhd = True
    elif constraint.entity_type is ProtectionEntityType.FSS_CO_CHANNEL:
      if dist_km <= FSS_CO_CHANNEL_NBRHD_DIST:
        cbsd_in_nbrhd = True
    elif constraint.entity_type is ProtectionEntityType.FSS_BLOCKING:
      if dist_km <= FSS_BLOCKING_NBRHD_DIST:
        cbsd_in_nbrhd = True
    elif constraint.entity_type is ProtectionEntityType.ESC_CAT_A:
      if grant.cbsd_category == 'A':
        if dist_km <= ESC_NBRHD_DIST_A:
          cbsd_in_nbrhd = True
    elif constraint.entity_type is ProtectionEntityType.ESC_CAT_B:
      if grant.cbsd_category == 'B':
        if dist_km <= ESC_NBRHD_DIST_B:
          cbsd_in_nbrhd = True
    else:
      raise ValueError('Unknown protection entity type' 
                              '%s' % constraint.entity_type)

    if cbsd_in_nbrhd:
      # Check frequency range
      overlapping_bw = min(grant.high_frequency, constraint.high_frequency) \
                          - max(grant.low_frequency, constraint.low_frequency)
      freq_check = (overlapping_bw > 0)
      # Append the grantn information if it is inside the neighborhood of
      # protection constraint
      if freq_check:
        grants_inside.append(grant)

  return grants_inside


def getAllGrantInformationFromCbsdDataDump(cbsd_data_records, 
      is_managing_sas=True):
  """Extract list of CbsdGrantInformation namedtuple

  Routine to extract CbsdGrantInformation tuple from CBSD data records from 
  FAD objects

  Args:
    cbsd_data_records: A list CbsdData object retrieved from FAD records.
    is_managing_sas: flag indicating cbsd dump is from managing SAS or 
                     peer SAS
                     True - Managing SAS, False - Peer SAS
  Returns:
    grant_objects: List of CBSD grant objects
  """

  grant_objects = []

  # Loop over each CBSD grant
  for i in range(len(cbsd_data_records)):
    registration = cbsd_data_records[i].get('registrationRequest')
    grants = cbsd_data_records[i].get('grantRequests')

    # Check CBSD location
    lat_cbsd = registration.get('installationParam', {}).get('latitude')
    lon_cbsd = registration.get('installationParam', {}).get('longitude')
    height_cbsd = registration.get('installationParam', {}).get('height')
    height_type_cbsd = registration.get('installationParam', {})\
        .get('heightType')
    if height_type_cbsd == 'AMSL':
      altitude_cbsd =\
          terrainDriver.GetTerrainElevation(lat_cbsd, lon_cbsd)
      height_cbsd = height_cbsd - altitude_cbsd

    # Sanity check on CBSD antenna height
    if height_cbsd < 1 or height_cbsd > 1000:
      raise ValueError('CBSD height is less than'
                           '1 m or greater than 1000 m.')

    global grant_counter
    for grant in grants:
      grant_counter += 1
      # Return CBSD information
      cbsd_grant = CbsdGrantInformation(
        # Get information from the registration
        latitude=lat_cbsd,
        longitude=lon_cbsd,
        height_agl=height_cbsd,
        indoor_deployment=registration.get('installationParam', {})
                                      .get('indoorDeployment'),
        antenna_azimuth=registration.get('installationParam', {})
                                    .get('antennaAzimuth'),
        antenna_gain=registration.get('installationParam', {})
                                 .get('antennaGain'),
        antenna_beamwidth=registration.get('installationParam', {})
                                      .get('antennaBeamwidth'),
        cbsd_category=registration.get('cbsdCategory'),
        grant_index=grant_counter,
        max_eirp=grant.get('operationParam', {}).get('maxEirp'),
        low_frequency=grant.get('operationParam', {})
                           .get('operationFrequencyRange', {})
                           .get('lowFrequency'),
        high_frequency=grant.get('operationParam', {})
                            .get('operationFrequencyRange', {})
                            .get('highFrequency'),
        is_managed_grant=is_managing_sas)
      grant_objects.append(cbsd_grant)
  return grant_objects

def computeInterferencePpaGwpzPoint(cbsd_grant, constraint, h_inc_ant, 
                                  max_eirp, region='SUBURBAN'):
  """
  Calculate interference contribution of each grant in the neighborhood to
  the protection constraint c.
  Args:
    cbsd_grant: a namedtuple of type CbsdGrantInformation
    constraint: protection constraint of type ProtectionConstraint
    h_inc_ant: reference incumbent antenna height (in meters)
    max_eirp: The maximum EIRP to be considered
    region: Region type of the GWPZ or PPA area
  Returns:
    interference: interference contribution
  """

  # Get the the propogation loss and incident angles for area entity 
  db_loss, incidence_angles, _ = wf_hybrid.CalcHybridPropagationLoss(
                                  cbsd_grant.latitude, cbsd_grant.longitude,
                                  cbsd_grant.height_agl, constraint.latitude,
                                  constraint.longitude, h_inc_ant,
                                  cbsd_grant.indoor_deployment,
                                  reliability=-1, 
                                  freq_mhz=FREQ_PROP_MODEL,
                                  region=region)

  # Compute CBSD antenna gain in the direction of protection point
  ant_gain = antenna.GetStandardAntennaGains(incidence_angles.hor_cbsd,
                                             cbsd_grant.antenna_azimuth,
                                             cbsd_grant.antenna_beamwidth,
                                             cbsd_grant.antenna_gain)

  # For ppa and gwpz CBSD antenna gain in the direction of protection
  # entity is the total antenna gain
  entity_ant_gain = ant_gain

  # Get the inteference value for area entity
  interference = calculateInterference(max_eirp, cbsd_grant.antenna_gain,
                                          entity_ant_gain, db_loss, 'true')
  return interference


def computeInterferenceEsc(cbsd_grant, constraint, esc_antenna_info, max_eirp):
  """
  Calculate interference contribution of each grant in the neighborhood to
  the protection constraint c.
  Args:
    cbsd_grant: a namedtuple of type CbsdGrantInformation
    constraint: protection constraint of type ProtectionConstraint
    esc_antenna_info: contains information on ESC antenna height, azimuth,
                      gain and pattern gain
    max_eirp: The maximum EIRP to be considered
  Returns:
    interference: interference contribution
  """

  # Get the the propogation loss and incident angles for ESC entity_ant_gain
  db_loss, incidence_angles, _ = wf_itm.\
      CalcItmPropagationLoss(
          cbsd_grant.latitude, cbsd_grant.longitude,
          cbsd_grant.height_agl, constraint.latitude,
          constraint.longitude, esc_antenna_info.antenna_height,
          cbsd_grant.indoor_deployment, reliability=-1,
          freq_mhz=FREQ_PROP_MODEL)

  # Compute CBSD antenna gain in the direction of protection point
  ant_gain = antenna.GetStandardAntennaGains(incidence_angles.hor_cbsd,
                                             cbsd_grant.antenna_azimuth,
                                             cbsd_grant.antenna_beamwidth,
                                             cbsd_grant.antenna_gain)

  # Compute ESC antenna gain in the direction of CBSD
  esc_ant_gain = antenna.\
      GetAntennaPatternGains(incidence_angles.hor_rx,
                             esc_antenna_info.antenna_azimuth,
                             esc_antenna_info.antenna_pattern_gain,
                             esc_antenna_info.antenna_gain)

  # Get the total antenna gain by summing the antenna gains from CBSD to ESC
  # and ESC to CBSD
  entity_ant_gain = ant_gain + esc_ant_gain

  # Compute the interference value for ESC entity
  interference = calculateInterference(max_eirp, cbsd_grant.antenna_gain,
                                          entity_ant_gain, db_loss, 'false')
  return interference


def computeInterferenceFss(cbsd_grant, constraint, fss_info, max_eirp):
  """
  Calculate interference contribution of each grant in the neighborhood to
  the protection constraint c.
  Args:
    cbsd_grant: a namedtuple of type CbsdGrantInformation
    constraint: protection constraint of type ProtectionConstraint
    fss_info: contains information on fss entities, antenna height
              and weights on the tangent and perpendicular 
              compnents.
    max_eirp: The maximum EIRP to be considered
  Returns:
    interference: interference contribution
  """

  # Get the the propogation loss and incident angles for FSS entity_type
  db_loss, incidence_angles, _ = wf_itm.\
      CalcItmPropagationLoss(
          cbsd_grant.latitude, cbsd_grant.longitude,
          cbsd_grant.height_agl, constraint.latitude,
          constraint.longitude, fss_info.antenna_height,
          cbsd_grant.indoor_deployment, reliability=-1,
          freq_mhz=FREQ_PROP_MODEL)

  # Compute CBSD antenna gain in the direction of protection point
  ant_gain = antenna.GetStandardAntennaGains(incidence_angles.hor_cbsd,
                                             cbsd_grant.antenna_azimuth,
                                             cbsd_grant.antenna_beamwidth,
                                             cbsd_grant.antenna_gain)

  # Compute FSS antenna gain in the direction of CBSD
  fss_ant_gain = antenna.GetFssAntennaGains(
                       incidence_angles.hor_rx, incidence_angles.ver_rx,
                       fss_info.antenna_azimuth,
                       fss_info.antenna_elevation,
                       fss_info.antenna_gain,
                       fss_info.weight1, fss_info.weight2)

  # Get the total antenna gain by summing the antenna gains from CBSD to FSS
  # and FSS to CBSD
  entity_ant_gain = ant_gain + fss_ant_gain

  # Compute the interference value for ESC entity
  interference = calculateInterference(max_eirp, cbsd_grant.antenna_gain,
                                         entity_ant_gain, db_loss, 'false')
  return interference


def computeInterferenceFssBlocking(cbsd_grant, constraint, fss_info, max_eirp):
  """Compute interference for FSS Blocking pass band

  Calculate interference contribution from a grant in the neighborhood to
  the FSS blocking pass band.
  Args:
    cbsd_grant: a namedtuple of type CbsdGrantInformation
    constraint: protection constraint of type ProtectionConstraint
    fss_info: contains information on fss entities, antenna height
              and weights on the tangent and perpendicular 
              compnents.
    max_eirp: The maximum EIRP to be considered
  Returns:
    interference: interference contribution
  """

  # Get the the propogation loss and incident angles for FSS entity 
  # blocking channels
  db_loss, incidence_angles, _ = wf_itm.\
      CalcItmPropagationLoss(
          cbsd_grant.latitude, cbsd_grant.longitude,
          cbsd_grant.height_agl, constraint.latitude,
          constraint.longitude, fss_info.antenna_height,
          cbsd_grant.indoor_deployment, reliability=-1,
          freq_mhz=FREQ_PROP_MODEL)

  # Compute CBSD antenna gain in the direction of protection point
  ant_gain = antenna.GetStandardAntennaGains(incidence_angles.hor_cbsd,
                                             cbsd_grant.antenna_azimuth,
                                             cbsd_grant.antenna_beamwidth,
                                             cbsd_grant.antenna_gain)

  # Compute FSS antenna gain in the direction of CBSD
  fss_ant_gain = antenna.GetFssAntennaGains(
             incidence_angles.hor_rx, incidence_angles.ver_rx,
             fss_info.antenna_azimuth, fss_info.antenna_elevation,
             fss_info.antenna_gain, fss_info.weight1, fss_info.weight2)

  # Compute EIRP of CBSD grant inside the frequency range of 
  # protection constraint
  eirp_cbsd = (max_eirp - cbsd_grant.antenna_gain) + ant_gain + \
               fss_ant_gain + linearToDb((cbsd_grant.high_frequency -
                               cbsd_grant.low_frequency) / ONE_MHZ)

  # Get 50MHz offset below the lower edge of the FSS earth station
  offset = constraint.low_frequency - 50.e6

  # Get CBSD grant frequency range
  cbsd_freq_range = cbsd_grant.high_frequency - cbsd_grant.low_frequency

  fss_mask = 0

  # if lower edge of the FSS passband is less than CBSD grant
  # lowFrequency and highFrequency
  if constraint.low_frequency < cbsd_grant.low_frequency and\
         constraint.low_frequency < cbsd_grant.high_frequency:
     fss_mask = 0.5
  # if CBSD grant lowFrequency and highFrequency is less than
  # 50MHz offset from the FSS passband lower edge
  elif cbsd_grant.low_frequency < offset and\
            cbsd_grant.high_frequency < offset:
       fss_mask = linearToDb((cbsd_freq_range / ONE_MHZ) * 0.25)
  # if CBSD grant lowFrequency is less than 50MHz offset and
  # highFrequency is greater than 50MHz offset
  elif cbsd_grant.low_frequency < offset and\
               cbsd_grant.high_frequency > offset:
       fss_mask = linearToDb(((offset - cbsd_grant.low_frequency) /
                                                        ONE_MHZ) * 0.25)
       fss_mask = fss_mask + \
                    linearToDb(((cbsd_grant.high_frequency - offset) / 
                                                 ONE_MHZ) * 0.6)
  # if FSS Passband lower edge frequency is grater than CBSD grant
  # lowFrequency and highFrequency and
  # CBSD grand low and high frequencies are greater than 50MHz offset
  elif constraint.low_frequency > cbsd_grant.low_frequency and \
         constraint.low_frequency > cbsd_grant.high_frequency and \
         cbsd_grant.low_frequency > offset and\
                cbsd_grant.high_frequency > offset:
       fss_mask = linearToDb((cbsd_freq_range / ONE_MHZ) * 0.6)

  # Calculate the interference contribution
  interference = eirp_cbsd - fss_mask - db_loss

  return interference


def calculateInterference(max_eirp, cbsd_ant_gain, entity_ant_gain, 
                          db_loss, area, reference_bandwidth=IAPBW):
  """Calculate interference caused by a grant 
  
  Utility API to calculate interference caused by a grant in the 
  neighborhood of the protected entity FSS/ESC/PPA/GWPZ.

  Args:
    max_eirp: The maximum EIRP value of the grant to be considered
    cbsd_ant_gain: The antenna gain of the CBSD containing the grant
    entity_ant_gain: The sum of antenna gains from CBSD to entity and 
                     entity to CBSD
    db_loss: The calculated prpogation loss
    area: True for area entity and false for point 
          entity_ant_gain
    reference_bandwidth: Reference bandwidth over which interference is 
                         calculated
  Returns:
    interference: Total interference contribution of the entity
  """

  if area == 'true':
    eirp_cbsd = ((max_eirp - cbsd_ant_gain) + entity_ant_gain + linearToDb
            (reference_bandwidth / ONE_MHZ) - 0)
  else:
    eirp_cbsd = ((max_eirp - cbsd_ant_gain) + entity_ant_gain + linearToDb
            (reference_bandwidth / ONE_MHZ) - IN_BAND_INSERTION_LOSS)

  interference = eirp_cbsd - db_loss
  return interference


