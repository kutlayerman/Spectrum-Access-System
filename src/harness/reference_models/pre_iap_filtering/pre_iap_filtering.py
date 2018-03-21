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
  This is the main Pre-IAP reference model which invokes the sub pre-IAP reference
  models to filter out grants and CBSDs before IAP model is invoked.
==================================================================================
"""
import copy
from reference_models.fss_purge import fss_purge
from reference_models.inter_sas_duplicate_grant import inter_sas_duplicate_grant


def performPreIapFiltering(protected_entities, input_sas_uut_fad, input_sas_test_harness_fads):
  """ The main function that invokes all pre-IAP filtering models.

  The grants/CBSDs to be purged are removed from the input parameters.
  
  Args:
    protected_entities: : A dictionary containing the list of protected entities. The key
      is a protected enity type and the value is a list of corresponding protected
      entity records. The format is {'entity_name':[record1, record2]}.
    input_sas_uut_fad_object: A FullActivityDump object containing the FAD records of SAS UUT.
    input_sas_test_harness_fads: A list of FullActivityDump objects containing the FAD records
      from SAS test harnesses.
  Returns:
    uut_fad_after_fss_purge: A FullActivityDump object containing the FAD record of SAS UUT 
      after purged grants removed.
    test_harness_fads_after_fss_purge: A list of FullActivityDump objects containing the FAD 
      record of SAS test harness after purged grants removed.
  """    

  sas_uut_fad_object = copy.deepcopy(input_sas_uut_fad)
  test_harness_fad_objects = copy.deepcopy(input_sas_test_harness_fads)

  # Invoke Inter SAS duplicate grant purge list reference model
  uut_fad_after_duplicate_removal, test_harness_fads_after_duplicate_removal = inter_sas_duplicate_grant.interSasDuplicateGrantReferenceModel\
                                (sas_uut_fad_object, test_harness_fad_objects)

  # TODO
  # Invoke PPA, EXZ, GWPZ, and FSS+GWBL purge list reference models

  # Invoke FSS purge list reference model
  uut_fad_after_fss_purge, test_harness_fads_after_fss_purge = fss_purge.fssPurgeModel(
                        uut_fad_after_duplicate_removal, test_harness_fads_after_duplicate_removal,
                        protected_entities['fssRecords'])
  return uut_fad_after_fss_purge, test_harness_fads_after_fss_purge
