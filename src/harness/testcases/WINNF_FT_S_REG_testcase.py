#    Copyright 2016 SAS Project Authors. All Rights Reserved.
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
import json
import os
import unittest

import sas
from util import winnforum_testcase


class RegistrationTestcase(unittest.TestCase):

  def setUp(self):
    self._sas, self._sas_admin = sas.GetTestingSas()
    self._sas_admin.Reset()

  def tearDown(self):
    pass

  @winnforum_testcase
  def test_WINNF_FT_S_REG_1(self):
    """Array Multi-Step registration for CBSDs (Cat A and B).

    The response should be SUCCESS.
    """

    # Load Devices
    device_a = json.load(
        open(os.path.join('testcases', 'testdata', 'device_a.json')))
    device_b = json.load(
        open(os.path.join('testcases', 'testdata', 'device_b.json')))
    device_c = json.load(
        open(os.path.join('testcases', 'testdata', 'device_c.json')))

    # Pre-load conditionals
    conditionals_a = {
        'cbsdCategory': device_a['cbsdCategory'],
        'fccId': device_a['fccId'],
        'cbsdSerialNumber': device_a['cbsdSerialNumber'],
        'airInterface': device_a['airInterface'],
        'installationParam': device_a['installationParam'],
        'measCapability': device_a['measCapability']
    }
    conditionals_b = {
        'cbsdCategory': device_b['cbsdCategory'],
        'fccId': device_b['fccId'],
        'cbsdSerialNumber': device_b['cbsdSerialNumber'],
        'airInterface': device_b['airInterface'],
        'installationParam': device_b['installationParam'],
        'measCapability': device_b['measCapability']
    }
    conditionals_c = {
        'cbsdCategory': device_c['cbsdCategory'],
        'fccId': device_c['fccId'],
        'cbsdSerialNumber': device_c['cbsdSerialNumber'],
        'airInterface': device_c['airInterface'],
        'installationParam': device_c['installationParam'],
        'measCapability': device_c['measCapability']
    }
    conditionals = {
        'registrationData': [conditionals_a, conditionals_b, conditionals_c]
    }

    # Inject FCC IDs
    self._sas_admin.InjectFccId({'fccId': device_a['fccId']})
    self._sas_admin.InjectFccId({'fccId': device_b['fccId']})
    self._sas_admin.InjectFccId({'fccId': device_c['fccId']})

    # Inject User IDs
    self._sas_admin.InjectUserId({'userId': device_a['userId']})
    self._sas_admin.InjectUserId({'userId': device_b['userId']})
    self._sas_admin.InjectUserId({'userId': device_c['userId']})

    self._sas_admin.PreloadRegistrationData(conditionals)

    # Remove conditionals from registration
    del device_a['cbsdCategory']
    del device_a['airInterface']
    del device_a['installationParam']
    del device_a['measCapability']
    del device_b['cbsdCategory']
    del device_b['airInterface']
    del device_b['installationParam']
    del device_b['measCapability']
    del device_c['cbsdCategory']
    del device_c['airInterface']
    del device_c['installationParam']
    del device_c['measCapability']

    # Register the devices
    devices = [device_a, device_b, device_c]
    request = {'registrationRequest': devices}
    response = self._sas.Registration(request)
    # Check registration response
    for x in range(0, 3):
      self.assertTrue('cbsdId' in response['registrationResponse'][x])
      self.assertEqual(
          response['registrationResponse'][x]['response']['responseCode'], 0)

  @winnforum_testcase
  def test_WINNF_FT_S_REG_5(self):
    """Missing Required parameters in Array Registration request.

    The response should be MISSING_PARAM 102.
    """

    # Load Devices
    device_a = json.load(
        open(os.path.join('testcases', 'testdata', 'device_a.json')))
    device_c = json.load(
        open(os.path.join('testcases', 'testdata', 'device_c.json')))
    device_e = json.load(
        open(os.path.join('testcases', 'testdata', 'device_e.json')))
    device_f = json.load(
        open(os.path.join('testcases', 'testdata', 'device_f.json')))

    # Pre-load conditionals
    conditionals_a = {
        'cbsdCategory': device_a['cbsdCategory'],
        'fccId': device_a['fccId'],
        'cbsdSerialNumber': device_a['cbsdSerialNumber'],
        'airInterface': device_a['airInterface'],
        'installationParam': device_a['installationParam'],
        'measCapability': device_a['measCapability']
    }
    conditionals_c = {
        'cbsdCategory': device_c['cbsdCategory'],
        'fccId': device_c['fccId'],
        'cbsdSerialNumber': device_c['cbsdSerialNumber'],
        'airInterface': device_c['airInterface'],
        'installationParam': device_c['installationParam'],
        'measCapability': device_c['measCapability']
    }
    conditionals_e = {
        'cbsdCategory': device_e['cbsdCategory'],
        'fccId': device_e['fccId'],
        'cbsdSerialNumber': device_e['cbsdSerialNumber'],
        'airInterface': device_e['airInterface'],
        'installationParam': device_e['installationParam'],
        'measCapability': device_e['measCapability']
    }
    conditionals_f = {
        'cbsdCategory': device_f['cbsdCategory'],
        'fccId': device_f['fccId'],
        'cbsdSerialNumber': device_f['cbsdSerialNumber'],
        'airInterface': device_f['airInterface'],
        'installationParam': device_f['installationParam'],
        'measCapability': device_f['measCapability']
    }
    conditionals = {
        'registrationData': [
            conditionals_a, conditionals_c, conditionals_e, conditionals_f
        ]
    }

    # Inject FCC IDs
    self._sas_admin.InjectFccId({'fccId': device_a['fccId']})
    self._sas_admin.InjectFccId({'fccId': device_c['fccId']})
    self._sas_admin.InjectFccId({'fccId': device_e['fccId']})
    self._sas_admin.InjectFccId({'fccId': device_f['fccId']})

    # Inject User IDs
    self._sas_admin.InjectUserId({'userId': device_a['userId']})
    self._sas_admin.InjectUserId({'userId': device_c['userId']})
    self._sas_admin.InjectUserId({'userId': device_e['userId']})
    self._sas_admin.InjectUserId({'userId': device_f['userId']})

    self._sas_admin.PreloadRegistrationData(conditionals)

    # Remove conditionals from registration
    del device_a['cbsdCategory']
    del device_a['airInterface']
    del device_a['installationParam']
    del device_a['measCapability']
    del device_c['cbsdCategory']
    del device_c['airInterface']
    del device_c['installationParam']
    del device_c['measCapability']
    del device_e['cbsdCategory']
    del device_e['airInterface']
    del device_e['installationParam']
    del device_e['measCapability']
    del device_f['cbsdCategory']
    del device_f['airInterface']
    del device_f['installationParam']
    del device_f['measCapability']

    # Device 2 missing cbsdSerialNumber
    del device_c['cbsdSerialNumber']

    # Device 3 missing fccId
    del device_e['fccId']

    # Device 4 missing userId
    del device_f['userId']

    # Register devices
    devices = [device_a, device_c, device_e, device_f]
    request = {'registrationRequest': devices}
    response = self._sas.Registration(request)
    # Check registration response
    self.assertTrue('cbsdId' in response['registrationResponse'][0])
    self.assertEqual(
        response['registrationResponse'][0]['response']['responseCode'], 0)
    for x in range(1, 4):
      self.assertEqual(
          response['registrationResponse'][x]['response']['responseCode'], 102)

  @winnforum_testcase
  def test_WINNF_FT_S_REG_9(self):
    """Blacklisted CBSD in Array Registration request (responseCode 101).

    The response should be BLACKLISTED 101.
    """

    # Load Devices
    device_a = json.load(
        open(os.path.join('testcases', 'testdata', 'device_a.json')))
    device_c = json.load(
        open(os.path.join('testcases', 'testdata', 'device_c.json')))
    device_e = json.load(
        open(os.path.join('testcases', 'testdata', 'device_e.json')))
    devices = [device_a, device_c, device_e]
    for device in devices:
      self._sas_admin.InjectFccId({'fccId': device['fccId']})
      self._sas_admin.InjectUserId({'userId': device['userId']})
    request = {'registrationRequest': devices}
    response = self._sas.Registration(request)['registrationResponse']
    # Check registration response
    for resp in response:
      self.assertTrue('cbsdId' in resp)
      self.assertEqual(resp['response']['responseCode'], 0)
    del request, response

    # Blacklist the third device
    self._sas_admin.BlacklistByFccId({'fccId': device_e['fccId']})

    # Re-register the devices
    request = {'registrationRequest': devices}
    response = self._sas.Registration(request)['registrationResponse']

    # Check registration response
    self.assertEqual(len(response), len(devices))
    for resp in response[:2]:
      self.assertEqual(resp['response']['responseCode'], 0)
      self.assertTrue('cbsdId' in resp)
    self.assertEqual(response[2]['response']['responseCode'], 101)

  @winnforum_testcase
  def test_WINNF_FT_S_REG_10(self):
    """Unsupported SAS protocol version in Array request (responseCode 100).

    The response should be FAILURE 100.
    """

    # Use higher than supported version
    self._sas._sas_version = 'v2.0'

    # Load Devices
    device_a = json.load(
        open(os.path.join('testcases', 'testdata', 'device_a.json')))
    device_c = json.load(
        open(os.path.join('testcases', 'testdata', 'device_c.json')))
    device_e = json.load(
        open(os.path.join('testcases', 'testdata', 'device_e.json')))
    devices = [device_a, device_c, device_e]
    for device in devices:
      self._sas_admin.InjectFccId({'fccId': device['fccId']})
      self._sas_admin.InjectUserId({'userId': device['userId']})
    request = {'registrationRequest': devices}
    try:
      response = self._sas.Registration(request)
      # Check response
      for resp in response['registrationResponse']:
        self.assertEqual(resp['response']['responseCode'], 100)
        self.assertFalse('cbsdId' in resp)
    except AssertionError as e:
      # Allow HTTP status 404
      self.assertEqual(e.args[0], 404)