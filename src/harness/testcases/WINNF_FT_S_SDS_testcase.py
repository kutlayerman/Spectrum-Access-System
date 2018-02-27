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
import logging
import os
import json
import security_testcase
from fake_crl_server import FakeCrlServerTestHarness
from OpenSSL import SSL
from util import winnforum_testcase, configurable_testcase, writeConfig, loadConfig,\
countdown

DOMAIN_PROXY_CERT = os.path.join('certs', 'domain_proxy.cert')
DOMAIN_PROXY_KEY = os.path.join('certs', 'domain_proxy.key')


class SasDomainProxySecurityTestcase(security_testcase.SecurityTestCase):
  # Tests changing the SAS UUT state must explicitly call the SasReset().

  @winnforum_testcase
  def test_WINNF_FT_S_SDS_1(self):
    """New registration with TLS_RSA_WITH_AES_128_GCM_SHA256 cipher.

    Checks that SAS UUT response satisfy cipher security conditions.
    Checks that a DP registration with this configuration succeed.
    """
    self.doTestCipher('AES128-GCM-SHA256', client_cert=DOMAIN_PROXY_CERT,
                      client_key=DOMAIN_PROXY_KEY)

  @winnforum_testcase
  def test_WINNF_FT_S_SDS_2(self):
    """New registration with TLS_RSA_WITH_AES_256_GCM_SHA384 cipher.

    Checks that SAS UUT response satisfy specific security conditions.
    Checks that a DP registration with this configuration succeed.
    """
    self.doTestCipher('AES256-GCM-SHA384', client_cert=DOMAIN_PROXY_CERT,
                      client_key=DOMAIN_PROXY_KEY)

  @winnforum_testcase
  def test_WINNF_FT_S_SDS_3(self):
    """New registration with TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256 cipher.

    Checks that SAS UUT response satisfy specific security conditions.
    Checks that a DP registration with this configuration succeed.
    """
    self.doTestCipher('ECDHE-ECDSA-AES128-GCM-SHA256',
                      client_cert=DOMAIN_PROXY_CERT,
                      client_key=DOMAIN_PROXY_KEY)

  @winnforum_testcase
  def test_WINNF_FT_S_SDS_4(self):
    """New registration with TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384 cipher.

    Checks that SAS UUT response satisfy specific security conditions.
    Checks that a DP registration with this configuration succeed.
    """
    self.doTestCipher('ECDHE-ECDSA-AES256-GCM-SHA384',
                      client_cert=DOMAIN_PROXY_CERT,
                      client_key=DOMAIN_PROXY_KEY)

  @winnforum_testcase
  def test_WINNF_FT_S_SDS_5(self):
    """New registration with TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256 cipher.

    Checks that SAS UUT response satisfy specific security conditions.
    Checks that a DP registration with this configuration succeed.
    """
    self.doTestCipher('ECDHE-RSA-AES128-GCM-SHA256',
                      client_cert=DOMAIN_PROXY_CERT,
                      client_key=DOMAIN_PROXY_KEY)

  def generate_SDS_6_default_config(self, filename):
    """Generates the WinnForum configuration for SDS_6"""
    # Create the actual config for domain proxy cert/key path

    config = {
      'domainProxyCert': self.getCertFilename("unrecognized_domain_proxy.cert"),
      'domainProxyKey': self.getCertFilename("unrecognized_domain_proxy.key")
    }
    writeConfig(filename, config)

  @configurable_testcase(generate_SDS_6_default_config)
  def test_WINNF_FT_S_SDS_6(self, config_filename):
    """Unrecognized root of trust certificate presented during registration.

    Checks that SAS UUT response with fatal alert with unknown_ca.
    """
    config = loadConfig(config_filename)
    self.assertTlsHandshakeFailure(client_cert=config['domainProxyCert'],
                                   client_key=config['domainProxyKey'])

  def generate_SDS_7_default_config(self, filename):
    """Generates the WinnForum configuration for SDS_7"""
    # Create the actual config for domain proxy cert/key path

    config = {
      'domainProxyCert': self.getCertFilename("corrupted_domain_proxy.cert"),
      'domainProxyKey': self.getCertFilename("corrupted_domain_proxy.key")
    }
    writeConfig(filename, config)

  @configurable_testcase(generate_SDS_7_default_config)
  def test_WINNF_FT_S_SDS_7(self,config_filename):
    """Corrupted certificate presented during registration.

    Checks that SAS UUT response with fatal alert message.
    """
    config = loadConfig(config_filename)
    self.assertTlsHandshakeFailure(client_cert=config['domainProxyCert'],
                                   client_key=config['domainProxyKey'])

  def generate_SDS_8_default_config(self, filename):
    """Generates the WinnForum configuration for SDS_8"""
    # Create the actual config for domain proxy cert/key path

    config = {
      'domainProxyCert': self.getCertFilename("self_signed_domain_proxy.cert"),
      'domainProxyKey': self.getCertFilename("domain_proxy.key")
    }
    writeConfig(filename, config)

  @configurable_testcase(generate_SDS_8_default_config)
  def test_WINNF_FT_S_SDS_8(self,config_filename):
    """Self-signed certificate presented during registration.

    Checks that SAS UUT response with fatal alert message.
    """
    config = loadConfig(config_filename)
    self.assertTlsHandshakeFailure(client_cert=config['domainProxyCert'],
                                   client_key=config['domainProxyKey'])

  def generate_SDS_9_default_config(self, filename):
    """Generates the WinnForum configuration for SDS_9"""
    # Create the actual config for domain proxy cert/key path

    config = {
      'domainProxyCert': self.getCertFilename("non_cbrs_signed_domain_proxy.cert"),
      'domainProxyKey': self.getCertFilename("non_cbrs_signed_domain_proxy.key")
    }
    writeConfig(filename, config)

  @configurable_testcase(generate_SDS_9_default_config)
  def test_WINNF_FT_S_SDS_9(self,config_filename):
    """Non-CBRS trust root signed certificate presented during registration.

    Checks that SAS UUT response with fatal alert message.
    """
    config = loadConfig(config_filename)
    self.assertTlsHandshakeFailure(client_cert=config['domainProxyCert'],
                                   client_key=config['domainProxyKey'])

  def generate_SDS_10_default_config(self, filename):
    """Generates the WinnForum configuration for SDS_10. """
    # Create the actual config for domain proxy cert/key path

    config = {
      'domainProxyCert': self.getCertFilename("wrong_type_domain_proxy.cert"),
      'domainProxyKey': self.getCertFilename("server.key")

    }
    writeConfig(filename, config)

  @configurable_testcase(generate_SDS_10_default_config)
  def test_WINNF_FT_S_SDS_10(self,config_filename):
    """Certificate of wrong type presented during registration.

    Checks that SAS UUT response with fatal alert message.
    """
    config = loadConfig(config_filename)
    try:
      self.assertTlsHandshakeFailure(client_cert=config['domainProxyCert'],
                                     client_key=config['domainProxyKey'])
    except AssertionError as e:
      self.SasReset()
      # Load Devices
      device_a = json.load(
        open(os.path.join('testcases', 'testdata', 'device_a.json')))
      # Register the devices
      devices = [device_a]
      request = {'registrationRequest': devices}
      response = self._sas.Registration(request, ssl_cert=config['domainProxyCert'],
                                        ssl_key=config['domainProxyKey'])['registrationResponse']
      # Check Registration Response
      self.assertEqual(response[0]['response']['responseCode'], 104)

  def generate_SDS_11_default_config(self, filename):
    """Generates the WinnForum configuration for SDS_11. """
    # Create the configuration for blacklisted sas cert/key,wait timer & fake CRL Server information

    config = {
      'domainProxyCert': self.getCertFilename("blacklisted_domain_proxy.cert"),
      'domainProxyKey' : self.getCertFilename("blacklisted_domain_proxy.key"),
      'crlServer' : {
                    'hostName':'localhost',
                    'port':'9006',
                    },
      'crlUrl' : "ca.crl",
      'crlFile': self.getCertFilename("ca_crl.crl"),
      'wait_timer':60
      }
    writeConfig(filename, config)

  @configurable_testcase(generate_SDS_11_default_config)
  def test_WINNF_FT_S_SDS_11(self,config_filename):
    """Blacklisted certificate presented during registration..
       Checks that SAS UUT response with fatal alert message.
    """
    # Read the configuration
    config = loadConfig(config_filename)

    # Create the Fake CRL Server
    self.fake_crl_server = FakeCrlServerTestHarness(
                           config['crlServer']['hostName'],
                           config['crlServer']['port'],
                           config['crlUrl'],
                           config['crlFile'])

    # Start the Fake CRL Server
    self.fake_crl_server.start()

    logging.info("Waiting for %s secs to allow the UUT to pull the revoked certificate "
                 "list from the fake CRL server " % config['wait_timer'])

    # Wait for the timer
    countdown(config['wait_timer'])

    # Tls handshake fails
    self.assertTlsHandshakeFailure(config['domainProxyCert'], config['domainProxyKey'])
    logging.info("TLS handshake failed as the domain proxy certificate has blacklisted")

    # Stop the Fake CRL Server
    self.fake_crl_server.stopServer()

  def generate_SDS_16_default_config(self, filename):
    """Generates the WinnForum configuration for SDS_16. """
    # Create the configuration for client cert/key,wait timer & fake CRL Server information

    config = {
      'domainProxyCert': self.getCertFilename("domain_proxy.cert"),
      'domainProxyKey' : self.getCertFilename("domain_proxy.key"),
      'crlServer' : {
                    'hostName':'localhost',
                    'port':'9006',
                    },
      'crlUrl' : "ca.crl",
      'crlFile': self.getCertFilename("ca_crl.crl"),
      'wait_timer':60
      }
    writeConfig(filename, config)

  @configurable_testcase(generate_SDS_16_default_config)
  def test_WINNF_FT_S_SDS_16(self,config_filename):
    """Certificate signed by a revoked CA presented during registration.
       Checks that SAS UUT response with fatal alert message.
    """
    # Read the configuration
    config = loadConfig(config_filename)

    # Create the Fake CRL Server
    self.fake_crl_server = FakeCrlServerTestHarness(
                           config['crlServer']['hostName'],
                           config['crlServer']['port'],
                           config['crlUrl'],
                           config['crlFile'])

    # Start the Fake CRL Server
    self.fake_crl_server.start()

    logging.info("Waiting for %s secs to allow the UUT to pull the revoked certificate "
                 "list from the fake CRL server " % config['wait_timer'])

    # Wait for the timer
    countdown(config['wait_timer'])

    # Tls handshake fails since CA is revoked
    self.assertTlsHandshakeFailure(config['domainProxyCert'], config['domainProxyKey'])
    logging.info("TLS handshake failed as the CA certificate has been revoked")

    # Stop the Fake CRL Server
    self.fake_crl_server.stopServer()

  def generate_SDS_12_default_config(self, filename):
    """Generates the WinnForum configuration for SDS.12"""
    # Create the actual config for domain proxy cert/key path

    config = {
        'domainProxyCert': self.getCertFilename("domain_proxy_expired.cert"),
        'domainProxyKey': self.getCertFilename("domain_proxy_expired.key")
    }
    writeConfig(filename, config)

  @configurable_testcase(generate_SDS_12_default_config)
  def test_WINNF_FT_S_SDS_12(self,config_filename):
    """Expired certificate presented during registration.

    Checks that SAS UUT response with fatal alert message.
    """
    config = loadConfig(config_filename)
    self.assertTlsHandshakeFailure(client_cert=config['domainProxyCert'],
                                   client_key=config['domainProxyKey'])

  @winnforum_testcase
  def test_WINNF_FT_S_SDS_13(self):
    """ Disallowed TLS method attempted during registration.

    Checks that SAS UUT response with fatal alert message.
    """
    self.assertTlsHandshakeFailure(DOMAIN_PROXY_CERT, DOMAIN_PROXY_KEY, ssl_method=SSL.TLSv1_1_METHOD)

  @winnforum_testcase
  def test_WINNF_FT_S_SDS_14(self):
    """Invalid ciphersuite presented during registration.

    Checks that SAS UUT response with fatal alert message.
    """
    self.assertTlsHandshakeFailure(DOMAIN_PROXY_CERT, DOMAIN_PROXY_KEY, ciphers='ECDHE-RSA-AES256-GCM-SHA384')


  def generate_SDS_15_default_config(self, filename):
    """ Generates the WinnForum configuration for SDS.15 """
    # Create the actual config for domain proxy cert/key path

    config = {
        'domainProxyCert': self.getCertFilename("domain_proxy_inapplicable.cert"),
        'domainProxyKey': self.getCertFilename("domain_proxy.key")
    }
    writeConfig(filename, config)

  @configurable_testcase(generate_SDS_15_default_config)
  def test_WINNF_FT_S_SDS_15(self,config_filename):
    """Certificate with inapplicable fields presented during registration.

    Checks that SAS UUT response with tls connection succedded and response should be 104.
    """
    config = loadConfig(config_filename)

    # Load the keys/certs and check that TLS handshake is valid
    device_a_cert = config['domainProxyCert']
    device_a_key = config['domainProxyKey']
    self.assertTlsHandshakeSucceed(self._sas_admin._base_url, ['AES128-GCM-SHA256'], device_a_cert, device_a_key)

    # Load device and inject fccId and userId
    device_a = json.load(open(os.path.join('testcases', 'testdata', 'device_a.json')))
    self.SasReset()
    self._sas_admin.InjectFccId({'fccId': device_a['fccId']})
    self._sas_admin.InjectUserId({'userId': device_a['userId']})

    # Send registration Request with certs(inapplicable fields) to SAS UUT
    request = {'registrationRequest': [device_a]}
    response = self._sas.Registration(request,ssl_cert=device_a_cert,ssl_key=device_a_key)['registrationResponse'][0]

    # Check registration response
    self.assertEqual(response['response']['responseCode'],104)

  def generate_SDS_17_default_config(self, filename):
    """Generates the WinnForum configuration for SDS_17. """
    # Create the configuration for short lived invalid domain proxy cert/key,wait timer

    config = {
      'domainProxyCert': self.getCertFilename("short_lived_domain_proxy.cert"),
      'domainProxyKey' : self.getCertFilename("short_lived_domain_proxy.key"),
      'wait_timer': 300
      }
    writeConfig(filename, config)

  @configurable_testcase(generate_SDS_17_default_config)
  def test_WINNF_FT_S_SDS_17(self,config_filename):
    """Invalid certificate following an approved registration request
       Checks that SAS UUT response with fatal alert message.
    """
    # Read the configuration
    config = loadConfig(config_filename)

    # Successful Tls Handshake
    self.assertTlsHandshakeSucceed(self._sas_admin._base_url, ['AES128-GCM-SHA256'],
                                   config['domainProxyCert'], config['domainProxyKey'])
    logging.info("TLS Handshake Succeeded")

    # Load the device_a file
    device_a = json.load(
        open(os.path.join('testcases', 'testdata', 'device_a.json')))

    with security_testcase.CiphersOverload(self._sas, ['AES128-GCM-SHA256'],
                                           config['domainProxyCert'],
                                           config['domainProxyKey']):self.assertRegistered([device_a])

    logging.info("Waiting for %s secs so the certificate will be invalid" % config['wait_timer'])

    # Wait for the timer
    countdown(config['wait_timer'])

    # Tls handshake fails
    self.assertTlsHandshakeFailure(config['domainProxyCert'], config['domainProxyKey'])
    logging.info("TLS handshake failed as the domain proxy certificate is invalid")

  def generate_SDS_18_default_config(self, filename):
    """Generates the WinnForum configuration for SDS_18. """
    # Create the configuration for invalid domain proxy cert/key & wait timer

    config = {
      'domainProxyCert': self.getCertFilename("short_lived_domain_proxy.cert"),
      'domainProxyKey' : self.getCertFilename("short_lived_domain_proxy.key"),
      'wait_timer': 300
      }
    writeConfig(filename, config)

  @configurable_testcase(generate_SDS_18_default_config)
  def test_WINNF_FT_S_SDS_18(self,config_filename):
    """Invalid certificate following an approved grant request
       Checks that SAS UUT response with fatal alert message.
    """
    # Read the configuration
    config = loadConfig(config_filename)

    # Successful Tls Handshake
    self.assertTlsHandshakeSucceed(self._sas_admin._base_url, ['AES128-GCM-SHA256'],
                                   config['domainProxyCert'], config['domainProxyKey'])
    logging.info("TLS Handshake Succeeded")

    # Load the device_a file
    device_a = json.load(
      open(os.path.join('testcases', 'testdata', 'device_a.json')))

    # Load the grant_0 file
    grant_0 = json.load(
      open(os.path.join('testcases', 'testdata', 'grant_0.json')))

    with security_testcase.CiphersOverload(self._sas, ['AES128-GCM-SHA256'],
                        config['domainProxyCert'],
                        config['domainProxyKey']):cbsd_ids,grant_ids = self.assertRegisteredAndGranted([device_a],[grant_0])

    logging.info("device_a is in Granted State")

    logging.info("Waiting for %s secs so the certificate will be invalid" % config['wait_timer'])

    # Wait for the timer
    countdown(config['wait_timer'])

    # Tls handshake fails
    self.assertTlsHandshakeFailure(config['domainProxyCert'], config['domainProxyKey'])
    logging.info("TLS handshake failed as the domain proxy certificate is invalid")

  def generate_SDS_19_default_config(self, filename):
    """Generates the WinnForum configuration for SDS_19. """
    # Create the configuration for invalid domain proxy cert/key & wait timer

    config = {
      'domainProxyCert': self.getCertFilename("short_lived_domain_proxy.cert"),
      'domainProxyKey' : self.getCertFilename("short_lived_domain_proxy.key"),
      'wait_timer': 300
      }
    writeConfig(filename, config)

  @configurable_testcase(generate_SDS_19_default_config)
  def test_WINNF_FT_S_SDS_19(self,config_filename):
    """Invalid certificate following an approved heartbeat request
       Checks that SAS UUT response with fatal alert message.
    """
    # Read the configuration
    config = loadConfig(config_filename)

    # Successful Tls Handshake
    self.assertTlsHandshakeSucceed(self._sas_admin._base_url, ['AES128-GCM-SHA256'],
                                   config['domainProxyCert'], config['domainProxyKey'])
    logging.info("TLS Handshake Succeeded")

    # Load the device_a file
    device_a = json.load(
      open(os.path.join('testcases', 'testdata', 'device_a.json')))

    # Load the grant_0 file
    grant_0 = json.load(
      open(os.path.join('testcases', 'testdata', 'grant_0.json')))

    with security_testcase.CiphersOverload(self._sas, ['AES128-GCM-SHA256'],
                        config['domainProxyCert'],
                        config['domainProxyKey']):cbsd_ids,grant_ids = self.assertRegisteredAndGranted([device_a],[grant_0])
    operation_states = [('GRANTED')]
    transfer_expiry_timers = self.assertHeartbeatsSuccessful(cbsd_ids,grant_ids,operation_states)

    logging.info("device_a is in HeartBeat Successful State")

    logging.info("Waiting for %s secs so the certificate will be invalid" % config['wait_timer'])

    # Wait for the timer
    countdown(config['wait_timer'])

    # Tls handshake fails
    self.assertTlsHandshakeFailure(config['domainProxyCert'], config['domainProxyKey'])
    logging.info("TLS handshake failed as the domain proxy certificate is invalid")


