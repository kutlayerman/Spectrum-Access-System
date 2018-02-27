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
from OpenSSL import SSL
from fake_crl_server import FakeCrlServerTestHarness
import security_testcase
from util import winnforum_testcase, configurable_testcase, writeConfig, loadConfig,\
countdown

SAS_CERT = os.path.join('certs', 'sas.cert')
SAS_KEY = os.path.join('certs', 'sas.key')

class SasToSasSecurityTestcase(security_testcase.SecurityTestCase):
  # Tests changing the SAS UUT state must explicitly call the SasReset().

  def generate_SSS_6_default_config(self, filename):
    """Generates the WinnForum configuration for SSS_6"""
    # Create the actual config for sas cert/key path

    config = {
        'sasCert': self.getCertFilename("unrecognized_sas.cert"),
        'sasKey': self.getCertFilename("unrecognized_sas.key")
    }
    writeConfig(filename, config)

  @configurable_testcase(generate_SSS_6_default_config)
  def test_WINNF_FT_S_SSS_6(self, config_filename):
    """Unrecognized root of trust certificate presented during registration.

    Checks that SAS UUT response with fatal alert with unknown_ca.
    """
    config = loadConfig(config_filename)
    self.assertTlsHandshakeFailure(client_cert=config['sasCert'],
                                   client_key=config['sasKey'])

  def generate_SSS_7_default_config(self, filename):
    """Generates the WinnForum configuration for SSS_7"""
    # Create the actual config for sas cert/key path

    config = {
        'sasCert': self.getCertFilename("corrupted_sas.cert"),
        'sasKey': self.getCertFilename("corrupted_sas.key")
    }
    writeConfig(filename, config)

  @configurable_testcase(generate_SSS_7_default_config)
  def test_WINNF_FT_S_SSS_7(self, config_filename):
    """Corrupted certificate presented during registration.

    Checks that SAS UUT response with fatal alert message.
    """
    config = loadConfig(config_filename)
    self.assertTlsHandshakeFailure(client_cert=config['sasCert'],
                                   client_key=config['sasKey'])

  def generate_SSS_8_default_config(self, filename):
    """Generates the WinnForum configuration for SSS_8"""
    # Create the actual config for sas cert/key path

    config = {
        'sasCert': self.getCertFilename("self_signed_sas.cert"),
        'sasKey': self.getCertFilename("sas.key")
    }
    writeConfig(filename, config)

  @configurable_testcase(generate_SSS_8_default_config)
  def test_WINNF_FT_S_SSS_8(self, config_filename):
    """Self-signed certificate presented during registration.

    Checks that SAS UUT response with fatal alert message.
    """
    config = loadConfig(config_filename)
    self.assertTlsHandshakeFailure(client_cert=config['sasCert'],
                                   client_key=config['sasKey'])

  def generate_SSS_9_default_config(self, filename):
    """Generates the WinnForum configuration for SSS_9"""
    # Create the actual config for domain proxy cert/key path

    config = {
        'sasCert': self.getCertFilename("non_cbrs_signed_sas.cert"),
        'sasKey': self.getCertFilename("non_cbrs_signed_sas.key")
    }
    writeConfig(filename, config)

  @configurable_testcase(generate_SSS_9_default_config)
  def test_WINNF_FT_S_SSS_9(self, config_filename):
    """Non-CBRS trust root signed certificate presented during registration.

    Checks that SAS UUT response with fatal alert message.
    """
    config = loadConfig(config_filename)
    self.assertTlsHandshakeFailure(client_cert=config['sasCert'],
                                   client_key=config['sasKey'])

  def generate_SSS_11_default_config(self, filename):
    """Generates the WinnForum configuration for SSS_11. """
    # Create the configuration for blacklisted sas cert/key,wait timer & fake CRL Server information

    config = {
      'sasCert': self.getCertFilename("blacklisted_sas.cert"),
      'sasKey' : self.getCertFilename("blacklisted_sas.key"),
      'crlServer' : {
                    'hostName':'localhost',
                    'port':'9006',
                    },
      'crlUrl' : "ca.crl",
      'crlFile': self.getCertFilename("ca_crl.crl"),
      'wait_timer':60
      }
    writeConfig(filename, config)

  @configurable_testcase(generate_SSS_11_default_config)
  def test_WINNF_FT_S_SSS_11(self,config_filename):
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
    self.assertTlsHandshakeFailure(config['sasCert'], config['sasKey'])
    logging.info("TLS handshake failed as the sas certificate has blacklisted")

    # Stop the Fake CRL Server
    self.fake_crl_server.stopServer()

  def generate_SSS_12_default_config(self, filename):
    """Generates the WinnForum configuration for SSS.12"""
    # Create the actual config for domain proxy cert/key path

    config = {
        'sasCert': self.getCertFilename("sas_expired.cert"),
        'sasKey': self.getCertFilename("sas_expired.key")
    }
    writeConfig(filename, config)

  @configurable_testcase(generate_SSS_12_default_config)
  def test_WINNF_FT_S_SSS_12(self, config_filename):
    """Expired certificate presented during registration.

    Checks that SAS UUT response with fatal alert message.
    """
    config = loadConfig(config_filename)
    self.assertTlsHandshakeFailure(client_cert=config['sasCert'],
                                   client_key=config['sasKey'])

  @winnforum_testcase
  def test_WINNF_FT_S_SSS_13(self):
    """ Disallowed TLS method attempted during registration.

    Checks that SAS UUT response with fatal alert message.
    """
    self.assertTlsHandshakeFailure(SAS_CERT, SAS_KEY, ssl_method=SSL.TLSv1_1_METHOD)

  @winnforum_testcase
  def test_WINNF_FT_S_SSS_14(self):
    """Invalid ciphersuite presented during registration.

    Checks that SAS UUT response with fatal alert message.
    """
    self.assertTlsHandshakeFailure(SAS_CERT, SAS_KEY, ciphers='ECDHE-RSA-AES256-GCM-SHA384')

  def generate_SSS_16_default_config(self, filename):
    """Generates the WinnForum configuration for SSS_16. """
    # Create the configuration for client cert/key,wait timer & fake CRL Server information

    config = {
      'sasCert': self.getCertFilename("sas.cert"),
      'sasKey' : self.getCertFilename("sas.key"),
      'crlServer' : {
                    'hostName':'localhost',
                    'port':'9006',
                    },
      'crlUrl' : "ca.crl",
      'crlFile': self.getCertFilename("ca_crl.crl"),
      'wait_timer':60
      }
    writeConfig(filename, config)

  @configurable_testcase(generate_SSS_16_default_config)
  def test_WINNF_FT_S_SSS_16(self,config_filename):
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
    self.assertTlsHandshakeFailure(config['sasCert'], config['sasKey'])
    logging.info("TLS handshake failed as the CA certificate has been revoked")

    # Stop the Fake CRL Server
    self.fake_crl_server.stopServer()
