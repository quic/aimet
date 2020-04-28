# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

import unittest
import logging

from aimet_common.utils import AimetLogger


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)

class UseLogger(unittest.TestCase):

    def test_log_areas(self):

        logger.info("            ")
        logger.info("Testing test_log_areas()")

        # Test the UTILS logger
        utils_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)
        utils_logger.debug("Testing Debug")
        utils_logger.info("Testing Info")
        utils_logger.warning("Testing Warning")
        utils_logger.error("Testing Error")
        utils_logger.critical("Testing Critical")
        utils_logger.critical("**************************************** \n")

        # Test the QUANT logger
        quant_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)
        quant_logger.debug("Testing Debug")
        quant_logger.info("Testing Info")
        quant_logger.warning("Testing Warning")
        quant_logger.error("Testing Error")
        quant_logger.critical("Testing Critical")
        quant_logger.critical("**************************************** \n")

        # Test the SVD logger
        svd_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)
        svd_logger.debug("Testing Debug")
        svd_logger.info("Testing Info")
        svd_logger.warning("Testing Warning")
        svd_logger.error("Testing Error")
        svd_logger.critical("Testing Critical")
        svd_logger.critical("**************************************** \n")

    def test_setting_log_level(self):

        logger.info("*** Testing test_setting_log_level() *** \n")
        svd_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)

        # The default logging level for SVD defined in default_logging_config.json is used.
        logger.info("Log at the default log level for SVD defined in default_logging_config.json")
        svd_logger.debug("Testing Debug")
        svd_logger.info("Testing Info")
        svd_logger.warning("Testing Warning")
        svd_logger.error("Testing Error")
        svd_logger.critical("Testing Critical")
        svd_logger.critical("****************************************\n")

        # Change the default log level for SVD.
        # Only CRITICAL level logs will be logged.
        logger.info("Change SVD area's logging level to Critical")
        AimetLogger.set_area_logger_level(AimetLogger.LogAreas.Svd, logging.CRITICAL)
        svd_logger.debug("Testing Debug")
        svd_logger.info("Testing Info")
        svd_logger.warning("Testing Warning")
        svd_logger.error("Testing Error")
        svd_logger.critical("Testing Critical")
        svd_logger.critical("****************************************\n")

        # Change the default log level for SVD.
        # All logs will be logged.
        logger.info("Change SVD area's logging level to Critical")
        AimetLogger.set_area_logger_level(AimetLogger.LogAreas.Svd, logging.DEBUG)
        svd_logger.debug("Testing Debug")
        svd_logger.info("Testing Info")
        svd_logger.warning("Testing Warning")
        svd_logger.error("Testing Error")
        svd_logger.critical("Testing Critical")
        svd_logger.critical("****************************************\n")



    def test_setting_log_level_for_all_areas(self):

        logger.info("*** test_setting_log_level_for_all_areas() ***\n")

        svd_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)
        quant_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)
        util_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)
        test_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)

        # The default logging level for all Log Areas defined in default_logging_config.json is used.
        logger.info("Log at the default log level for  all Log Areas defined in default_logging_config.json")
        svd_logger.debug("Testing Debug")
        svd_logger.info("Testing Info")

        quant_logger.warning("Testing Warning")
        quant_logger.error("Testing Error")

        util_logger.critical("Testing Critical")
        util_logger.info("Testing Info")

        test_logger.critical("Testing Critical")
        test_logger.critical("****************************************\n")

        # Change the default log level for all areas
        # Only CRITICAL level logs will be logged.
        logger.info("Change the logging level for all Log Areas to WARNING")
        AimetLogger.set_level_for_all_areas(logging.WARNING)

        svd_logger.debug("Testing Debug")
        svd_logger.info("Testing Info")

        quant_logger.warning("Testing Warning")
        quant_logger.error("Testing Error")

        util_logger.critical("Testing Critical")
        util_logger.info("Testing Info")

        test_logger.critical("Testing Critical")
        test_logger.critical("****************************************\n")

