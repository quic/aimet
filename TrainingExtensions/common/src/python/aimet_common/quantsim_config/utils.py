# /usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Utilities for parsing configurations from json config file """

import os
from typing import Union

# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_common.quantsim_config.json_config_importer import ConfigDictKeys, ConfigDictType, JsonConfigImporter

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


def get_configs(config_file: Union[str, None]) -> ConfigDictType:
    """
    Import JSON config file and return configs as dictionary.
    :param config_file: Config file.
    :return: Dictionary containing configurations.
    """
    if not config_file:
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_config.json')
        logger.info('No config file provided, defaulting to config file at %s', config_file)
    configs = JsonConfigImporter.import_json_config_file(config_file)
    return configs


def get_unsigned_symmetric_flag(configs: ConfigDictType) -> bool:
    """
    Returns unsigned symmetric flag if it is set in config file else returns True.
    :return: Unsigned symmetric flag.
    """
    default_unsigned_symmetric = True
    default_configs = configs[ConfigDictKeys.DEFAULTS]
    if ConfigDictKeys.UNSIGNED_SYMMETRIC in default_configs:
        unsigned_symmetric = default_configs[ConfigDictKeys.UNSIGNED_SYMMETRIC]
    else:
        unsigned_symmetric = default_unsigned_symmetric
    return unsigned_symmetric


def get_strict_symmetric_flag(configs: ConfigDictType) -> bool:
    """
    Returns strict symmetric flag if it is set in config file else returns False.
    :return: Strict symmetric flag.
    """
    default_strict_symmetric = False
    default_configs = configs[ConfigDictKeys.DEFAULTS]
    if ConfigDictKeys.STRICT_SYMMETRIC in default_configs:
        strict_symmetric = default_configs[ConfigDictKeys.STRICT_SYMMETRIC]
    else:
        strict_symmetric = default_strict_symmetric
    return strict_symmetric
