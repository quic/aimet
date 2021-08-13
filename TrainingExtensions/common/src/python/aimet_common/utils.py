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
""" Utility classes and functions that are used by NightlyTests files as well as
    common to both PyTorch and TensorFlow. """

import math
import os
import logging
import logging.config
import logging.handlers
import signal
import socket
import subprocess
import time
from enum import Enum
import json
import yaml

try:
    # The build system updates Product, Version and Feature set information in the package_info file.
    from aimet_common.package_info import Product, Version_Info, Postfix

except ImportError:
    # Default values for Product, Version and Feature set information.
    Product = 'AIMET'
    Version_Info = ''
    Postfix = ''


class ModelApi(Enum):
    """ Enum differentiating between Pytorch or Tensorflow """
    pytorch = 0
    tensorflow = 1


class SingletonType(type):
    """ SingletonType is used as a metaclass by other classes for which only one instance must be created.

    A metaclass inherits from "type' and it's instances are other classes.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """ This function overrides the behavior of type's __call__ function.

        The overriding behavior is needed  so that only one instance of the derived
        class is created. The argument cls is a class variable (similar to self for instances).

        Using AimetLogger  class as an example, when AimetLogger() is called, SingletonType
        (the metaclass) class's __call__ is called which in turn calls AimetLogger's __call__
        creating an instance of AimetLogger. The creation happens only once, making
        aimetLooger a singleton.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class AimetLogger(metaclass=SingletonType):
    """ The aimet Logger class. Multiple Area Loggers have been defined.
    Each Area Logger could be set at a different logging level. """
    _logger = None

    class LogAreas(Enum):
        """ Defines the LogAreas used in aimet. """
        Quant = 'Quant'
        Svd = 'Svd'
        Test = 'Test'
        Utils = 'Utils'
        CompRatioSelect = 'CompRatioSelect'
        ChannelPruning = 'ChannelPruning'
        Winnow = 'Winnow'
        ConnectedGraph = 'ConnectedGraph'
        CrosslayerEqualization = 'CrossLayerEqualization'
        MixedPrecision = 'MixedPrecision'

    def __init__(self):
        self._logger = logging.getLogger()

        dir_name = os.path.dirname(__file__)
        rel_path = "default_logging_config.json"
        abs_file_path = os.path.join(dir_name, rel_path)

        with open(abs_file_path, encoding='utf-8') as logging_configuration_file:
            try:
                config_dict = json.loads(logging_configuration_file.read())
            except:
                raise ValueError("Logging configuration file: default_logging_config.json contains invalid format")

        logging.config.dictConfig(config_dict)

        # Validate JSON  file default_logging_config.json for correct Logging Areas
        #TODO This results in a pylint error: Instance of 'RootLogger' has no 'loggerDict' member.
        # Need to fix this issue and then remove the pylint disablement.
        configured_items = list(logging.root.manager.loggerDict.items()) # pylint: disable=no-member

        log_areas_list = list()
        for x in AimetLogger.LogAreas:
            log_areas_list.append(x.value)

        configured_areas_list = list()
        for name, _ in configured_items:
            configured_areas_list.append(name)

        for area in log_areas_list:
            if area not in configured_areas_list:
                raise ValueError(" ERROR: LogArea: {} NOT configured".format(area))

        log_package_info()

    @staticmethod
    def get_area_logger(area):
        """ Returns a specific Area logger. """
        AimetLogger()
        area_logger = logging.getLogger(area.value)
        return area_logger

    @staticmethod
    def set_area_logger_level(area, level):
        """ Sets a logging level for a single area logger. """
        area_logger = logging.getLogger(area.value)
        area_logger.setLevel(level)

    @staticmethod
    def set_level_for_all_areas(level):
        """ Sets the same logging level for all area debuggers. """
        for area in AimetLogger.LogAreas:
            AimetLogger.set_area_logger_level(area, level)


def round_up_to_multiplicity(multiplicity: int, num: int, max_allowable_num: int):
    """
    Function to round a number to the nearest multiplicity given the multiplicity
    :param multiplicity: multiplicity for rounding
    :param num: input number to be rounded
    :param max_allowable_num: maximum value for num allowed
    :return: number rounded up to nearest multiplicity
    """
    larger_multiple = math.ceil(float(num) / float(multiplicity)) * multiplicity
    if larger_multiple >= max_allowable_num:
        return max_allowable_num
    return int(larger_multiple)


def round_down_to_multiplicity(multiplicity: int, num: int):
    """
    Function to round a number to the nearest multiplicity given the multiplicity
    :param multiplicity: multiplicity for rounding
    :param num: input number to be rounded
    :return: number rounded down to nearest multiplicity
    """
    if num - multiplicity <= 0:
        return num

    if num % multiplicity == 0:
        num = num - 1
    lower_multiple = math.floor(float(num) / float(multiplicity)) * multiplicity
    return int(lower_multiple)


# Depending on pytorch or tensorflow, the ordering of dimensions in tensor/product shapes will be different.
# In pytorch, the number of channels is always index 1
# In tensorflow, the number of channels is always the last dimension in the shape
api_channel_index_dict = {ModelApi.pytorch: 1, ModelApi.tensorflow: -1}


def kill_process_with_name_and_port_number(name: str, port_number: int):
    """ Kill a process that is associated with a port number displayed by the command: ps -x """

    logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)
    p = subprocess.Popen(['ps', '-x'], stdout=subprocess.PIPE)
    out, _ = p.communicate()

    for line in out.splitlines():
        str_line = line.decode()
        port_num_str = str(port_number)
        if name in str_line and '--port=' + port_num_str in str_line:
            pid = int(line.split(None, 1)[0])
            logger.info("Killing Bokeh server with process id: %s", format(pid))
            os.kill(pid, signal.SIGKILL)
            break


def start_bokeh_server_session(port: int):
    """
    start a bokeh server programmatically. Used for testing purposes.
    :param port: port number
    :return: Returns the Bokeh Server URL and the process object used to create the child server process
    """

    host_name = socket.gethostname()
    bokeh_serve_command = "bokeh serve  --allow-websocket-origin=" + \
                          host_name + ":" + str(port) + " --allow-websocket-origin=localhost:" + str(port) + " --port=" + str(port) + " &"
    process = subprocess.Popen(bokeh_serve_command,  # pylint: disable=subprocess-popen-preexec-fn
                               shell=True,
                               preexec_fn=os.setsid)
    url = "http://" + host_name + ":" + str(port)
    # Doesn't allow document to be added to server unless there is some sort of wait time.
    time.sleep(4)
    return url, process


def log_package_info():
    """
    Log the Product, Version and Postfix.
    :return:
    """

    # The Product is always a non-empty string.
    if Version_Info != '' and Postfix != '':
        # Log Product-Version-Postfix
        logging.info("%s-%s-%s", Product, Version_Info, Postfix)
    elif Version_Info != '' and Postfix == '':
        # Log Product-Version
        logging.info("%s-%s", Product, Version_Info)
    else:
        # If Version is empty, the Postfix is not logged.
        # Log Product.
        logging.info("%s", Product)

def save_hist_yaml(encoding_file_path: str, encodings_dict: dict):
    """
    Function which saves encoding in YAML and JSON file format
    :param encoding_file_path: file name to use to generate the yaml and json file
    :param encodings_dict: dictionary containing the encoding
    """
    encoding_file_path_yaml = encoding_file_path + '.yaml'
    with open(encoding_file_path_yaml, 'w') as encoding_fp_yaml:
        yaml.dump(encodings_dict, encoding_fp_yaml, default_flow_style=None, allow_unicode=True)


def save_json_yaml(encoding_file_path: str, encodings_dict: dict):
    """
    Function which saves encoding in YAML and JSON file format
    :param encoding_file_path: file name to use to generate the yaml and json file
    :param encodings_dict: dictionary containing the encoding
    """
    encoding_file_path_json = encoding_file_path
    encoding_file_path_yaml = encoding_file_path + '.yaml'
    with open(encoding_file_path_json, 'w') as encoding_fp_json:
        json.dump(encodings_dict, encoding_fp_json, sort_keys=True, indent=4)
    with open(encoding_file_path_yaml, 'w') as encoding_fp_yaml:
        yaml.dump(encodings_dict, encoding_fp_yaml, default_flow_style=False, allow_unicode=True)
