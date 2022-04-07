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

from aimet_common.quantsim_config.json_config_importer import ConfigDictKeys, ConfigDictType
from aimet_common.quantsim_config.quantsim_config import OnnxConnectedGraphTypeMapper
from aimet_tensorflow.utils.common import onnx_tf_conn_graph_type_pairs
from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.quantsim_config.quantsim_config import MAP_TF_PARAM_NAME_TO_QUANTSIM_NAME


def get_is_symmetric_flag_for_op_param(configs: ConfigDictType, conn_graph: ConnectedGraph,
                                       tf_op_name: str, param_name: str):
    """
    NOTE: Checks config file in reverse order of specificity.

    Returns is_symmetric flag for op's param if it is set in config file else returns
    False. First check all ops of specific types, second check all params of specific
    and lastly check for default types.

    :param configs: Dictionary containing configs.
    :param conn_graph: Connected graph.
    :param tf_op_name: TensorFlow operation name.
    :param param_name: Parameter name.
    :return: Is_symmetric flag for given op's param.
    """
    # pylint: disable=too-many-locals
    assert param_name in MAP_TF_PARAM_NAME_TO_QUANTSIM_NAME.keys(), "param name is invalid."
    default_is_symmetric = False

    # third level of specificity which applies to specific op_type's parameters.
    op_type_configs = configs[ConfigDictKeys.OP_TYPE]
    onnx_conn_graph_name_mapper = OnnxConnectedGraphTypeMapper(onnx_tf_conn_graph_type_pairs)
    conn_graph_op = conn_graph.get_op_from_module_name(tf_op_name)
    onnx_types = onnx_conn_graph_name_mapper.get_onnx_type_from_conn_graph_type(conn_graph_op.type)
    if onnx_types:
        for onnx_type in onnx_types:
            if onnx_type in op_type_configs:
                op_type_config = op_type_configs[onnx_type]
                if ConfigDictKeys.PARAMS in op_type_config:
                    param_config = op_type_config[ConfigDictKeys.PARAMS].get(param_name)
                    if param_config and ConfigDictKeys.IS_SYMMETRIC in param_config:
                        is_symmetric = param_config[ConfigDictKeys.IS_SYMMETRIC]
                        return is_symmetric

    # Second level of specificity which applies to all parameters.
    param_config = configs[ConfigDictKeys.PARAMS].get(param_name)
    if param_config and ConfigDictKeys.IS_SYMMETRIC in param_config:
        is_symmetric = param_config[ConfigDictKeys.IS_SYMMETRIC]
        return is_symmetric

    # First level of specificity which applies to all the ops and parameters.
    default_configs = configs[ConfigDictKeys.DEFAULTS]
    default_param_configs = default_configs[ConfigDictKeys.PARAMS]
    if ConfigDictKeys.IS_SYMMETRIC in default_param_configs:
        is_symmetric = default_param_configs[ConfigDictKeys.IS_SYMMETRIC]
        return is_symmetric

    return default_is_symmetric
