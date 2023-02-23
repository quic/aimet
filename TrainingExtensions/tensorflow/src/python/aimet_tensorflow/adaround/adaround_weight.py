# /usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Top level API for Adaptive Rounding - Post-Training Quantization (PTQ) """

import os
import json
import shutil
from typing import List, Tuple, Callable, Union, Dict
import numpy as np
from tqdm import tqdm
import tensorflow as tf

# Import AIMET specific modules
import aimet_common.libpymo as libpymo
from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_common.quantsim_config.json_config_importer import JsonConfigImporter, ConfigDictKeys, ConfigDictType
from aimet_tensorflow.utils import graph_saver
from aimet_tensorflow.utils.common import get_ordered_ops
from aimet_tensorflow.utils.op.conv import WeightTensorUtils
from aimet_tensorflow.quantsim_config.quantsim_config import MAP_TF_PARAM_NAME_TO_QUANTSIM_NAME
from aimet_tensorflow.adaround.activation_sampler import ActivationSampler
from aimet_tensorflow.adaround.adaround_loss import AdaroundHyperParameters
from aimet_tensorflow.adaround.adaround_optimizer import AdaroundOptimizer
from aimet_tensorflow.adaround.adaround_wrapper import AdaroundWrapper

AdaroundSupportedOps = ('Conv2D', 'DepthwiseConv2dNative', 'MatMul', 'Conv2DBackpropInput')

ActFuncMap = {'Relu': tf.nn.relu, 'Relu6': tf.nn.relu6, 'Tanh': tf.nn.tanh, 'Sigmoid': tf.nn.sigmoid,
              'Softmax': tf.nn.softmax}

tf_op_type_to_onnx_type_dict = {
        "Conv2D": "Conv",
        "DepthwiseConv2dNative": "Conv",
        "MatMul": "Gemm",
    }

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)
WORKING_DIR = '/tmp/adaround/'


class AdaroundParameters:
    """
    Configuration parameters for Adaround
    """
    def __init__(self, data_set: tf.data.Dataset, num_batches: int, default_num_iterations: int = 10000,
                 default_reg_param: float = 0.01, default_beta_range: Tuple = (20, 2), default_warm_start: float = 0.2):
        """
        :param data_set: TF Data set
        :param num_batches: Number of batches
        :param default_num_iterations: Number of iterations to adaround each layer. Default 10000
        :param default_reg_param: Regularization parameter, trading off between rounding loss vs reconstruction loss.
         Default 0.01
        :param default_beta_range: Start and stop beta parameter for annealing of rounding loss (start_beta, end_beta).
         Default (20, 2)
        :param default_warm_start: warm up period, during which rounding loss has zero effect. Default 20% (0.2)
        """
        self.data_set = data_set
        self.num_batches = num_batches
        self.num_iterations = default_num_iterations
        self.reg_param = default_reg_param
        self.beta_range = default_beta_range
        self.warm_start = default_warm_start

    def __eq__(self, other: "AdaroundParameters"):
        return self.data_set == other.data_set and\
               self.num_batches == other.num_batches and\
               self.num_iterations == other.num_iterations and\
               self.reg_param == other.reg_param and\
               self.beta_range == other.beta_range and\
               self.warm_start == other.warm_start


class Adaround:
    """
    Weight-rounding mechanism for Post Training Quantization (PTQ)
    """
    @classmethod
    def apply_adaround(cls, session: tf.compat.v1.Session, starting_op_names: List[str], output_op_names: List[str],
                       params: AdaroundParameters, path: str, filename_prefix: str, default_param_bw: int = 4,
                       default_quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                       default_config_file: str = None) -> tf.compat.v1.Session:
        """
        Returns Tf session - model with optimized weight rounding of every op (Conv and Linear) and also saves the
        corresponding quantization encodings to a separate JSON-formatted file that can then be imported by
        QuantSim for inference or QAT

        :param session: Tf session with model to adaround
        :param starting_op_names: List of starting op names of the model
        :param output_op_names: List of output op names of the model
        :param params: Parameters for adaround
        :param path: path where to store parameter encodings
        :param filename_prefix: Prefix to use for filename of the encodings file
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters. Default 4
        :param default_quant_scheme:  Quantization scheme. Supported options are QuantScheme.post_training_tf or
         QuantScheme.post_training_tf_enhanced. Default QuantScheme.post_training_tf_enhanced
        :param default_config_file: Default configuration file for model quantizers
        :return: Tf session with Adarounded weight and saves corresponding parameter encodings JSON file
         at provided path
        """
        # pylint: disable=too-many-arguments
        if not os.path.exists(WORKING_DIR):
            os.makedirs(WORKING_DIR)

        param_encodings,\
        session_soft_rounded_weight = cls._apply_adaround_helper(session,
                                                                 starting_op_names,
                                                                 output_op_names,
                                                                 params,
                                                                 default_param_bw,
                                                                 default_quant_scheme,
                                                                 default_config_file)

        # Export quantization encodings to JSON-formatted file at provided path
        cls.export_encoding_to_json(path, filename_prefix, param_encodings)

        if os.path.exists(WORKING_DIR):
            logger.info('Deleting temporary working directory %s', WORKING_DIR)
            shutil.rmtree(WORKING_DIR)

        logger.info('Completed Adarounding Model')

        return session_soft_rounded_weight

    @classmethod
    def _apply_adaround_helper( # pylint: disable=too-many-locals
            cls,
            session: tf.compat.v1.Session,
            starting_op_names: List[str],
            output_op_names: List[str],
            params: AdaroundParameters,
            param_bw: int,
            quant_scheme: QuantScheme,
            config_file: str,
    ) -> Tuple[Dict, tf.compat.v1.Session]:
        """
        Helper for apply_adaround().

        NOTE: Soft rounding is only used for op-wise optimization procedure as we need gradients
         for the rounding to be learned and after that we switch to hard rounding (i.e. using
         true fixed point numbers) to be used for collecting later layers activations data.

        When optimization is fully converged (i.e. wrapper.alpha is always exact 0 or 1), there
        is no difference between soft rounding and hard rounding.

        :param session: Tf session with model to adaround.
        :param starting_op_names: List of starting op names of the model.
        :param output_op_names: List of output op names of the model.
        :param params: Parameters for adaround.
        :param param_bw: bitwidth (4-31) to use for quantizing layer parameters.
        :param quant_scheme: Quantization scheme.
        :param config_file: configuration file.
        :return: Dictionary containing encoding for adarounded parameters,
         TF session with soft rounding weights.
        """
        # Create copies which will have model's weights quantized with hard and soft rounding.
        session_hard_rounded_weight = graph_saver.save_and_load_graph(WORKING_DIR, session)
        session_soft_rounded_weight = graph_saver.save_and_load_graph(WORKING_DIR, session)

        # Get parameters from config file.
        configs, strict_symmetric, unsigned_symmetric, enable_per_channel = Adaround.get_config_dict_keys(config_file)

        # Optimization Hyper parameters
        opt_params = AdaroundHyperParameters(params.num_iterations, params.reg_param, params.beta_range,
                                             params.warm_start)
        # Activation sampler
        act_sampler = ActivationSampler(params.data_set)

        # Get Adaround supported ops based on occurrence in the model
        ordered_ops = cls._get_ordered_list_of_ops(session.graph, starting_op_names, output_op_names)

        param_encodings = {}
        for op in tqdm(ordered_ops):
            logger.info("Started Optimizing weight rounding of op: %s", op.name)

            # Using name, get corresponding op from session with soft and hard rounded weights.
            hard_rounded_op = session_hard_rounded_weight.graph.get_operation_by_name(op.name)
            soft_rounded_op = session_soft_rounded_weight.graph.get_operation_by_name(op.name)

            # Collect input and output activations data
            all_inp_data, all_out_data = act_sampler.sample_activation(op, hard_rounded_op, session,
                                                                       session_hard_rounded_weight, starting_op_names,
                                                                       params.num_batches)
            is_symmetric = cls.get_is_symmetric_flag_for_op_param(configs, op.type,
                                                                  param_name="weight",
                                                                  framework_to_onnx_type_dict=tf_op_type_to_onnx_type_dict)


            # Find next following activation function
            act_func = cls._get_act_func(op)

            # Perform Adaround optimization in separate graph
            graph = tf.Graph()
            with graph.as_default():
                output_height, output_width, output_channels = None, None, None
                if op.type == 'Conv2DBackpropInput':
                    output_height, output_width, output_channels = \
                        cls.get_conv2d_transpose_output_tensor_shape(op.get_attr("data_format").decode('utf-8'),
                                                                     all_out_data)
                wrapper = AdaroundWrapper(session, op, param_bw, quant_scheme, is_symmetric,
                                          strict_symmetric, unsigned_symmetric, enable_per_channel, output_height,
                                          output_width, output_channels)
                hard_rounded_weight, soft_rounded_weight = AdaroundOptimizer().adaround_wrapper(wrapper, act_func,
                                                                                                all_inp_data,
                                                                                                all_out_data,
                                                                                                opt_params)

            # Update param encodings dictionary
            cls._update_param_encodings_dict(param_encodings, op, wrapper.encoding, is_symmetric)

            # Update with hard and soft rounded weights
            WeightTensorUtils.update_tensor_for_op(session_hard_rounded_weight, hard_rounded_op, hard_rounded_weight)
            WeightTensorUtils.update_tensor_for_op(session_soft_rounded_weight, soft_rounded_op, soft_rounded_weight)

        # Close intermediate session
        session_hard_rounded_weight.close()

        return param_encodings, session_soft_rounded_weight

    @staticmethod
    def get_config_dict_keys(config_file: str) -> Tuple[ConfigDictType, bool, bool, bool]:
        """
        Get config dictionary keys from config file. Config file will default if no provided one.
        :param config_file: configuration file.
        :return: Config dictionary, strict symmetric flag, unsigned symmetric flag, enable per channel flag.
        """
        configs = JsonConfigImporter.import_json_config_file(config_file)
        strict_symmetric = configs[ConfigDictKeys.DEFAULTS].get(ConfigDictKeys.STRICT_SYMMETRIC, False)
        unsigned_symmetric = configs[ConfigDictKeys.DEFAULTS].get(ConfigDictKeys.UNSIGNED_SYMMETRIC, False)

        # Read per-channel quantization field. Default = False
        per_channel_enabled = configs[ConfigDictKeys.DEFAULTS].get(ConfigDictKeys.PER_CHANNEL_QUANTIZATION, False)

        return configs, strict_symmetric, unsigned_symmetric, per_channel_enabled

    @staticmethod
    def _get_ordered_list_of_ops(graph: tf.Graph, input_op_names: List[str], output_op_names: List[str]) \
            -> List[tf.Operation]:
        """
        Get Adaround supported ops based on occurrence in the model
        :param graph: Model represented as TF data flow graph
        :param input_op_names: List of input op names
        :param output_op_names: List of output op names of the model
        :return: List of Adaround supported ops
        """
        # Get all the ops in the model based on occurrence
        list_of_ordered_ops = get_ordered_ops(graph, input_op_names, output_op_names)

        ordered_ops = []

        for op in list_of_ordered_ops:
            if op.type in AdaroundSupportedOps:
                ordered_ops.append(op)

        return ordered_ops

    @staticmethod
    def _get_act_func(op: tf.Operation) -> Union[Callable, None]:
        """
        Gets immediate following activation function else returns None
        :param op: Tf op
        :return: Callable Tf activation function or None
        """
        act_func = None
        consumer_ops = op.outputs[0].consumers()

        if not consumer_ops:
            return act_func

        # op -> act_func
        if consumer_ops[0].type in ActFuncMap:
            act_func = ActFuncMap[consumer_ops[0].type]

        # op -> bias_add -> act_func
        elif consumer_ops[0].type in ['Add', 'BiasAdd']:
            if consumer_ops[0].outputs[0].consumers() and consumer_ops[0].outputs[0].consumers()[0].type in ActFuncMap:
                act_func = ActFuncMap[consumer_ops[0].outputs[0].consumers()[0].type]

        logger.info("op: %s 's next following act func: %s", op.name, act_func)
        return act_func

    @classmethod
    def export_encoding_to_json(cls, path: str, filename_prefix: str, param_encodings: Dict):
        """
        Save Adadrounded op's parameter encodings to JSON file
        :param path: path where to store param encodings
        :param filename_prefix: filename to store exported weight encodings in JSON format
        :param param_encodings: Parameter encodings dictionary
        """
        # export encodings to JSON file
        os.makedirs(os.path.abspath(path), exist_ok=True)
        encoding_file_path = os.path.join(path, filename_prefix + '.encodings')
        with open(encoding_file_path, 'w') as encoding_fp:
            json.dump(param_encodings, encoding_fp, sort_keys=True, indent=4)

    @staticmethod
    def _update_param_encodings_dict(encoding_dict: Dict, op: tf.Operation,
                                     encoding: Union[libpymo.TfEncoding, List[libpymo.TfEncoding]],
                                     is_symmetric: bool):
        """
        Add op's parameter encoding to dictionary to be used for exporting
        :param encoding_dict: Encoding dictionary
        :param op: Tf op
        :param encoding: Encoding
        :param is_symmetric: Symmetric vs Asymmetric boolean
        """
        tensor_name = op.inputs[1].name
        # Wrap Per Tensor encoding in a list for list comprehension
        encoding = encoding if isinstance(encoding, list) else [encoding]
        encoding_dict[tensor_name] = [{'min': enc.min,
                                       'max': enc.max,
                                       'scale': enc.delta,
                                       'offset': enc.offset,
                                       'bitwidth': enc.bw,
                                       'is_symmetric': is_symmetric} for enc in encoding]

    @staticmethod
    def get_is_symmetric_flag_for_op_param(configs: ConfigDictType, tf_op_type: str, param_name: str,
                                           framework_to_onnx_type_dict: dict) -> bool:
        """
        NOTE: Checks config file in reverse order of specificity.

        Returns is_symmetric flag for op's param if it is set in config file else returns
        False. First check all ops of specific types, second check all params of specific
        and lastly check for default types. If not specified, it will return default is
        symmetric False.

        :param configs: Dictionary containing configs.
        :param tf_op_type: TensorFlow operation type.
        :param param_name: Parameter name.
        :param framework_to_onnx_type_dict: Dictionary mapping framework type to ONNX type.
        :return: is_symmetric flag for given op's param.
        """
        assert param_name in MAP_TF_PARAM_NAME_TO_QUANTSIM_NAME.keys(), "param name is invalid."

        # third level of specificity which applies to specific op_type's parameters.
        try:
            onnx_type = framework_to_onnx_type_dict[tf_op_type]
            return configs[ConfigDictKeys.OP_TYPE] \
                [onnx_type] \
                [ConfigDictKeys.PARAMS] \
                [param_name] \
                [ConfigDictKeys.IS_SYMMETRIC]
        except KeyError:
            pass

        # Second level of specificity which applies to all parameters only.
        try:
            return configs[ConfigDictKeys.PARAMS] \
                [param_name] \
                [ConfigDictKeys.IS_SYMMETRIC]
        except KeyError:
            pass

        # First level of specificity which applies to all the ops and parameters.
        try:
            return configs[ConfigDictKeys.DEFAULTS] \
                [ConfigDictKeys.PARAMS] \
                [ConfigDictKeys.IS_SYMMETRIC]
        except KeyError:
            pass

        # Default is_symmetric False.
        return False

    @staticmethod
    def get_conv2d_transpose_output_tensor_shape(data_format: str, output_data: np.ndarray):
        """
        Get output height, width, and channels from output_data for use in adarounding conv2d transpose op.
        :param data_format: Data format for the op (NHWC or NCHW)
        :param output_data: numpy array containing sampled output of the op
        :return: Tuple containing output height, width, and channels of the op
        """
        if data_format == 'NHWC':
            output_height = output_data.shape[1]
            output_width = output_data.shape[2]
            output_channels = output_data.shape[3]
        else:
            output_height = output_data.shape[2]
            output_width = output_data.shape[3]
            output_channels = output_data.shape[1]
        return output_height, output_width, output_channels
