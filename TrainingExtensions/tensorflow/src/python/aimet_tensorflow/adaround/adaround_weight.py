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
import tensorflow as tf
import libpymo

# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_tensorflow.utils import graph_saver
from aimet_tensorflow.utils.common import get_ordered_ops
from aimet_tensorflow.utils.op.conv import WeightTensorUtils
from aimet_tensorflow.adaround.activation_sampler import ActivationSampler
from aimet_tensorflow.adaround.adaround_loss import AdaroundHyperParameters
from aimet_tensorflow.adaround.adaround_optimizer import AdaroundOptimizer
from aimet_tensorflow.adaround.adaround_wrapper import AdaroundWrapper

AdaroundSupportedOps = ('Conv2D', 'DepthwiseConv2dNative', 'MatMul')
ActFuncMap = {'Relu': tf.nn.relu, 'Relu6': tf.nn.relu6, 'Tanh': tf.nn.tanh, 'Sigmoid': tf.nn.sigmoid,
              'Softmax': tf.nn.softmax}

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


class Adaround:
    """
    Weight-rounding mechanism for Post Training Quantization (PTQ)
    """
    @classmethod
    def apply_adaround(cls, session: tf.compat.v1.Session, starting_op_names: List[str], output_op_names: List[str],
                       params: AdaroundParameters, path: str, filename_prefix: str, default_param_bw: int = 4,
                       default_quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                       default_is_symmetric: bool = False) -> tf.compat.v1.Session:
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
        :param default_is_symmetric: True if symmetric encodings is used, else asymmetric encodings.
         Default False.
        :return: Tf session with Adarounded weight and saves corresponding parameter encodings JSON file
         at provided path
        """
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        if not os.path.exists(WORKING_DIR):
            os.makedirs(WORKING_DIR)

        # Create copies which will have model's weights quantized with hard and soft rounding
        session_hard_rounded_weight = graph_saver.save_and_load_graph(WORKING_DIR, session)
        session_soft_rounded_weight = graph_saver.save_and_load_graph(WORKING_DIR, session)

        # Optimization Hyper parameters
        opt_params = AdaroundHyperParameters(params.num_iterations, params.reg_param, params.beta_range,
                                             params.warm_start)
        # Activation sampler
        act_sampler = ActivationSampler(params.data_set)

        # Get Adaround supported ops based on occurrence in the model
        ordered_ops = cls._get_ordered_list_of_ops(session.graph, starting_op_names, output_op_names)
        param_encodings = {}

        for op in ordered_ops:
            logger.info("Started Optimizing weight rounding of op: %s", op.name)

            # Using name, get corresponding op
            hard_rounded_op = session_hard_rounded_weight.graph.get_operation_by_name(op.name)
            soft_rounded_op = session_soft_rounded_weight.graph.get_operation_by_name(op.name)

            # Collect input and output activations data
            all_inp_data, all_out_data = act_sampler.sample_activation(op, hard_rounded_op, session,
                                                                       session_hard_rounded_weight, starting_op_names,
                                                                       params.num_batches)
            # Find next following activation function
            act_func = cls._get_act_func(op)

            # Perform Adaround optimization in separate graph
            graph = tf.Graph()
            with graph.as_default():
                wrapper = AdaroundWrapper(session, op, default_param_bw, default_is_symmetric, default_quant_scheme)
                hard_rounded_weight,\
                soft_rounded_weight = AdaroundOptimizer().adaround_wrapper(wrapper, act_func, all_inp_data,
                                                                           all_out_data, opt_params)

            # Update param encodings dictionary
            cls._update_param_encodings_dict(param_encodings, op, wrapper.encoding, default_is_symmetric)

            # Update with hard and soft rounded weights
            WeightTensorUtils.update_tensor_for_op(session_hard_rounded_weight, hard_rounded_op, hard_rounded_weight)
            WeightTensorUtils.update_tensor_for_op(session_soft_rounded_weight, soft_rounded_op, soft_rounded_weight)

        # Export quantization encodings to JSON-formatted file at provided path
        cls.export_encoding_to_json(path, filename_prefix, param_encodings)

        # Close intermediate session
        session_hard_rounded_weight.close()

        if os.path.exists(WORKING_DIR):
            logger.info('Deleting temporary working directory %s', WORKING_DIR)
            shutil.rmtree(WORKING_DIR)

        logger.info('Completed Adarounding Model')

        return session_soft_rounded_weight

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
        consumer_ops = op.outputs[0].consumers()

        # op -> act_func
        if consumer_ops[0].type in ActFuncMap:
            act_func = ActFuncMap[consumer_ops[0].type]

        # op -> bias_add -> act_func
        elif consumer_ops[0].type in ['Add', 'BiasAdd'] and\
                consumer_ops[0].outputs[0].consumers()[0].type in ActFuncMap:
            act_func = ActFuncMap[consumer_ops[0].outputs[0].consumers()[0].type]

        else:
            act_func = None

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
        encoding_file_path = os.path.join(path, filename_prefix + '.encodings')
        with open(encoding_file_path, 'w') as encoding_fp:
            json.dump(param_encodings, encoding_fp, sort_keys=True, indent=4)

    @staticmethod
    def _update_param_encodings_dict(encoding_dict: Dict, op: tf.Operation, encoding: libpymo.TfEncoding,
                                     is_symmetric: bool):
        """
        Add op's parameter encoding to dictionary to be used for exporting
        :param encoding_dict: Encoding dictionary
        :param op: Tf op
        :param encoding: Encoding
        :param is_symmetric: Symmetric vs Asymmetric boolean
        """
        tensor_name = op.inputs[1].name
        encoding_dict[tensor_name] = [{'min': encoding.min,
                                       'max': encoding.max,
                                       'scale': encoding.delta,
                                       'offset': encoding.offset,
                                       'bitwidth': encoding.bw,
                                       'is_symmetric': is_symmetric}]
