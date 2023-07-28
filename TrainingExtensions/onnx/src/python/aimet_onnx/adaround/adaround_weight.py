# /usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
import shutil
import contextlib
from typing import Tuple, Dict, List, Callable
from onnx import onnx_pb

from tqdm import tqdm

# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme, QuantizationDataType

from aimet_torch.adaround.adaround_loss import AdaroundHyperParameters
from aimet_torch.adaround.adaround_tensor_quantizer import AdaroundTensorQuantizer
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.qc_quantize_op import OpMode
from aimet_onnx.meta.utils import get_module_act_func_pair, get_ordered_ops
from aimet_onnx import utils

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

# The following modules with weights are supported by Adaround
AdaroundSupportedModules = ['Conv', 'ConvTranspose', 'MatMul', 'Gemm']
WORKING_DIR = '/tmp/adaround/'


class AdaroundParameters:
    """
    Configuration parameters for Adaround
    """
    def __init__(self, data_loader, num_batches: int,
                 default_num_iterations: int = None, default_reg_param: float = 0.01,
                 default_beta_range: Tuple = (20, 2), default_warm_start: float = 0.2,
                 forward_fn: Callable = None):
        """
        :param data_loader: Data loader
        :param num_batches: Number of batches to be used for Adaround.
         A commonly recommended value for this parameter is the smaller value among (1) len(data_loader) and (2) ceil(2000/batch_size)
        :param default_num_iterations: Number of iterations to adaround each layer.
         The default value is 10K for models with 8- or higher bit weights, and 15K for models with lower than 8 bit weights.
        :param default_reg_param: Regularization parameter, trading off between rounding loss vs reconstruction loss.
         Default 0.01
        :param default_beta_range: Start and stop beta parameter for annealing of rounding loss (start_beta, end_beta).
         Default (20, 2)
        :param default_warm_start: warm up period, during which rounding loss has zero effect. Default 20% (0.2)
        :param forward_fn: Optional adapter function that performs forward pass given a model and inputs
         yielded from the data loader. The function expects model as first argument and inputs to model
         as second argument.
        """
        if len(data_loader) < num_batches:
            raise ValueError(f'Can not fetch {num_batches} batches from '
                             f'a data loader of length {len(data_loader)}.')

        self.data_loader = data_loader
        self.num_batches = num_batches
        self.num_iterations = default_num_iterations
        self.reg_param = default_reg_param
        self.beta_range = default_beta_range
        self.warm_start = default_warm_start
        self.forward_fn = forward_fn


class Adaround:
    """
    Weight-rounding mechanism for Post Training Quantization (PTQ)
    """
    @classmethod
    def apply_adaround(cls, model: onnx_pb.ModelProto, params: AdaroundParameters,
                       path: str, filename_prefix: str, default_param_bw: int = 4,
                       param_bw_override_list: List[Tuple[str, int]] = None,
                       ignore_quant_ops_list: List[str] = None,
                       default_quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                       default_config_file: str = None) -> onnx_pb.ModelProto:
        """
        Returns model with optimized weight rounding of every module (Conv and Linear) and also saves the
        corresponding quantization encodings to a separate JSON-formatted file that can then be imported by
        QuantSim for inference or QAT

        :param model: Model to Adaround
        :param params: Parameters for Adaround
        :param path: path where to store parameter encodings
        :param filename_prefix: Prefix to use for filename of the encodings file
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters
        :param param_bw_override_list: List of Tuples. Each Tuple is a param name and the corresponding parameter bitwidth
                                       to be used for that param.
        :param ignore_quant_ops_list: Ops listed here are skipped during quantization needed for AdaRounding. Do not
                                      specify Conv and Linear modules in this list. Doing so, will affect accuracy.
        :param default_quant_scheme: Quantization scheme. Supported options are using Quant Scheme Enum
                                    QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced
        :param default_config_file: Default configuration file for model quantizers
        :return: Model with Adarounded weights and saves corresponding parameter encodings JSON file at provided path
        """
        # pylint: disable=too-many-arguments
        # Create Quant sim with given parameters
        quant_sim = QuantizationSimModel(model, quant_scheme=default_quant_scheme,
                                         default_param_bw=default_param_bw,
                                         config_file=default_config_file)

        # For the params in the param_bw_override_list, override the default parameter bitwidths in the QuantSim
        if param_bw_override_list:
            cls._override_param_bitwidth(quant_sim, param_bw_override_list)

        if ignore_quant_ops_list:
            cls._exclude_modules(quant_sim, ignore_quant_ops_list)

        # Compute only param encodings
        cls._compute_param_encodings(quant_sim)

        return cls._apply_adaround(quant_sim, model, params, path, filename_prefix)

    @classmethod
    def _apply_adaround(cls, quant_sim: QuantizationSimModel, model: onnx_pb.ModelProto, params: AdaroundParameters,
                        path: str, filename_prefix: str) -> onnx_pb.ModelProto:
        """
        Returns model with optimized weight rounding of every module (Conv and Linear) and also saves the
        corresponding quantization encodings to a separate JSON-formatted file that can then be imported by
        QuantSim for inference or QAT

        :param quant_sim: QuantizationSimModel object to optimize weight rounding.
                          The activation quantizers are expected to have been disabled.
        :param model: Original fp32 model from which quant_sim was created.
        :param params: Parameters for Adaround
        :param path: path where to store parameter encodings
        :param filename_prefix: Prefix to use for filename of the encodings file
        :return: Model with Adarounded weights and saves corresponding parameter encodings JSON file at provided path
        """

        # Sanity check: All the input/output quantizers should be disabled
        for quantizer_name in quant_sim.activation_names:
            assert not quant_sim.qc_quantize_op_dict[quantizer_name].enabled

        # Get the module - activation function pair using ConnectedGraph
        module_act_func_pair = get_module_act_func_pair(model)

        cls._adaround_model(model, quant_sim, module_act_func_pair, params)

        # Export quantization encodings to JSON-formatted file
        cls._export_encodings_to_json(path, filename_prefix, quant_sim)

        quant_sim.remove_quantization_nodes()
        logger.info('Completed Adarounding Model')
        return quant_sim.model

    @classmethod
    def _adaround_model(cls, model: onnx_pb.ModelProto, quant_sim: QuantizationSimModel, module_act_func_pair: Dict,
                        params: AdaroundParameters):
        """
        Optimize weight rounding of every module (AdaroundSupportedModules) of model in sequential manner
        based on occurrence
        :param model: Original fp32 model from which quant_sim was created.
        :param quant_sim: QuantizationSimModel object to optimize weight rounding.
                          The activation quantizers are expected to have been disabled.
        :param module_act_func_pair: Dictionary of module to immediate following activation function
        :param params: Adaround parameters
        """
        # pylint: disable=too-many-locals, protected-access

        num_iterations = params.num_iterations

        if num_iterations is None:
            lowest_weight_bw = 32
            for param_name in quant_sim.param_names:
                quantizer = quant_sim.qc_quantize_op_dict[param_name]
                if quantizer.enabled and quantizer.data_type == QuantizationDataType.int:
                    lowest_weight_bw = min(lowest_weight_bw, quantizer.bitwidth)
            # If the lowest wegith bitwidth is < 8, then set num_iterations to 15K by default
            if lowest_weight_bw < 8:
                num_iterations = 15000
            else:
                num_iterations = 10000

        try:
            # Cache model input data to WORKING_DIR
            cached_dataset = utils.CachedDataset(params.data_loader, params.num_batches, WORKING_DIR)

            # Optimization Hyper parameters
            opt_params = AdaroundHyperParameters(num_iterations, params.reg_param, params.beta_range,
                                                 params.warm_start)

            # AdaRound must be applied to modules in the order of occurrence
            modules = get_ordered_ops(model)

        finally:
            if os.path.exists(WORKING_DIR):
                logger.info('Deleting model inputs from location: %s', WORKING_DIR)
                shutil.rmtree(WORKING_DIR)

    @staticmethod
    def _compute_param_encodings(quant_sim: QuantizationSimModel):
        """
        Compute encodings for parameters, needed for initializing Adaround quantizers
        :param quant_sim: Quant sim
        """
        for op_name, qc_op in quant_sim.qc_quantize_op_dict.items():
            if op_name in quant_sim.param_names:
                qc_op.op_mode = OpMode.oneShotQuantizeDequantize
                qc_op.compute_encodings()
                qc_op.op_mode = OpMode.quantizeDequantize


    @staticmethod
    @contextlib.contextmanager
    def _create_param_to_tensor_quantizer_dict(quant_sim: QuantizationSimModel) -> Dict[str, AdaroundTensorQuantizer]:
        """
        Create Adaround tensor quantizers for weight tensor

        :param quant_sim: Quant sim
        :return: Dict of param name to AdaroundTensorQuantizer
        """
        param_to_tq_dict = {}
        for param_name in quant_sim.param_names:
            quantizer = quant_sim.qc_quantize_op_dict[param_name]
            ch_axis = -1
            if quantizer.quant_info.usePerChannelMode:
                ch_axis = quantizer.quant_info.channelAxis
            adaround_quantizer = AdaroundTensorQuantizer(quantizer.bitwidth, 'Adaptive', quantizer.quant_scheme,
                                                         quantizer.use_symmetric_encodings, quantizer.enabled, ch_axis)

            adaround_quantizer.use_strict_symmetric = quantizer.use_strict_symmetric
            adaround_quantizer.use_unsigned_symmetric = quantizer.use_unsigned_symmetric

            # Set the encodings and replace by Adaround tensor quantizer
            adaround_quantizer.encoding = quantizer.encoding
            param_to_tq_dict[param_name] = adaround_quantizer

        return param_to_tq_dict

    @classmethod
    def _export_encodings_to_json(cls, path: str, filename_prefix: str, quant_sim: QuantizationSimModel):
        """
        Save Adadrounded module's parameter encodings to JSON file
        :param path: path where to store param encodings
        :param filename_prefix: filename to store exported weight encodings in JSON format
        :param quant_sim: QunatSim that contains the model and Adaround tensor quantizers
        """

    @classmethod
    def _update_param_encodings_dict(cls, quant_module, name: str, param_encodings: Dict):
        """
        Add module's weight parameter encodings to dictionary to be used for exporting encodings
        :param quant_module: quant module
        :param name: name of module
        :param param_encodings: Dictionary of param encodings
        """

    @staticmethod
    def _create_encodings_dict_for_quantizer(quantizer: TensorQuantizer) -> List[Dict]:
        """
        Return encodings for given qunatizer
        :param quantizer: Tensor quantizer associated with module's param
        :return: Dictionary containing encodings
        """

    @staticmethod
    def _override_param_bitwidth(quant_sim: QuantizationSimModel,
                                 param_bw_override_list: List[Tuple[str, int]]):
        """
        For the QuantSim, for the list of modules in the param_bw_override_list,
        overrides the default parameter bitwidths with the provided bitwidth.

        :param quant_sim: The QuantSim that was created using a deepcopy of the original model.
        :param param_bw_override_list: List of Tuples. Each Tuple is a param name and the corresponding parameter bitwidth
                                       to be used for that param.
        """
        # For the params specified in the param_bw_override_list, set the weight quantizer bitwidth
        for (param_name, bw) in param_bw_override_list:
            quant_sim.qc_quantize_op_dict[param_name] = bw

    @classmethod
    def _exclude_modules(cls, quant_sim: QuantizationSimModel,
                         ignore_quant_ops_list: List[str]):
        """
        For the modules mentioned in the ignore_quant_ops_list, remove the corresponding quant wrappers from the
        quantSim and excludes modules from adaround optimization.

        :param model: The original model
        :param quant_sim: The QuantSim that was created using a deepcopy of the original model.
        :param ignore_quant_ops_list: The list of quantizers for which the Quantization wrappers are removed from the
                                      QuantSim object.
        """
