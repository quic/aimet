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
import itertools
import json
import shutil
from typing import Tuple, Union, Dict, List, Callable, Any
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme, QuantizationDataType

from aimet_torch import utils
from aimet_torch.save_utils import SaveUtils
from aimet_torch.meta import connectedgraph_utils
from aimet_torch.quantsim import QuantizationSimModel, QcQuantizeWrapper
from aimet_torch.qc_quantize_op import StaticGridQuantWrapper, QcQuantizeOpMode
from aimet_torch.tensor_quantizer import StaticGridPerChannelQuantizer
from aimet_torch.adaround.adaround_tensor_quantizer import AdaroundTensorQuantizer
from aimet_torch.adaround.adaround_optimizer import AdaroundOptimizer
from aimet_torch.adaround.adaround_loss import AdaroundHyperParameters

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

# The following modules with weights are supported by Adaround
AdaroundSupportedModules = (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)
WORKING_DIR = '/tmp/adaround/'


class AdaroundParameters:
    """
    Configuration parameters for Adaround
    """
    def __init__(self, data_loader: DataLoader, num_batches: int,
                 default_num_iterations: int = None, default_reg_param: float = 0.01,
                 default_beta_range: Tuple = (20, 2), default_warm_start: float = 0.2,
                 forward_fn: Callable[[torch.nn.Module, Any], Any] = None):
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
    def apply_adaround(cls, model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple], params: AdaroundParameters,
                       path: str, filename_prefix: str, default_param_bw: int = 4,
                       param_bw_override_list: List[Tuple[torch.nn.Module, int]] = None,
                       ignore_quant_ops_list: List[torch.nn.Module] = None,
                       default_quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                       default_config_file: str = None) -> torch.nn.Module:
        """
        Returns model with optimized weight rounding of every module (Conv and Linear) and also saves the
        corresponding quantization encodings to a separate JSON-formatted file that can then be imported by
        QuantSim for inference or QAT

        :param model: Model to Adaround
        :param dummy_input: Dummy input to the model. Used to parse model graph. If the model has more than one input,
                            pass a tuple. User is expected to place the tensors on the appropriate device.
        :param params: Parameters for Adaround
        :param path: path where to store parameter encodings
        :param filename_prefix: Prefix to use for filename of the encodings file
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters
        :param param_bw_override_list: List of Tuples. Each Tuple is a module and the corresponding parameter bitwidth
                                       to be used for that module.
        :param ignore_quant_ops_list: Ops listed here are skipped during quantization needed for AdaRounding. Do not
                                      specify Conv and Linear modules in this list. Doing so, will affect accuracy.
        :param default_quant_scheme: Quantization scheme. Supported options are using Quant Scheme Enum
                                    QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced
        :param default_config_file: Default configuration file for model quantizers
        :return: Model with Adarounded weights and saves corresponding parameter encodings JSON file at provided path
        """
        # pylint: disable=too-many-arguments
        # Create Quant sim with given parameters
        quant_sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme=default_quant_scheme,
                                         default_param_bw=default_param_bw,
                                         config_file=default_config_file)

        # For the modules in the param_bw_override_list, override the default parameter bitwidths in the QuantSim
        if param_bw_override_list:
            cls._override_param_bitwidth(model, quant_sim, param_bw_override_list)

        if ignore_quant_ops_list:
            cls._exclude_modules(model, quant_sim, ignore_quant_ops_list)

        # Compute only param encodings
        cls._compute_param_encodings(quant_sim)

        return cls._apply_adaround(quant_sim, model, dummy_input, params, path, filename_prefix)

    @classmethod
    def _apply_adaround(cls, quant_sim: QuantizationSimModel, model: torch.nn.Module,
                        dummy_input: Union[torch.Tensor, Tuple], params: AdaroundParameters,
                        path: str, filename_prefix: str) -> torch.nn.Module:
        """
        Returns model with optimized weight rounding of every module (Conv and Linear) and also saves the
        corresponding quantization encodings to a separate JSON-formatted file that can then be imported by
        QuantSim for inference or QAT

        :param quant_sim: QuantizationSimModel object to optimize weight rounding.
                          The activation quantizers are expected to have been disabled.
        :param model: Original fp32 model from which quant_sim was created.
        :param dummy_input: Dummy input to the model. Used to parse model graph. If the model has more than one input,
                            pass a tuple. User is expected to place the tensors on the appropriate device.
        :param params: Parameters for Adaround
        :param path: path where to store parameter encodings
        :param filename_prefix: Prefix to use for filename of the encodings file
        :return: Model with Adarounded weights and saves corresponding parameter encodings JSON file at provided path
        """

        # Sanity check: All the input/output quantizers should be disabled
        _, input_quantizers, output_quantizers = utils.get_all_quantizers(quant_sim.model)
        for quantizer in itertools.chain(input_quantizers, output_quantizers):
            assert not quantizer.enabled

        # Get the module - activation function pair using ConnectedGraph
        module_act_func_pair = connectedgraph_utils.get_module_act_func_pair(model, dummy_input)

        cls._adaround_model(model, quant_sim, module_act_func_pair, params, dummy_input)

        # Update every module (AdaroundSupportedModules) weight with Adarounded weight (Soft rounding)
        cls._update_modules_with_adarounded_weights(quant_sim)

        # Export quantization encodings to JSON-formatted file
        cls._export_encodings_to_json(path, filename_prefix, quant_sim)

        SaveUtils.remove_quantization_wrappers(quant_sim.model)
        logger.info('Completed Adarounding Model')
        return quant_sim.model

    @classmethod
    def _adaround_model(cls, model: torch.nn.Module, quant_sim: QuantizationSimModel, module_act_func_pair: Dict,
                        params: AdaroundParameters, dummy_input: Union[torch.Tensor, Tuple]):
        """
        Optimize weight rounding of every module (AdaroundSupportedModules) of model in sequential manner
        based on occurrence
        :param model: Original fp32 model from which quant_sim was created.
        :param quant_sim: QuantizationSimModel object to optimize weight rounding.
                          The activation quantizers are expected to have been disabled.
        :param module_act_func_pair: Dictionary of module to immediate following activation function
        :param params: Adaround parameters
        :param dummy_input: Dummy input to the model
        """
        # pylint: disable=too-many-locals

        num_iterations = params.num_iterations

        if num_iterations is None:
            param_quantizers, _, _ = utils.get_all_quantizers(quant_sim.model)
            lowest_weight_bw = min(
                quantizer.bitwidth for quantizer in param_quantizers
                if quantizer.enabled and quantizer.data_type == QuantizationDataType.int
            )
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
            modules = utils.get_ordered_list_of_modules(model, dummy_input)

            for name, module in tqdm(modules):
                if isinstance(module, AdaroundSupportedModules):
                    # Using name, get corresponding quantized wrapper module from Quant sim model
                    quant_wrapper = cls._get_quant_wrapper(quant_sim.model, name)
                    if quant_wrapper:
                        # Replace quant module's tensor quantizer with Adaround tensor quantizer
                        cls._replace_tensor_quantizer(quant_wrapper)

                        # Get module's next following activation function
                        act_func = module_act_func_pair[module]

                        logger.info("Started Optimizing weight rounding of module: %s", name)
                        AdaroundOptimizer.adaround_module(module, quant_wrapper, model, quant_sim.model, act_func,
                                                          cached_dataset, params.forward_fn, opt_params)
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
        for quant_module in quant_sim.model.modules():
            if isinstance(quant_module, StaticGridQuantWrapper):
                # Adaround requires input and output quantizers to be disabled
                for quatizer in quant_module.input_quantizers:
                    quatizer.enabled = False
                for quatizer in quant_module.output_quantizers:
                    quatizer.enabled = False

                # pylint: disable=protected-access
                for name, param in quant_module._module_to_wrap.named_parameters():
                    param_quantizer = quant_module.param_quantizers[name]
                    param_quantizer.reset_encoding_stats()
                    param_quantizer.update_encoding_stats(param.data)
                    param_quantizer.compute_encoding()

                # Wrapper mode must be set to ACTIVE because the wrapper's quantize_dequantize_params() will only call
                # into the param tensor quantizer's quantize_dequantize() if the mode is not PASSTHROUGH.
                quant_module.set_mode(QcQuantizeOpMode.ACTIVE)

    @staticmethod
    def _replace_tensor_quantizer(quant_module: StaticGridQuantWrapper):
        """
        Replace the quantized module's weight tensor quantizer with the Adaround tensor quantizer
        :param quant_module: quant module
        """
        assert quant_module.param_quantizers['weight'], '%s does not have weight parameter.' % quant_module
        assert quant_module.param_quantizers['weight'].encoding, '%s encoding needs to be set.' % quant_module

        quantizer = quant_module.param_quantizers['weight']
        ch_axis = 0
        if isinstance(quantizer, StaticGridPerChannelQuantizer):
            # pylint: disable=protected-access
            ch_axis = quantizer._ch_axis

        adaround_quantizer = AdaroundTensorQuantizer(quantizer.bitwidth, 'Adaptive', quantizer.quant_scheme,
                                                     quantizer.use_symmetric_encodings, quantizer.enabled, ch_axis)
        adaround_quantizer.use_strict_symmetric = quantizer.use_strict_symmetric
        adaround_quantizer.use_unsigned_symmetric = quantizer.use_unsigned_symmetric

        # Set the encodings and replace by Adaround tensor quantizer
        adaround_quantizer.encoding = quantizer.encoding
        quant_module.param_quantizers['weight'] = adaround_quantizer

    @staticmethod
    def _get_quant_wrapper(quant_sim_model: torch.nn.Module, module_name: str) -> Union[StaticGridQuantWrapper, None]:
        """
        For given module name, get the quantized wrapper module from the QuantSim model
        :param quant_sim_model: Model with simulation ops
        :param module_name: Module name
        :return: Quantized wrapper module or None
        """
        quant_module = None

        for name, module in quant_sim_model.named_modules():
            if name == module_name and isinstance(module, StaticGridQuantWrapper):
                quant_module = module
                break

        return quant_module

    @classmethod
    def _update_modules_with_adarounded_weights(cls, quant_sim: QuantizationSimModel):
        """
        Update every module (Conv and Linear)'s weight parameter with Adarounded weight (Soft rounding)
        :param quant_sim: The QuantSim that contains the model and Adaround tensor quantizers
        """
        # pylint: disable=protected-access
        for quant_module in quant_sim.model.modules():
            if isinstance(quant_module, StaticGridQuantWrapper) and \
                    isinstance(quant_module._module_to_wrap, AdaroundSupportedModules):
                quantizer = quant_module.param_quantizers['weight']

                # It is possible that a module with weights defined in the model may not be used in the
                # forward pass. These modules will not have a AdaroundTensorQuantizer associated with them
                if isinstance(quantizer, AdaroundTensorQuantizer):
                    cls._update_module_params(quant_module._module_to_wrap, quantizer)

    @staticmethod
    def _update_module_params(module: torch.nn.Module, quantizer: AdaroundTensorQuantizer):
        """
        Update module's weight parameter with Adarounded weight
        :param module: module which was Adarounded
        :param quantizer: Tensor quantizer associated with the module
        """
        for param_name, param in module.named_parameters():
            # Only the weight parameter is Adarounded
            if param_name == 'weight':
                orig_weight = param.detach().clone()

                # Use soft rounding to compute Adarounded weight
                quantizer.use_soft_rounding = True
                adaround_weight = quantizer.adaround_weights(orig_weight)

                param.data.zero_()
                param.data.add_(adaround_weight.data)

    @classmethod
    def _export_encodings_to_json(cls, path: str, filename_prefix: str, quant_sim: QuantizationSimModel):
        """
        Save Adadrounded module's parameter encodings to JSON file
        :param path: path where to store param encodings
        :param filename_prefix: filename to store exported weight encodings in JSON format
        :param quant_sim: QunatSim that contains the model and Adaround tensor quantizers
        """
        # pylint: disable=protected-access
        # Create a dictionary to export to JSON file
        param_encodings = {}

        for name, quant_module in quant_sim.model.named_modules():
            if isinstance(quant_module, StaticGridQuantWrapper) and \
                    isinstance(quant_module._module_to_wrap, AdaroundSupportedModules):
                quantizer = quant_module.param_quantizers['weight']

                if isinstance(quantizer, AdaroundTensorQuantizer):
                    cls._update_param_encodings_dict(quant_module, name, param_encodings)

        # export encodings to JSON file
        os.makedirs(os.path.abspath(path), exist_ok=True)
        encoding_file_path = os.path.join(path, filename_prefix + '.encodings')
        with open(encoding_file_path, 'w') as encoding_fp:
            json.dump(param_encodings, encoding_fp, sort_keys=True, indent=4)

    @classmethod
    def _update_param_encodings_dict(cls, quant_module: StaticGridQuantWrapper, name: str, param_encodings: Dict):
        """
        Add module's weight parameter encodings to dictionary to be used for exporting encodings
        :param quant_module: quant module
        :param name: name of module
        :param param_encodings: Dictionary of param encodings
        """
        for orig_param_name, param_quantizer in quant_module.param_quantizers.items():
            if orig_param_name == 'weight':
                param_name = name + '.' + orig_param_name
                encodings = cls._create_encodings_dict_for_quantizer(param_quantizer)
                param_encodings[param_name] = encodings

    @staticmethod
    def _create_encodings_dict_for_quantizer(quantizer: AdaroundTensorQuantizer) -> List[Dict]:
        """
        Return encodings for given qunatizer
        :param quantizer: Tensor quantizer associated with module's param
        :return: Dictionary containing encodings
        """
        quant_encodings = quantizer.encoding
        if not isinstance(quantizer.encoding, list):
            quant_encodings = [quant_encodings]

        encodings_dict = []
        for enc in quant_encodings:
            encodings_dict.append({'min': enc.min,
                                   'max': enc.max,
                                   'scale': enc.delta,
                                   'offset': enc.offset,
                                   'bitwidth': enc.bw,
                                   'is_symmetric': str(quantizer.use_symmetric_encodings),
                                   'dtype': 'int' if quantizer.data_type == QuantizationDataType.int else 'float'})
        return encodings_dict

    @staticmethod
    def _override_param_bitwidth(model: torch.nn.Module, quant_sim: QuantizationSimModel,
                                 param_bw_override_list: List[Tuple[torch.nn.Module, int]]):
        """
        For the QuantSim, for the list of modules in the param_bw_override_list,
        overrides the default parameter bitwidths with the provided bitwidth.

        :param model: The original model
        :param quant_sim: The QuantSim that was created using a deepcopy of the original model.
        :param param_bw_override_list: List of Tuples. Each Tuple is a module and the corresponding parameter bitwidth
                                       to be used for that module.
        """
        # Create a mapping of original model's AdaRoundable module and their name
        module_to_name = {}
        for name, module in model.named_modules():
            if isinstance(module, AdaroundSupportedModules):
                module_to_name[module] = name

        # Create a mapping of QuantSim model's AdaRoundable module name and their module
        name_to_module = {}
        for q_name, q_module in quant_sim.model.named_modules():
            if isinstance(q_module, QcQuantizeWrapper):
                if isinstance(q_module._module_to_wrap, AdaroundSupportedModules):  # pylint: disable=protected-access
                    name_to_module[q_name] = q_module

        # For the modules specified in the param_bw_override_list, set the weight quantizer bitwidth
        for (module, bw) in param_bw_override_list:
            module_name = module_to_name[module]
            quant_wrapper = name_to_module[module_name]
            quant_wrapper.param_quantizers['weight'].bitwidth = bw

    @classmethod
    def _exclude_modules(cls, model: torch.nn.Module, quant_sim: QuantizationSimModel,
                         ignore_quant_ops_list: List[torch.nn.Module]):
        """
        For the modules mentioned in the ignore_quant_ops_list, remove the corresponding quant wrappers from the
        quantSim and excludes modules from adaround optimization.

        :param model: The original model
        :param quant_sim: The QuantSim that was created using a deepcopy of the original model.
        :param ignore_quant_ops_list: The list of modules for which the Quantization wrappers are removed from the
                                      QuantSim object.
        """
        quant_wrappers_to_exclude = []
        for module in ignore_quant_ops_list:
            for m in module.modules():
                name = utils.get_layer_name(model, m)
                quant_wrapper = cls._get_quant_wrapper(quant_sim.model, name)
                if quant_wrapper:
                    quant_wrappers_to_exclude.append(quant_wrapper)

        quant_sim.exclude_layers_from_quantization(quant_wrappers_to_exclude)
