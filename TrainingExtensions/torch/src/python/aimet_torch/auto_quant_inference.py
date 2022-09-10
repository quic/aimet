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
"""Automatic Post-Training Quantization"""
import copy
from typing import Any, Union, Collection, Tuple, List, Callable
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aimet_common.defs import QuantScheme
from aimet_common.quantsim import validate_quantsim_inputs
from aimet_common.utils import AimetLogger
from aimet_torch import utils
from aimet_torch.auto_quant import NUM_SAMPLES_FOR_PERFORMANCE_EVALUATION
from aimet_torch.batch_norm_fold import fold_all_batch_norms_to_weight
import aimet_torch.model_preparer as ModelPreparer
from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_torch.quantsim import QuantizationSimModel

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AutoQuant)

class AutoQuantInfer:
    '''
    Automates and applies quantization techniques
    '''

    def __init__(  # pylint: disable=too-many-arguments
            self,
            model: torch.nn.Module,
            unlabeled_dataset_iterable: Union[DataLoader, Collection],
            eval_callback: Callable[[torch.nn.Module], float],
            dummy_input: Union[torch.Tensor, Tuple],
            use_cuda: bool = False,
            cuda_device_num: int = 0,
            ignore_errors: bool = False) -> None:
        '''
        :param model: Model to be quantized
        :param unlabeled_dataset_iterable: A collection that iterates over an unlabeled dataset, used for computing encodings
        :param eval_callback: Function that calculates the evaluation score
        :param dummy_input: Dummy input for the model
        :param use_cuda: Flag set to True if CUDA should be used
        :param cuda_device_num: Device index for CUDA selection
        :param ignore_errors: Flag set to False by default, if True will try to proceed with execution and handle errors internally if possible
        '''

        self.fp32_model, self.dummy_input = self._validate_inputs(model, unlabeled_dataset_iterable, eval_callback, dummy_input, use_cuda, cuda_device_num)

        self.dataset = unlabeled_dataset_iterable
        self.eval_callback = eval_callback
        self.ignore_errors = ignore_errors
        self._quantsim_model = None
        self._bn_folding_applied = False

        def forward_pass_callback(model, _: Any = None):
            device = utils.get_device(model)
            with utils.in_eval_mode(model), torch.no_grad():
                for input_data in tqdm(unlabeled_dataset_iterable):
                    input_data = utils.change_tensor_device_placement(input_data, device)
                    if isinstance(input_data, torch.Tensor):
                        model(input_data)
                    else:
                        assert isinstance(input_data, (tuple, list))
                        model(*input_data)

        self.forward_pass_callback = forward_pass_callback

    @staticmethod
    def _validate_inputs(model: torch.nn.Module,
                         unlabeled_dataset: Union[DataLoader, Collection],
                         eval_callback: Callable[[torch.nn.Module], float],
                         dummy_input: Union[torch.Tensor, Tuple],
                         use_cuda: bool,
                         cuda_device_num: int):
        """
        Confirms inputs are of the correct type and that the dummy_input and model are on the correct device based on use_cuda and CUDA availability
        :param model: Model to be quantized
        :param unlabeled_dataset: A collection that iterates over an unlabeled dataset, used for computing encodings
        :param eval_callback: Function that calculates the evaluation score
        :param dummy_input: Dummy input for the model
        :param use_cuda: Flag set to True if CUDA should be used
        :param cuda_device_num: Device index for CUDA selection
        """
        if not isinstance(model, torch.nn.Module):
            raise ValueError('Model must be of type torch.nn.Module, not ' +  str(type(model).__name__))

        if not isinstance(eval_callback, Callable):
            raise ValueError('eval_callback must be of type Callable, not ' +  str(type(eval_callback).__name__))

        if not isinstance(unlabeled_dataset, (DataLoader, Collection)):
            raise ValueError('unlabeled_dataset must be of type DataLoader or Collection, not ' +  str(type(unlabeled_dataset).__name__))

        if not isinstance(dummy_input, (torch.Tensor, Tuple)):
            raise ValueError('dummy_input must be of type torch.Tensor or Tuple, not ' +  str(type(dummy_input).__name__))

        if isinstance(dummy_input, Tuple):
            dummy_input = torch.tensor(dummy_input)

        if not isinstance(use_cuda, bool):
            raise ValueError('use_cuda must be of type bool, not ' +  str(type(use_cuda).__name__))

        if use_cuda:
            if torch.cuda.is_available():
                torch.cuda.device(cuda_device_num)
                model.to('cuda')
                dummy_input = dummy_input.to('cuda')
            else:
                _logger.warning("use_cuda was selected but CUDA is not available. CPU will be used instead.")
                model.to('cpu')
                dummy_input = dummy_input.to('cpu')
        else:
            model.to('cpu')
            dummy_input = dummy_input.to('cpu')

        return model, dummy_input

    def inference(self,
                  param_bw: int = 8,
                  output_bw: int = 8,
                  quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                  rounding_mode: str = 'nearest',
                  config_file: str = None) -> Tuple[torch.nn.Module, float]:
        '''
        Creates a quantization model and performs inference

        :param param_bw: Parameter bitwidth
        :param output_bw: Output bitwidth
        :param quant_scheme: Quantization scheme
        :param rounding_mode: Rounding mode
        :param config_file: Path to configuration file for model quantizers
        :return: QuantizationSimModel, model accuracy as float
        '''

        # ensures that AIMET features can be applied to the model
        model = self.fp32_model

        if not ModelValidator.validate_model(model, model_input=self.dummy_input):
            try:
                model = ModelPreparer.prepare_model(model)
                _logger.info("Model preparation has passed successfully")
            except Exception:  # pylint: disable=broad-except
                if self.ignore_errors:
                    _logger.info("Model preparation has failed.")
                else:
                    raise ValueError("Model validation and model preparation have failed. Please make the necessary changes to the model and run again.")

            if not ModelValidator.validate_model(model, model_input=self.dummy_input):
                if self.ignore_errors:
                    _logger.info("Model validation has failed after attempting model preparation.")
                else:
                    raise ValueError('Model validation has failed after model preparation. Please make the necesary changes to the model and run again.')
            else:
                _logger.info("Model validation has passed successfully")
                self.fp32_model = model
        else:
            _logger.info("Model validation has passed successfully")

        validate_quantsim_inputs(quant_scheme, rounding_mode, output_bw, param_bw)

        # TODO: Save bn fold statistics to be used for optimize()?
        bn_model, _ = self._apply_batchnorm_folding()
        _logger.info("Batch norm folding is complete")
        self._bn_folding_applied = True

        sim_model = self._create_quantsim_and_encodings(model=bn_model, param_bw=param_bw, output_bw=output_bw,
                                                        rounding_mode=rounding_mode, config_file=config_file,
                                                        quant_scheme=quant_scheme)
        _logger.info("Quantization simulation is complete")
        self._quantsim_model = sim_model
        accuracy = self._evaluate_model_performance(sim_model)

        return sim_model, accuracy

    def optimize(  # pylint: disable=too-many-arguments
            self,
            param_bw: int = 8,
            output_bw: int = 8,
            quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
            rounding_mode: str = 'nearest',
            config_file: str = None,
            allowed_accuracy_drop: float = 0.0,
            cache_id: str = None,
            results_dir: str = "/tmp") -> Tuple[torch.nn.Module, float]:
        '''
        Integrate and apply post-training quantization techniques.

        :param param_bw: Parameter bitwidth
        :param output_bw: Output bitwidth
        :param quant_scheme: Quantization scheme
        :param rounding_mode: Rounding mode
        :param config_file: Path to configuration file for model quantizers
        :param allowed_accuracy_drop: Maximum allowed accuracy drop
        :param cache_id: A string that composes a cache id in combination with results_dir.
            If specified, AutoQuant will load/save the PTQ results from/to the file system
            if previous PTQ results produced under the same results_dir and cache_id exist,
        :param results_dir: Directory to save the results of PTQ techniques
        :return: Best model and its accuracy as a float
        '''

        #TODO

    def _evaluate_model_performance(self, model) -> float:
        """
        Evaluate the model performance.
        :param model: Model whose performance is being evaluated
        :return Model accuracy as float
        """
        return self.eval_callback(model, NUM_SAMPLES_FOR_PERFORMANCE_EVALUATION)

    def _apply_batchnorm_folding(self) -> Tuple[torch.nn.Module, List[Tuple]]:
        """
        Apply batchnorm folding.

        NOTE: Input model is not mutated.

        :param model: Model to apply batchnorm folding.
        :param dummy_input: Dummy input to the model.
        :return: Output model and folded pairs.
        """
        model = copy.deepcopy(self.fp32_model)
        if isinstance(self.dummy_input, torch.Tensor):
            input_shape = tuple(self.dummy_input.shape)
        else:
            input_shape = [tuple(x.shape) for x in self.dummy_input]
        folded_pairs = fold_all_batch_norms_to_weight(model, input_shape)
        return model, folded_pairs

    def _create_quantsim_and_encodings(  # pylint: disable=too-many-arguments
            self,
            model: torch.nn.Module,
            param_bw: int = 8,
            output_bw: int = 8,
            rounding_mode: str = None,
            config_file: str = None,
            quant_scheme: QuantScheme = None,
            encoding_path: str = None) -> QuantizationSimModel:
        """
        Create a QuantizationSimModel and compute encoding. If `encoding_path` is not None,
        it is prioritized over other arguments (`default_output_bw`, `defalt_param_bw`, ...).

        :param model: Model to quantize.
        :param dummy_input: Dummy input to the model.
        :param quant_scheme: Quantization scheme. Defaults to self.default_quant_scheme.
        :param rounding_mode: Rounding mode. Defaults to self.default_rounding_mode.
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs andoutputs.
                                  Defaults to self.default_output_bw.
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
                                 Defaults to self.default_param_bw.
        :param config_file: Path to configuration file for model quantizers.
                            Defaults to self.default_config_file.
        :param encoding_path: Path to parameter encodings file.
        :return: Quantsim model.
        """
        sim = QuantizationSimModel(model, self.dummy_input, default_param_bw=param_bw, default_output_bw=output_bw,
                                   rounding_mode=rounding_mode, config_file=config_file, quant_scheme=quant_scheme)

        if encoding_path:
            sim.set_and_freeze_param_encodings(encoding_path)

        sim.compute_encodings(self.forward_pass_callback, None)

        return sim
