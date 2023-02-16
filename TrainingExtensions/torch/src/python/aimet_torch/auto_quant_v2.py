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
# pylint: disable=too-many-lines

"""Temporary buffer file for adding new features to AutoQuant"""
import copy
import contextlib
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import functools
import itertools
import string
import traceback
import math
import os
import sys
import io
from unittest.mock import patch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Mapping
import torch
from torch.utils.data import DataLoader
import jinja2
from tqdm import tqdm

from aimet_torch import utils
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.utils import get_all_quantizers, in_eval_mode
from aimet_torch.onnx_utils import OnnxExportApiArgs
from aimet_torch.model_preparer import prepare_model
from aimet_torch.model_validator.model_validator import ModelValidator

from aimet_common.auto_quant import Diagnostics
from aimet_common.cache import Cache
from aimet_common.defs import QuantScheme
from aimet_common.utils import AimetLogger, Spinner
from aimet_common.quantsim import _validate_quant_scheme, _validate_rounding_mode, _validate_bitwidth


_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AutoQuant)

cache = Cache()


# The number of samples to be used for performance evaluation.
# NOTE: None means "all".
NUM_SAMPLES_FOR_PERFORMANCE_EVALUATION = None


@dataclass(frozen=True)
class _QuantSchemePair:
    param_quant_scheme: QuantScheme
    output_quant_scheme: QuantScheme
    param_percentile: Optional[float] = None
    output_percentile: Optional[float] = None


QUANT_SCHEME_CANDIDATES = (
    # Weight:     tf
    # Activation: tf
    _QuantSchemePair(QuantScheme.post_training_tf,
                     QuantScheme.post_training_tf),

    # Weight:     tf_enhanced
    # Activation: tf
    _QuantSchemePair(QuantScheme.post_training_tf_enhanced,
                     QuantScheme.post_training_tf),

    # Weight:     tf_enhanced
    # Activation: tf_enhanced
    _QuantSchemePair(QuantScheme.post_training_tf_enhanced,
                     QuantScheme.post_training_tf_enhanced),

    # Weight:     tf_enhanced
    # Activation: percentile(99.9)
    _QuantSchemePair(QuantScheme.post_training_tf_enhanced,
                     QuantScheme.post_training_percentile,
                     output_percentile=99.9),

    # Weight:     tf_enhanced
    # Activation: percentile(99.99)
    _QuantSchemePair(QuantScheme.post_training_tf_enhanced,
                     QuantScheme.post_training_percentile,
                     output_percentile=99.99),
)


def _choose_default_quant_scheme(param_bw: int,
                                 output_bw: int,
                                 eval_fn: Callable[[_QuantSchemePair], float]) -> _QuantSchemePair:
    """
    Choose a default param/output quant scheme among QUANT_SCHEME_CANDIDATES.

    :param param_bw: Parameter bitwidth
    :param output_bw: Output bitwidth
    :param eval_fn: A callable that takes a pair of quant schemes
        (for param and output respectively) and return the eval score
    :return: The quant scheme that yields the best eval score.
    """
    candidates = QUANT_SCHEME_CANDIDATES

    # If the weight representation has sufficient precision (i.e. bitwidth >= 16),
    # always use tf scheme
    if param_bw >= 16:
        candidates = [
            candidate for candidate in candidates
            if candidate.param_quant_scheme == QuantScheme.post_training_tf
        ]

    # If the output representation has sufficient precision (i.e. bitwidth >= 16),
    # always use tf scheme
    if output_bw >= 16:
        candidates = [
            candidate for candidate in candidates
            if candidate.output_quant_scheme == QuantScheme.post_training_tf
        ]

    # If we have only one candidate left, we don't need to evaluated
    # the quant scheme for comparison
    if len(candidates) == 1:
        return candidates[0]

    assert candidates

    # Find the quant scheme that yields the best eval score
    return max(candidates, key=eval_fn)


class _AutoQuantV2: # pylint: disable=too-many-instance-attributes
    """
    Integrate and apply post-training quantization techniques.

    AutoQuant includes 1) batchnorm folding, 2) cross-layer equalization,
    and 3) Adaround.
    These techniques will be applied in a best-effort manner until the model
    meets the evaluation goal given as allowed_accuracy_drop.
    """

    def __init__( # pylint: disable=too-many-arguments
            self,
            allowed_accuracy_drop: float,
            unlabeled_dataset_iterable: DataLoader,
            eval_callback: Callable[[torch.nn.Module, Optional[int]], float],
            default_param_bw: int = 8,
            default_output_bw: int = 8,
            default_quant_scheme: QuantScheme = None,
            default_rounding_mode: str = 'nearest',
            default_config_file: str = None,
    ) -> None:
        """
        :param allowed_accuracy_drop: Maximum allowed accuracy drop.
        :param unlabeled_dataset_iterable: Unlabeled data loader used for encoding computation.
                The values yielded by this data loader are expected to be able to be
                passed directly to the model. By default, this iterable will
                be also used for Adaround unless otherwise specified by
                `self.set_adaround_params`.
        :param eval_callback: A function that maps model and the number samples
                to the evaluation score. This callback is expected to return a
                scalar value representing the model performance evaluated
                against exactly `N` samples, where `N` is the number of samples
                passed as the second argument of this callback.
                NOTE: If `N` is None, the model is expected to be evaluated against
                the whole evaluation dataset.
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs andoutputs.
        :param default_quant_scheme: Quantization scheme. Supported values are
                QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced.
        :param default_rounding_mode: Rounding mode. Supported options are 'nearest' or 'stochastic'
        :param default_config_file: Path to configuration file for model quantizers
        """
        if not isinstance(unlabeled_dataset_iterable, DataLoader):
            raise TypeError(f"Expected `unlabeled_dataset_iterable` to be an instance of torch.data.DataLoader (got {type(unlabeled_dataset_iterable)})")

        if allowed_accuracy_drop < 0:
            raise ValueError(
                "`allowed_accuracy_drop` must be a positive value. Got {:.2f}"
                .format(allowed_accuracy_drop)
            )

        if default_quant_scheme:
            _validate_quant_scheme(default_quant_scheme)

        _validate_rounding_mode(default_rounding_mode)
        _validate_bitwidth(default_param_bw, default_output_bw)

        @functools.wraps(eval_callback)
        def eval_callback_wrapper(model: torch.nn.Module,
                                  num_samples: Optional[int]) -> float:
            """
            Wrapper to ensure that model is in eval mode before entering eval_callback.
            """
            with in_eval_mode(model), torch.no_grad():
                return eval_callback(model, num_samples)

        self.allowed_accuracy_drop = allowed_accuracy_drop
        self.eval_callback = eval_callback_wrapper
        self.default_param_bw = default_param_bw
        self.default_output_bw = default_output_bw
        if default_quant_scheme is not None:
            # By default, use the same quant scheme for param and output
            self.default_quant_scheme = _QuantSchemePair(default_quant_scheme,
                                                         default_quant_scheme)
        else:
            self.default_quant_scheme = None
        self.default_rounding_mode = default_rounding_mode
        self.default_config_file = default_config_file

        def forward_pass_callback(model, _: Any = None):
            device = utils.get_device(model)
            with in_eval_mode(model), torch.no_grad():
                for input_data in tqdm(unlabeled_dataset_iterable):
                    input_data = utils.change_tensor_device_placement(input_data, device)
                    if isinstance(input_data, torch.Tensor):
                        model(input_data)
                    else:
                        assert isinstance(input_data, (tuple, list))
                        model(*input_data)

        self.forward_pass_callback = forward_pass_callback

        # Use at most 2000 samples for AdaRound.
        num_samples = min(len(unlabeled_dataset_iterable.dataset), 2000)
        batch_size = unlabeled_dataset_iterable.batch_size or 1
        num_batches = math.ceil(num_samples / batch_size)
        self.adaround_params = AdaroundParameters(unlabeled_dataset_iterable, num_batches)

        self._export_kwargs = dict(
            onnx_export_args=OnnxExportApiArgs(),
            propagate_encodings=False,
        )
        self._model_preparer_kwargs = dict(
            modules_to_exclude=None,
            concrete_args=None,
        )

    def _evaluate_model_performance(self, model) -> float:
        """
        Evaluate the model performance.
        """
        return self.eval_callback(model, NUM_SAMPLES_FOR_PERFORMANCE_EVALUATION)

    def set_adaround_params(self, adaround_params: AdaroundParameters) -> None:
        """
        Set Adaround parameters.
        If this method is not called explicitly by the user, AutoQuant will use
        `unlabeled_dataset_iterable` (passed to `__init__`) for Adaround.

        :param adaround_params: Adaround parameters.
        """
        self.adaround_params = adaround_params

    def set_export_params(self,
                          onnx_export_args: OnnxExportApiArgs = -1,
                          propagate_encodings: bool = None) -> None:
        """
        Set parameters for QuantizationSimModel.export.

        :param onnx_export_args: optional export argument with onnx specific overrides
                if not provide export via torchscript graph
        :param propagate_encodings: If True, encoding entries for intermediate ops
                (when one PyTorch ops results in multiple ONNX nodes) are filled with
                the same BW and data_type as the output tensor for that series of ops.
        """
        # Here, we use -1 to indicate `onnx_export_args` wasn't specified
        # since onnx_export_args being None has its own meaning.
        if onnx_export_args != -1:
            self._export_kwargs.update(onnx_export_args=onnx_export_args)
        if propagate_encodings is not None:
            self._export_kwargs.update(propagate_encodings=propagate_encodings)

    def set_model_preparer_params(
            self,
            modules_to_exclude: List[torch.nn.Module] = None,
            concrete_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Set parameters for model preparer.

        :param modules_to_exclude: List of modules to exclude when tracing.
        :param concrete_args: Parameter for model preparer. Allows you to partially specialize
            your function, whether it's to remove control flow or data structures. If the
            model has control flow, torch.fx won't be able to trace the model. Check
            torch.fx.symbolic_trace API in detail.
        """
        self._model_preparer_kwargs["modules_to_exclude"] = copy.copy(modules_to_exclude)
        self._model_preparer_kwargs["concrete_args"] = copy.copy(concrete_args)

    def _create_quantsim_and_encodings( # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
            self,
            model: torch.nn.Module,
            dummy_input: Union[torch.Tensor, Tuple],
            rounding_mode: str = None,
            default_output_bw: int = None,
            output_quant_scheme: QuantScheme = None,
            output_percentile: float = None,
            default_param_bw: int = None,
            param_quant_scheme: QuantScheme = None,
            param_percentile: float = None,
            config_file: str = None,
            encoding_path: str = None,
    ) -> QuantizationSimModel:
        """
        Create a QuantizationSimModel and compute encoding. If `encoding_path` is not None,
        it is prioritized over other arguments (`default_output_bw`, `defalt_param_bw`, ...).

        :param model: Model to quantize.
        :param dummy_input: Dummy input to the model.
        :param rounding_mode: Rounding mode. Defaults to self.default_rounding_mode.
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs andoutputs.
            Defaults to self.default_output_bw.
        :param output_quant_scheme: Quantization scheme for output quantizers.
            Defaults to self.default_quant_scheme.output_quant_scheme.
        :param output_percentile: Percentile value for outputs.
            Only valid if output quant scheme is percentile scheme.
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
            Defaults to self.default_param_bw.
        :param param_quant_scheme: Quantization scheme for param quantizers.
            Defaults to self.default_quant_scheme.param_quant_scheme.
        :param param_percentile: Percentile value for parameters.
            Only valid if param quant scheme is percentile scheme.
        :param config_file: Path to configuration file for model quantizers.
                            Defaults to self.default_config_file.
        :param encoding_path: Path to parameter encodings file.
        :return: Quantsim model.
        """
        if default_output_bw is not None:
            assert default_output_bw <= 32

        if default_param_bw is not None:
            assert default_param_bw <= 32

        if output_quant_scheme is None or param_quant_scheme is None:
            assert self.default_quant_scheme is not None

        kwargs = dict(
            rounding_mode=(rounding_mode or self.default_rounding_mode),
            default_output_bw=(default_output_bw or self.default_output_bw),
            default_param_bw=(default_param_bw or self.default_param_bw),
            config_file=(config_file or self.default_config_file),
        )
        sim = QuantizationSimModel(model, dummy_input, **kwargs)

        param_quantizers, input_quantizers, output_quantizers = utils.get_all_quantizers(sim.model)

        if self.default_quant_scheme is not None:
            output_quant_scheme = output_quant_scheme or\
                                   self.default_quant_scheme.output_quant_scheme
            output_percentile = output_percentile or self.default_quant_scheme.output_percentile
            param_quant_scheme = param_quant_scheme or\
                                 self.default_quant_scheme.param_quant_scheme
            param_percentile = param_percentile or self.default_quant_scheme.param_percentile

        # Set input/output quantizers' quant schemes
        for quantizer in itertools.chain(input_quantizers, output_quantizers):
            quantizer.quant_scheme = output_quant_scheme
            if quantizer.quant_scheme == QuantScheme.post_training_percentile and\
                    output_percentile is not None:
                quantizer.set_percentile_value(output_percentile)

        # Set param quantizers' quant schemes
        for quantizer in param_quantizers:
            quantizer.quant_scheme = param_quant_scheme or\
                                     self.default_quant_scheme.param_quant_scheme
            if quantizer.quant_scheme == QuantScheme.post_training_percentile and\
                    param_percentile is not None:
                quantizer.set_percentile_value(param_percentile)

        if encoding_path:
            sim.set_and_freeze_param_encodings(encoding_path)

        param_quantizers, input_quantizers, output_quantizers = utils.get_all_quantizers(sim.model)

        # Disable input/output quantizers, using fp32 to simulate int32.
        if default_output_bw == 32:
            for quantizer in input_quantizers + output_quantizers:
                quantizer.enabled = False

        # Disable param quantizers, using fp32 to simulate int32.
        if default_param_bw == 32:
            for quantizer in param_quantizers:
                quantizer.enabled = False

        # Skip encoding computation if none of the quantizers are enabled
        if any(quantizer.enabled for quantizer in param_quantizers +\
                                                  input_quantizers +\
                                                  output_quantizers):
            sim.compute_encodings(self.forward_pass_callback, None)

        return sim

    @cache.mark("batchnorm_folding")
    def _apply_batchnorm_folding( # pylint: disable=no-self-use
            self,
            model: torch.nn.Module,
            dummy_input: Union[torch.Tensor, Tuple],
    ) -> Tuple[torch.nn.Module, List[Tuple]]:
        """
        Apply batchnorm folding.

        NOTE: Input model is not mutated.

        :param model: Model to apply batchnorm folding.
        :param dummy_input: Dummy input to the model.
        :return: Output model and folded pairs.
        """
        model = copy.deepcopy(model)
        if isinstance(dummy_input, torch.Tensor):
            input_shape = tuple(dummy_input.shape)
        else:
            input_shape = [tuple(x.shape) for x in dummy_input]
        folded_pairs = fold_all_batch_norms(model, input_shape)
        return model, folded_pairs

    @cache.mark("cle")
    def _apply_cross_layer_equalization( # pylint: disable=no-self-use
            self,
            model: torch.nn.Module,
            dummy_input: Union[torch.Tensor, Tuple],
    ) -> torch.nn.Module:
        """
        Apply cross-layer equalization.

        NOTE: Input model is not mutated.

        :param model: Model to apply cross-layer-equalization.
        :param dummy_input: Dummy input to the model.
        :return: Output model.
        """
        model = copy.deepcopy(model)
        if isinstance(dummy_input, torch.Tensor):
            input_shape = tuple(dummy_input.shape)
        else:
            input_shape = [tuple(x.shape) for x in dummy_input]
        equalize_model(model, input_shape)
        return model

    @cache.mark("adaround")
    def _apply_adaround(
            self,
            model: torch.nn.Module,
            dummy_input: Union[torch.Tensor, Tuple],
            results_dir: str,
    ) -> Tuple[torch.nn.Module, str]:
        """
        Apply adaround.

        NOTE1: Input model is not mutated.
        NOTE2: Parameters `param_bw_override_list` and `ignore_quant_ops_list` are always set to None.

        :param model: Model to apply adaround.
        :param dummy_input: Dummy input to the model.
        :param results_dir: Directory to save the results of AdaRound.
        :return: Output model and the path to the parameter encoding file.
        """
        # NOTE: We dont need to make a deepcopy of model here, since Adaround.apply_adaround
        # internally creates and returns a deepcopy of model.

        filename_prefix = "adaround"
        adaround_encoding_path = os.path.join(results_dir,
                                              "{}.encodings".format(filename_prefix))

        sim = self._create_quantsim_and_encodings(model, dummy_input)

        _, input_quantizers, output_quantizers = get_all_quantizers(sim.model)
        for quantizer in itertools.chain(input_quantizers, output_quantizers):
            quantizer.enabled = False

        model = Adaround._apply_adaround(sim, model, dummy_input, self.adaround_params, # pylint: disable=protected-access
                                         path=results_dir, filename_prefix=filename_prefix)

        return model, adaround_encoding_path

    def apply(
            self,
            fp32_model: torch.nn.Module,
            dummy_input_on_cpu: Union[torch.Tensor, Tuple],
            dummy_input_on_gpu: Optional[Union[torch.Tensor, Tuple]] = None,
            results_dir: str = "/tmp",
            cache_id: str = None,
            strict_validation: bool = False,
    ) -> Tuple[torch.nn.Module, float, str]:
        """
        Apply post-training quantization techniques.

        :param fp32_model: Model to apply PTQ techniques.
        :param dummy_input_on_cpu: Dummy input to the model in CPU memory.
        :param dummy_input_on_gpu: Dummy input to the model in GPU memory.
            This parameter is required if and only if the fp32_model is on GPU.
        :param results_dir: Directory to save the results.
        :param cache_id: A string that composes a cache id in combination with results_dir.
            If specified, AutoQuant will load/save the PTQ results from/to the file system
            if previous PTQ results produced under the same results_dir and cache_id exist,
        :param strict_validation: Flag set to True by default. When False, AutoQuant will
            proceed with execution and try to handle errors internally if possible. This
            may produce unideal or unintuitive results.

        :raises ValueError: If the model is on GPU but dummy_input_on_gpu is not specified.
        :raises RuntimeError: If none of the PTQ techniques were finished successfully.

        :return: Tuple of  (best model, eval score, encoding path front).
        """
        result = self._apply_helper(self._auto_quant_main,
                                    fp32_model,
                                    dummy_input_on_cpu,
                                    dummy_input_on_gpu,
                                    results_dir,
                                    cache_id,
                                    strict_validation=strict_validation)
        return result["model"],\
               result["accuracy"],\
               result["encoding_path"]

    def _apply_helper( # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, unused-argument
            self,
            auto_quant_main_fn: Callable,
            fp32_model: torch.nn.Module,
            dummy_input_on_cpu: Union[torch.Tensor, Tuple],
            dummy_input_on_gpu: Optional[Union[torch.Tensor, Tuple]] = None,
            results_dir: str = "/tmp",
            cache_id: str = None,
            concrete_args: Dict[str, Any] = None,
            strict_validation: bool = False,
    ) -> Dict[str, Any]:
        """
        Helper for self.apply().

        :param auto_quant_main_fn: Function that implements the main logic of AutoQuant.
        :param fp32_model: Model to apply PTQ techniques.
        :param dummy_input_on_cpu: Dummy input to the model in CPU memory.
        :param dummy_input_on_gpu: Dummy input to the model in GPU memory.
            This parameter is required if and only if the fp32_model is on GPU.
        :param results_dir: Directory to save the results.
        :param cache_id: A string that composes a cache id in combination with results_dir.
            If specified, AutoQuant will load/save the PTQ results from/to the file system
            if previous PTQ results produced under the same results_dir and cache_id exist,
        :param concrete_args: Parameter for model preparer. Allows you to partially specialize
            your function, whether it's to remove control flow or data structures. If the
            model has control flow, torch.fx won't be able to trace the model. Check
            torch.fx.symbolic_trace API in detail.
        :param strict_validation: Flag set to True by default. When False, AutoQuant will
            proceed with execution and try to handle errors internally if possible. This
            may produce unideal or unintuitive results.

        :raises ValueError: If the model is on GPU but dummy_input_on_gpu is not specified.
        :raises RuntimeError: If none of the PTQ techniques were finished successfully.

        :return: The best ptq result as a dictionary.
        """
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        if utils.get_device(fp32_model) == torch.device("cpu"):
            dummy_input = dummy_input_on_cpu
        else:
            if dummy_input_on_gpu is None:
                raise ValueError(
                    "If model is placed on GPU, dummy_input_on_gpu must be also provided."
                )
            dummy_input = dummy_input_on_gpu

        if cache_id is None:
            cache_dir = None
        else:
            cache_dir = os.path.join(results_dir, ".auto_quant_cache", cache_id)

        eval_manager = _EvalManager(
            quantsim_factory=self._create_quantsim_and_encodings,
            eval_func=self._evaluate_model_performance,
            dummy_input=dummy_input,
            dummy_input_on_cpu=dummy_input_on_cpu,
            results_dir=results_dir,
            strict_validation=strict_validation,
        )

        try:
            with eval_manager.analysis_session("Prepare Model") as sess:
                fp32_model = prepare_model(fp32_model, **self._model_preparer_kwargs)

                if sess.result["status"] == "error-ignored":
                    _logger.warning(
                        "Model preparation has failed."
                        " Falling back to the original model (Reason: %s)", str(sess.result["error"])
                    )


                if ModelValidator.validate_model(fp32_model, dummy_input):
                    _logger.info(
                        "Model validation has succeeded. Proceeding to AutoQuant algorithm."
                    )
                else:
                    if strict_validation:
                        raise ValueError(
                            "Model validation has failed."
                            " Please make the necessary changes to the model and run again."
                        )
                    _logger.warning(
                        "Model validation has failed. Proceeding to AutoQuant algorithm regardless."
                    )

            with in_eval_mode(fp32_model), cache.enable(cache_dir):
                _logger.info("Starting AutoQuant")

                fp32_acc = self._evaluate_model_performance(fp32_model)
                target_acc = fp32_acc - self.allowed_accuracy_drop

                _logger.info("Target eval score: %f", target_acc)
                _logger.info("FP32 eval score (W32A32): %f", fp32_acc)

                orig_quant_scheme = self.default_quant_scheme

                # Default quant scheme is not set. Choose quant scheme automatically.
                if self.default_quant_scheme is None:
                    with eval_manager.analysis_session("QuantScheme Selection") as sess:
                        def eval_fn(quant_scheme: _QuantSchemePair):
                            return sess.eval(
                                fp32_model,
                                param_quant_scheme=quant_scheme.param_quant_scheme,
                                param_percentile=quant_scheme.param_percentile,
                                output_quant_scheme=quant_scheme.output_quant_scheme,
                                output_percentile=quant_scheme.output_percentile,
                            )

                        self.default_quant_scheme = _choose_default_quant_scheme(self.default_param_bw,
                                                                                 self.default_output_bw,
                                                                                 eval_fn)
                try:
                    ret = auto_quant_main_fn(fp32_model, target_acc, dummy_input,
                                             eval_manager, results_dir, strict_validation)
                finally:
                    self.default_quant_scheme = orig_quant_scheme

                acc = ret["accuracy"]
                if acc is not None:
                    _logger.info("Best eval score: %f", acc)

                    if acc < target_acc:
                        _logger.info(
                            "AutoQuant is unable to match the target accuracy. "
                            "Consider Quantization Aware Training."
                        )

                return ret
        finally:
            eval_manager.export_diagnostics()

    def _auto_quant_main( # pylint: disable=broad-except, too-many-locals, too-many-branches
            self,
            fp32_model: torch.nn.Module,
            target_acc: float,
            dummy_input: Union[torch.Tensor, Tuple],
            eval_manager: "_EvalManager",
            results_dir: str = "/tmp",
            strict_validation: bool = False,
    ) -> Dict[str, Any]:
        """
        Helper function of apply().

        :param fp32_model: Model to apply PTQ techniques.
        :param target_acc: Target eval score.
        :param dummy_input: Dummy input to the model.
            The device of dumyy_input should be same as that of model.
        :param eval_manager: _Evalmanager object.
        :param strict_validation: Flag set to True by default. When False, AutoQuant will
            proceed with execution and try to handle errors internally if possible. This
            may produce unideal or unintuitive results.
        :param results_dir: Directory to save the results.

        :raises RuntimeError: If none of the PTQ techniques were finished successfully.

        :return: The best ptq result as a dictionary.
        """
        with eval_manager.analysis_session(f"W32 Evaluation") as sess:
            w32_eval_score = sess.eval(model=fp32_model, default_param_bw=32)

            # Early exit
            if w32_eval_score < target_acc:
                _logger.info(
                    "W32A%d eval score (%f) is lower "
                    "than the target eval score (%f). This means it is unlikely that "
                    "the target eval score can be met using PTQ techniques. "
                    "Please consider finetuning the model using range learning.",
                    self.default_output_bw, w32_eval_score, target_acc
                )

                # Since AutoQuant pipeline exited early, all the return values are set to None
                return {
                    "model": None,
                    "accuracy": None,
                    "encoding_path": None,
                    "applied_techniques": None,
                }

            sess.result["target_satisfied"] = True

        # Batchnorm Folding
        with eval_manager.ptq_session("Batchnorm Folding") as sess:
            model, _ = self._apply_batchnorm_folding(fp32_model, dummy_input)
            sess.set_ptq_result(model=model,
                                applied_techniques=["batchnorm_folding"],
                                export_kwargs=self._export_kwargs)

        best_result = eval_manager.get_best_ptq_result()
        if best_result and best_result.accuracy >= target_acc:
            sess.result["target_satisfied"] = True
            return best_result.as_dict()

        # Cross-Layer Equalization
        with eval_manager.ptq_session("Cross-Layer Equalization") as sess:
            model = self._apply_cross_layer_equalization(fp32_model, dummy_input)
            sess.set_ptq_result(model=model,
                                applied_techniques=["cross_layer_equalization"],
                                export_kwargs=self._export_kwargs)

        best_result = eval_manager.get_best_ptq_result()
        if best_result and best_result.accuracy >= target_acc:
            sess.result["target_satisfied"] = True
            return best_result.as_dict()

        if best_result is None:
            model = fp32_model
            applied_techniques = []
        else:
            if "cross_layer_equalization" not in best_result.applied_techniques:
                sess.result["effective"] = False
            model = best_result.load_model()
            applied_techniques = best_result.applied_techniques

        # AdaRound
        with eval_manager.ptq_session("AdaRound") as sess:
            model, encoding_path = self._apply_adaround(model,
                                                        dummy_input,
                                                        results_dir)
            sess.set_ptq_result(model=model,
                                encoding_path=encoding_path,
                                applied_techniques=[*applied_techniques, "adaround"],
                                export_kwargs=self._export_kwargs)

        best_result = eval_manager.get_best_ptq_result()
        if best_result:
            if "adaround" not in best_result.applied_techniques:
                sess.result["effective"] = False
            if best_result.accuracy >= target_acc:
                sess.result["target_satisfied"] = True
            return best_result.as_dict()

        raise RuntimeError("None of batchnorm folding, CLE, or Adaround "
                           "has been finished successfully.")


@dataclass
class PtqResult:
    """
    Evaluation results.
    :param tag: Identifier string of the evaluation result.
    :param model_path: Path to the serialized model.
    :param encoding_path: Path to the encoding file.
    :param accuracy: Accuracy of the model.
    """
    model_path: str
    device: torch.device
    encoding_path: str
    accuracy: float
    applied_techniques: List[str]

    def load_model(self) -> torch.nn.Module:
        """
        Load model.
        :return: Loaded model.
        """
        return torch.load(self.model_path).to(self.device)

    def as_dict(self):
        """Convert to dictionary"""
        return dict(model=self.load_model(),
                    accuracy=self.accuracy,
                    encoding_path=self.encoding_path,
                    applied_techniques=self.applied_techniques)


class _EvalManager:
    """
    Evaluation manager for AutoQuant.
    """
    def __init__(self,
                 quantsim_factory: Callable,
                 eval_func: Callable[[torch.nn.Module], float],
                 dummy_input: Union[torch.Tensor, Tuple],
                 dummy_input_on_cpu: Union[torch.Tensor, Tuple],
                 results_dir: str,
                 strict_validation: bool):
        """
        :param quantsim_factory: A factory function that returns QuantizationSimModel.
        :param eval_func: Evaluation function.
        :param dummy_input: Dummy input to the model. Assumed to be located on the same device as the model.
        :param dummy_input_on_cpu: Dummy input to the model in CPU memory.
        :param results_dir: Base directory to save the temporary serialized model.
        """
        self._quantsim_factory = quantsim_factory
        self._eval_func = eval_func
        self._dummy_input = dummy_input
        self._dummy_input_on_cpu = dummy_input_on_cpu
        self._results_dir = results_dir
        self._strict_validation = strict_validation

        os.makedirs(self._results_dir, exist_ok=True)

        self._all_sessions: List[_EvalSession] = []
        self._ptq_sessions: List[_PtqSession] = []

    def get_best_ptq_result(self) -> Optional[PtqResult]:
        """
        Get the results with the highest evaluation score among the ptq results evaluated so far.
        :return: The best evaluation result so far.
        """
        ptq_results = [sess.ptq_result for sess in self._ptq_sessions
                       if sess.ptq_result is not None]
        if not ptq_results:
            return None

        return max(ptq_results, key=lambda ptq_result: ptq_result.accuracy)

    def analysis_session(self, title: str) -> "_EvalSession":
        """
        Return a session for analysis only.
        :param title: Title of the session.
        :return: Analysis session.
        """
        return self._get_session(title, _EvalSession)

    def ptq_session(self, title: str) -> "_PtqSession":
        """
        Return a session for analysis only.
        :param title: Title of the session.
        :return: PTQ session.
        """
        sess = self._get_session(title, _PtqSession)
        self._ptq_sessions.append(sess)
        return sess

    def _get_session(self, title: str, session_cls: type):
        """
        Session factory.
        :param title: Title of the session.
        :session_cls: Class of the session.
        :return: Session object.
        """
        session = session_cls(title,
                              self._quantsim_factory,
                              self._eval_func,
                              self._dummy_input,
                              self._dummy_input_on_cpu,
                              results_dir=os.path.join(self._results_dir, ".trace"),
                              strict_validation=self._strict_validation)
        self._all_sessions.append(session)
        return session

    HTML_TEMPLATE_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "auto_quant_diagnostics_template.html",
    )
    CSS_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "auto_quant_diagnostics_template.css",
    )

    def export_diagnostics(self) -> str:
        """
        Export diagnostics in html format.
        :return: Diagnostics string in html format.
        """
        loader = jinja2.FileSystemLoader(os.path.dirname(self.HTML_TEMPLATE_FILE))
        env = jinja2.Environment(loader=loader)
        template = env.get_template(os.path.basename(self.HTML_TEMPLATE_FILE))

        if any(sess.diagnostics.contains_bokeh() for sess in self._all_sessions):
            from bokeh.resources import CDN
            head = CDN.render()
        else:
            head = ""

        body = {
            sess.title: sess.diagnostics
            for sess in self._all_sessions
            if not sess.diagnostics.is_empty()
        }

        html = template.render(head=head, body=body)

        result = OrderedDict()
        result["ptq_techniques"] = OrderedDict()

        for sess in self._all_sessions:
            if isinstance(sess, _PtqSession):
                result["ptq_techniques"][sess.title_lowercase] = sess.result
            else:
                result[sess.title_lowercase] = sess.result

        metadata = _build_flowchart_metadata(result)
        metadata.update(css=open(self.CSS_FILE).read())

        class DefaultFormatter(string.Formatter):
            """
            Formatter that fill in the format string with empty string ('') if not specified.

            For example:

            >>> DefaultFormatter().format('{first} & {second}', second='something')
            ' & something'
            """
            def get_value(self, key, args, kwargs):
                try:
                    return super().get_value(key, args, kwargs)
                except (KeyError, IndexError):
                    return ''

        html = DefaultFormatter().format(html, **metadata)

        filename = os.path.join(self._results_dir, "diagnostics.html")
        with open(filename, "w") as f:
            f.write(html)
        return html


class _EvalSession: # pylint: disable=too-many-instance-attributes
    """
    Evaluation session for AutoQuant.

    Each session object contains a title and diagnostics produced during the session.
    The collected diagnostics will be exported into a html file by _EvalManager.
    """
    def __init__(
            self,
            title: str,
            quantsim_factory: Callable,
            eval_func: Callable[[torch.nn.Module], float],
            dummy_input: Union[torch.Tensor, Tuple],
            dummy_input_on_cpu: Union[torch.Tensor, Tuple],
            results_dir: str,
            strict_validation: bool,
    ):
        """
        :param title: Title of the session.
        :param quantsim_factory: A factory function that returns QuantizationSimModel.
        :param eval_func: Evaluation function.
        :param dummy_input: Dummy input to the model. Assumed to be located on the same device as the model.
        :param dummy_input_on_cpu: Dummy input to the model in CPU memory.
        :param results_dir: Base directory to save the temporary serialized model.
        """
        self.title = title
        self._quantsim_factory = quantsim_factory
        self._eval_func = eval_func
        self._dummy_input = dummy_input
        self._dummy_input_on_cpu = dummy_input_on_cpu
        self._results_dir = results_dir
        self._strict_validation = strict_validation
        self.result = {
            "status": None,
            "error": None,
            "target_satisfied": False,
            "effective": True,
        }

        self._spinner = Spinner(self.title)

        os.makedirs(self._results_dir, exist_ok=True)

        self.diagnostics = Diagnostics()

        # Map session title to file name.
        # e.g. title: "Cross-Layer Equalization" -> filename: "cross_layer_equalization"
        self.title_lowercase = self.title.lower().replace("-", " ")
        self.title_lowercase = "_".join(self.title_lowercase.split())

        stdout_write = sys.stdout.write
        self._log = io.StringIO()

        # Redirects stdout to self._log
        def write_wrapper(*args, **kwargs):
            self._log.write(*args, **kwargs)
            return stdout_write(*args, **kwargs)

        self._stdout_redirect = patch.object(sys.stdout, "write", write_wrapper)

    def eval(self, model: torch.nn.Module, **kwargs):
        """
        Evaluate the model.
        :param model: Model to evaluate.
        :param **kwargs: Additional arguments to the quantsim factory.
        :return: Eval score.
        """
        sim = self._quantsim_factory(model, self._dummy_input, **kwargs)
        acc = self._eval_func(sim.model)
        return acc

    def __enter__(self):
        self._spinner.__enter__()
        self._stdout_redirect.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._spinner.__exit__(exc_type, exc_val, exc_tb)

        if exc_val:
            buffer = io.StringIO()
            traceback.print_exception(exc_type, exc_val, exc_tb, file=buffer)

            if self._strict_validation:
                self._log.write(buffer.getvalue())
            else:
                self._log.write(
                    "################################################################\n"
                    "################################################################\n"
                    "################################################################\n"
                    "WARNING: The following exception was raised but ignored:\n\n"
                    f"{buffer.getvalue()}"
                    "################################################################\n"
                    "################################################################\n"
                    "################################################################\n"
                )

        self._stdout_redirect.stop()
        self.diagnostics.add(self._log.getvalue())

        self.result["error"] = exc_val
        if not exc_val:
            self.result["status"] = "success"
        elif self._strict_validation:
            self.result["status"] = "error-failed"
        else:
            self.result["status"] = "error-ignored"

        if exc_val and not self._strict_validation:
            # Return True so that the error doesn't propagate further
            return True
        return None


class _PtqSession(_EvalSession):
    """
    PTQ session.

    Each PTQ session object should call `set_ptq_result` exactly once
    inside a with-as block.
    """
    def __init__(self, *args, **kwargs):
        super(_PtqSession, self).__init__(*args, **kwargs)
        self._ptq_result = None

    @property
    def ptq_result(self) -> Optional[PtqResult]:
        """Getter of self._ptq_result."""
        return self._ptq_result

    def set_ptq_result(
            self,
            applied_techniques: List[str],
            model: torch.nn.Module = None,
            sim: QuantizationSimModel = None,
            acc: float = None,
            export_kwargs: Mapping = None,
            **kwargs
    ) -> None:
        """
        Set the result of PTQ. Should be called exactly once inside a with-as block.

        Exactly one among model and (sim, acc) pair should be specified.
        1) If sim and acc is specified, save them as the result of this session.
        2) If model is specified, evaluate the quantized accuracy of the model and save the result.

        :param model: Result of PTQ.
        :param sim: Result of PTQ. The quamtization encoding (compute_encodings()) is
                    assumed to have been computed in advance.
        :param acc: Eval score.
        :param **kwargs: Additional arguments to the quantsim factory.
        :return: None
        """
        if export_kwargs is None:
            export_kwargs = {}

        if sim is None:
            assert acc is None
            assert model is not None
            sim = self._quantsim_factory(model, self._dummy_input, **kwargs)
            acc = self._eval_func(sim.model)
        else:
            assert acc is not None
            assert model is None

        self._set_ptq_result(sim, acc, applied_techniques, export_kwargs)

    def _set_ptq_result(
            self,
            sim: QuantizationSimModel,
            acc: float,
            applied_techniques: List[str],
            export_kwargs: Mapping,
    ) -> PtqResult:
        """
        Set the result of PTQ. Should be called exactly once inside a with-as block.

        :param sim: Result of PTQ. The quamtization encoding (compute_encodings()) is
                    assumed to have been computed in advance.
        :param acc: Eval score.
        :param export_kwargs: Additional kwargs for sim.export
        :return: PtqResult object.
        """
        if self._ptq_result is not None:
            raise RuntimeError(
                "sess.eval() can be called only once per each _EvalSession instance."
            )

        device = utils.get_device(sim.model)
        model_path, encoding_path = self._export(sim, export_kwargs)
        self._ptq_result = PtqResult(
            model_path=model_path,
            device=device,
            encoding_path=encoding_path,
            accuracy=acc,
            applied_techniques=applied_techniques,
        )
        return self._ptq_result

    def _export(self, sim: QuantizationSimModel, export_kwargs: Mapping) -> Tuple[str, str]:
        """
        Export quantsim.
        :param sim: QuantizationSimModel object to export.
        :param export_kwargs: Additional kwargs for sim.export
        :return: The paths where model and encoding are saved
        """
        sim.export(path=self._results_dir,
                   filename_prefix=self.title_lowercase,
                   dummy_input=self._dummy_input_on_cpu,
                   **export_kwargs)
        model_path = os.path.join(self._results_dir, f"{self.title_lowercase}.pth")
        encoding_path = os.path.join(self._results_dir, f"{self.title_lowercase}.encodings")
        _logger.info("The results of %s is saved in %s and %s.",
                     self.title, model_path, encoding_path)
        return model_path, encoding_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Raises error if set_ptq_result is not called."""
        if self._ptq_result is not None:
            _logger.info("Session finished: %s. (eval score: %f)",
                         self.title, self._ptq_result.accuracy)
        return super(_PtqSession, self).__exit__(exc_type, exc_val, exc_tb)


@contextlib.contextmanager
def spy_auto_quant(auto_quant: _AutoQuantV2):
    """
    Install a spy that collects the handles to the ptq result of
    each stage of AutoQuant.

    Typical usage::
        >>> auto_quant = AutoQuant(...)
        ... with auto_quant_spy(auto_quant) as spy:
        ...     _ = auto_quant.apply(...)
        ...
        ... for result in spy.get_all_ptq_results():
        ...     print(result.applied_techniques)
        ...     print(result.accuracy)
        ...     print(result.encoding_path)
        ...     model = result.load_model()
        ...     ...
    """
    # pylint: disable=protected-access
    class Spy:
        """
        Spy that collects the handles to the ptq result of
        each stage of AutoQuant.
        """
        def __init__(self):
            self._eval_manager = None

        def get_all_ptq_results(self) -> List[PtqResult]:
            """Return handles to the results of AutoQuant"""
            if self._eval_manager is None:
                return []
            return [sess.ptq_result for sess in self._eval_manager._ptq_sessions]

    spy = Spy()

    _auto_quant_main = auto_quant._auto_quant_main

    def _auto_quant_main_wrapper(fp32_model, target_acc, dummy_input,
                                 eval_manager, results_dir="/tmp"):
        spy._eval_manager = eval_manager
        return _auto_quant_main(fp32_model, target_acc, dummy_input,
                                eval_manager, results_dir)

    try:
        setattr(auto_quant, "_auto_quant_main", _auto_quant_main_wrapper)
        yield spy
    finally:
        setattr(auto_quant, "_auto_quant_main", _auto_quant_main)


def _build_flowchart_metadata(result: Mapping) -> Dict: # pylint: disable=too-many-return-statements
    """
    Build flowchart metadata for the html template of summary report

    :param result: Result of AutoQuant with the following format:

        result := {
            "prepare_model": _stage_result,
            "quantscheme_selection": _stage_result,
            "w32_evaluation": _stage_result,
            "ptq_techniques" [
                "batchnorm_folding": _stage_result,
                "cross_layer_equalization": _stage_result,
                "adaround": _stage_result,
            ]

        }

        where _stage_result is a dictionary defined as below:

        _stage_result := {
            "status": str,
            "error": Exception,
            "target_satisfied": bool,
            "effective": bool,
        }

    :return: Dictionary that contains flowchart metadata for html template
    """
    metadata = defaultdict(str)
    metadata.update(
        edge_prepare_model_in='data-visited="true"',
        node_prepare_model='data-visited="true"',
    )

    status = result['prepare_model']['status']
    metadata.update(
        node_prepare_model=f'data-visited="true" data-stage-result="{status}"',
    )

    if status == 'error-failed':
        return metadata

    metadata.update(
        edge_prepare_model_out='data-visited="true"',
    )

    if "quantscheme_selection" in result:
        status = result['quantscheme_selection']['status']
        metadata.update(
            node_quant_scheme_selection=f'data-visited="true" data-stage-result="{status}"',
        )

        if status == 'error-failed':
            return metadata

    metadata.update(
        edge_quant_scheme_selection_out='data-visited="true"',
        node_test_w32_eval_score='data-visited="true"',
    )

    if not result["w32_evaluation"]["target_satisfied"]:
        metadata.update(
            edge_test_w32_eval_score_if_false='data-visited="true"',
            node_result_fail='data-visited="true"',
        )
        return metadata

    metadata.update(
        edge_test_w32_eval_score_if_true='data-visited="true"',
    )


    for ptq_name, ptq_result in result["ptq_techniques"].items():
        status = ptq_result['status']
        effective = ptq_result['effective']
        if status == "success" and not effective:
            status = "discarded"
        metadata.update({
            f"node_{ptq_name}": f'data-visited="true" data-stage-result="{status}"',
        })

        if status == 'error-failed':
            return metadata

        metadata.update({
            f'edge_{ptq_name}_out': 'data-visited="true"',
            f'node_test_{ptq_name}': 'data-visited="true"',
        })

        if ptq_result['target_satisfied']:
            metadata.update({
                f'edge_test_{ptq_name}_if_true': 'data-visited="true"',
                'node_result_success': 'data-visited="true"',
            })
            return metadata

        metadata.update({
            f'edge_test_{ptq_name}_if_false': 'data-visited="true"',
        })

    metadata.update(
        node_result_fail='data-visited="true"',
    )

    return metadata
