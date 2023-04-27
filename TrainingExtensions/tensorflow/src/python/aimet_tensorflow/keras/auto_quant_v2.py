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

"""Automatic Post-Training Quantization"""
import itertools
import math
import os
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import List, Callable, Dict, Any, Tuple, Optional, Mapping
import io
from unittest.mock import patch
import traceback
import shutil
from numpy.ma import copy
from tqdm import tqdm
import jinja2
import tensorflow as tf

from aimet_common.cache import Cache
from aimet_common.defs import QuantScheme
from aimet_common.quantsim import validate_quantsim_inputs
from aimet_common.utils import AimetLogger, Spinner
from aimet_common.auto_quant import Diagnostics
from aimet_tensorflow.adaround.adaround_weight import AdaroundParameters
from aimet_tensorflow.keras.adaround_weight import Adaround
from aimet_tensorflow.keras.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow.keras.cache import KerasModelSerializationProtocol
from aimet_tensorflow.keras.cross_layer_equalization import equalize_model
from aimet_tensorflow.keras.quantsim import QuantizationSimModel


_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AutoQuant) # pylint: disable=invalid-name

cache = Cache() # pylint: disable=invalid-name

# The number of samples to be used for performance evaluation.
# NOTE: None means "all".
NUM_SAMPLES_FOR_PERFORMANCE_EVALUATION = None

@dataclass(frozen=True)
class _QuantSchemePair:
    param_quant_scheme: QuantScheme
    output_quant_scheme: QuantScheme
    param_percentile: Optional[float] = None
    output_percentile: Optional[float] = None

    def __str__(self):
        def scheme_to_str(quant_scheme, percentile):
            if quant_scheme == QuantScheme.post_training_percentile:
                return f"{percentile}%ile"
            if quant_scheme in (QuantScheme.post_training_tf,
                                QuantScheme.training_range_learning_with_tf_init):
                return "tf"
            if quant_scheme in (QuantScheme.post_training_tf_enhanced,
                                QuantScheme.training_range_learning_with_tf_enhanced_init):
                return "tf-enhanced"
            raise ValueError

        param_str = scheme_to_str(self.param_quant_scheme, self.param_percentile)
        output_str = scheme_to_str(self.output_quant_scheme, self.output_percentile)
        return f"W@{param_str} / A@{output_str}"


_QUANT_SCHEME_CANDIDATES = (
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

def _validate_inputs(model: tf.keras.Model, # pylint: disable=too-many-arguments
                     dataset: tf.data.Dataset,
                     eval_callback: Callable[[tf.keras.Model], float],
                     results_dir: str,
                     strict_validation: bool,
                     quant_scheme: QuantScheme,
                     param_bw: int,
                     output_bw: int,
                     rounding_mode: str):
    """
    Confirms inputs are of the correct type
    :param model: Model to be quantized
    :param dataset: A collection that iterates over an unlabeled dataset,
        used for computing encodings
    :param eval_callback: Function that calculates the evaluation score
    :param results_dir: Directory to save the results of PTQ techniques
    :param strict_validation: Flag set to True by default.
        When False, AutoQuant will proceed with execution
        and try to handle errors internally if possible.
        This may produce unideal or unintuitive results.
    :param quant_scheme: Quantization scheme
    :param param_bw: Parameter bitwidth
    :param output_bw: Output bitwidth
    :param rounding_mode: Rounding mode
    """
    if not isinstance(model, tf.keras.Model):
        raise ValueError('Model must be of type tf.keras.Model, not ' + str(type(model).__name__))

    if not isinstance(dataset, tf.data.Dataset):
        raise ValueError('dataset must be of type tf.data.Dataset, not ' + str(
            type(dataset).__name__))

    if not isinstance(eval_callback, Callable):
        raise ValueError('eval_callback must be of type Callable, not ' +
                         str(type(eval_callback).__name__))

    if not isinstance(results_dir, str):
        raise ValueError('results_dir must be of type str, not ' +
                         str(type(results_dir).__name__))

    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    if not isinstance(strict_validation, bool):
        raise ValueError('strict_validation must be of type bool, not ' +
                         str(type(strict_validation).__name__))

    validate_quantsim_inputs(quant_scheme, rounding_mode, output_bw, param_bw)



class AutoQuant: # pylint: disable=too-many-instance-attributes
    """
    Integrate and apply post-training quantization techniques.

    AutoQuant includes 1) batchnorm folding, 2) cross-layer equalization,
    and 3) Adaround.
    These techniques will be applied in a best-effort manner until the model
    meets the evaluation goal given as allowed_accuracy_drop.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, # pylint: disable=too-many-arguments, too-many-locals
                 model: tf.keras.Model,
                 eval_callback: Callable[[tf.keras.Model], float],
                 dataset: tf.data.Dataset,
                 param_bw: int = 8,
                 output_bw: int = 8,
                 quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                 rounding_mode: str = "nearest",
                 config_file: str = None,
                 results_dir: str = "/tmp",
                 cache_id: str = None,
                 strict_validation: bool = True) -> None:
        '''
        :param model: Model to be quantized. Assumes model is on the correct device
        :param eval_callback: Function that calculates the evaluation score
        :param dataset: A collection that iterates over an unlabeled dataset,
            used for computing encodings
        :param param_bw: Parameter bitwidth
        :param output_bw: Output bitwidth
        :param quant_scheme: Quantization scheme
        :param rounding_mode: Rounding mode
        :param config_file: Path to configuration file for model quantizers
        :param results_dir: Directory to save the results of PTQ techniques
        :param cache_id: ID associated with cache results
        :param strict_validation: Flag set to True by default.
            When False, AutoQuant will proceed with execution and
            handle errors internally if possible.
            This may produce unideal or unintuitive results.
        '''

        _validate_inputs(model, dataset, eval_callback, results_dir, strict_validation, \
                         quant_scheme, param_bw, output_bw, rounding_mode)
        self.fp32_model = model
        self.dataset = dataset
        self.eval_callback = eval_callback

        self._quantsim_params = dict(
            param_bw=param_bw,
            output_bw=output_bw,
            quant_scheme=_QuantSchemePair(quant_scheme, quant_scheme),
            rounding_mode=rounding_mode,
            config_file=config_file,
        )

        self.results_dir = results_dir
        if cache_id:
            self.cache_dir = os.path.join(results_dir, ".auto_quant_cache", cache_id)
        else:
            self.cache_dir = None

        def forward_pass_callback(model, _: Any = None):
            for input_data in tqdm(self.dataset):
                model(input_data)

        self.forward_pass_callback = forward_pass_callback

        # Use at most 2000 samples for AdaRound.
        batch_size = None
        for data in self.dataset:
            if not batch_size:
                batch_size = len(data)
                break
        batch_size = batch_size or 1
        num_batches = math.ceil(2000 / batch_size)
        num_batches = min(num_batches, len(self.dataset))

        self.adaround_params = AdaroundParameters(self.dataset,
                                                  num_batches)
        self.eval_manager = _EvalManager(
            quantsim_factory=self._create_quantsim_and_encodings,
            eval_func=self._evaluate_model_performance,
            results_dir=self.results_dir,
            strict_validation=strict_validation
        )
        self._quant_scheme_candidates = _QUANT_SCHEME_CANDIDATES

    def _evaluate_model_performance(self, model: tf.keras.Model) -> float:
        """
        Evaluate the model performance

        :param model: Model to evaluate
        :return: Evaluation score
        """
        return self.eval_callback(model, NUM_SAMPLES_FOR_PERFORMANCE_EVALUATION)

    def run_inference(self) -> Tuple[QuantizationSimModel, float]:

        '''
        Creates a quantization model and performs inference

        :return: QuantizationSimModel, model accuracy as float
        '''

        model = self.fp32_model

        with self.eval_manager.session("Batchnorm Folding", ptq=True) as sess:
            model, _ = self._apply_batchnorm_folding(model)
            if sess.ptq_result is None:
                sess.set_ptq_result(applied_techniques=["batchnorm_folding"], model=model,)

        sim = self._create_quantsim_and_encodings(model)

        if sess.ptq_result is None:
            # BN folding failed. Need to measure the eval score
            acc = self._evaluate_model_performance(sim.model)
        else:
            # BN folding success. No need to measure the eval score again
            acc = sess.ptq_result.accuracy

        return sim, acc

    def optimize(self, allowed_accuracy_drop: float = 0.0) -> Tuple[tf.keras.Model, float, str]:
        """
        Integrate and apply post-training quantization techniques.

        :param allowed_accuracy_drop: Maximum allowed accuracy drop
        :return: Tuple of (best model, eval score, encoding path)
        """
        result = self._optimize_helper(self._optimize_main, allowed_accuracy_drop)
        return result["model"], \
            result["accuracy"], \
            result["encoding_path"]

    def set_adaround_params(self, adaround_params: AdaroundParameters):
        """
        Set Adaround parameters.
        If this method is not called explicitly by the user, AutoQuant will use
        `dataset` (passed to `__init__`) for Adaround.

        :param adaround_params: Adaround parameters.
        """
        self.adaround_params = adaround_params


    def _create_quantsim_and_encodings(self, # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
                                       model: tf.keras.Model,
                                       rounding_mode: str = None,
                                       output_bw: int = None,
                                       output_quant_scheme: QuantScheme = None,
                                       output_percentile: float = None,
                                       param_bw: int = None,
                                       param_quant_scheme: QuantScheme = None,
                                       param_percentile: float = None,
                                       config_file: str = None,
                                       encoding_path: str = None) -> QuantizationSimModel:
        """

        :param rounding_mode: Rounding mode. Defaults to self._quantsim_params["rounding_mode"].
        :param output_bw: Default bitwidth (4-31) to use for quantizing layer inputs andoutputs.
            Defaults to self._quantsim_params["output_bw"].
        :param output_quant_scheme: Quantization scheme for output quantizers.
            Defaults to self._quantsim_params["quant_scheme"].output_quant_scheme.
        :param output_percentile: Percentile value for outputs.
            Only valid if output quant scheme is percentile scheme.
        :param param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
            Defaults to self._quantsim_params["param_bw"].
        :param param_quant_scheme: Quantization scheme for param quantizers.
            Defaults to self._quantsim_params["quant_scheme"].param_quant_scheme.
        :param param_percentile: Percentile value for parameters.
            Only valid if param quant scheme is percentile scheme.
        :param config_file: Path to configuration file for model quantizers.
                            Defaults to self._quantsim_params["config_file"].
        :param encoding_path: Path to parameter encodings file.
        :return: Quantsim model.
        """
        if output_bw is not None:
            assert output_bw <= 32

        if param_bw is not None:
            assert param_bw <= 32

        if output_quant_scheme is None or param_quant_scheme is None:
            assert self._quantsim_params["quant_scheme"] is not None

        kwargs = dict(
            rounding_mode=(rounding_mode or self._quantsim_params["rounding_mode"]),
            default_output_bw=(output_bw or self._quantsim_params["output_bw"]),
            default_param_bw=(param_bw or self._quantsim_params["param_bw"]),
            config_file=(config_file or self._quantsim_params["config_file"]),
        )
        sim = QuantizationSimModel(model, **kwargs)
        input_quantizers, param_quantizers, output_quantizers = sim._get_quantizer_list() # pylint: disable=protected-access

        default_quant_scheme = self._quantsim_params.get("quant_scheme")

        if default_quant_scheme is not None:
            output_quant_scheme = output_quant_scheme or \
                                  default_quant_scheme.output_quant_scheme
            output_percentile = output_percentile or default_quant_scheme.output_percentile
            param_quant_scheme = param_quant_scheme or \
                                 default_quant_scheme.param_quant_scheme
            param_percentile = param_percentile or default_quant_scheme.param_percentile

        # Set input/output quantizers' quant schemes
        for quantizer in itertools.chain(input_quantizers, output_quantizers):
            quantizer.quant_scheme = output_quant_scheme
            if quantizer.quant_scheme == QuantScheme.post_training_percentile and \
                    output_percentile is not None:
                quantizer.set_percentile_value(output_percentile)

        # Set param quantizers' quant schemes
        for quantizer in param_quantizers:
            quantizer.quant_scheme = param_quant_scheme
            if quantizer.quant_scheme == QuantScheme.post_training_percentile and \
                    param_percentile is not None:
                quantizer.set_percentile_value(param_percentile)

        if encoding_path:
            sim.set_and_freeze_param_encodings(encoding_path)

        input_quantizers, param_quantizers, output_quantizers = sim._get_quantizer_list() # pylint: disable=protected-access

        # Disable input/output quantizers, using fp32 to simulate int32.
        if output_bw == 32:
            for quantizer in input_quantizers + output_quantizers:
                quantizer.disable()
        # Disable param quantizers, using fp32 to simulate int32.
        if param_bw == 32:
            for quantizer in param_quantizers:
                quantizer.disable()

        # Skip encoding computation if none of the quantizers are enabled
        if any(quantizer.is_enabled() for quantizer in param_quantizers + \
                                                  input_quantizers + \
                                                  output_quantizers):
            sim.compute_encodings(self.forward_pass_callback, None)
        return sim

    # pylint: disable=no-self-use
    def _apply_batchnorm_folding(self, model: tf.keras.Model) -> Tuple[tf.keras.Model, List[Tuple]]:
        """
        Apply batchnorm folding
        Note: Input model is not mutated
        :param model: Model to apply batchnorm folding
        :return: Output model and folded pairs
        """
        original_weight = model.get_weights()
        model = tf.keras.models.clone_model(model)
        model.set_weights(original_weight)
        folded_pairs, model = fold_all_batch_norms(model)
        return model, folded_pairs

    # pylint: disable=no-self-use
    @cache.mark("cle", KerasModelSerializationProtocol())
    def _apply_cross_layer_equalization(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Apply cross-layer equalization
        Note: Input model is not mutated
        :param model: Model to apply cross-layer-equalization
        :return: CLE applied model
        """
        original_weight = model.get_weights()
        model = tf.keras.models.clone_model(model)
        model.set_weights(original_weight)
        return equalize_model(model)

    def _apply_adaround(self,
                        model: tf.keras.Model) -> Tuple[tf.keras.Model, str]:
        """
        Apply adaround

        NOTE: Input model is not mutated
        :param model: Model to apply adaround
        :param results_dir: Directory to save the results of AdaRound
        :return: Output model and the path to the parameter encoding file
        """
        filename_prefix = "adaround"
        adaround_encoding_path = os.path.join(self.results_dir,
                                              f"{filename_prefix}.encodings")
        if self._quantsim_params["param_bw"] == 4:
            self.adaround_params.num_iterations = 15000

        _apply_adaround_cached = cache.mark("adaround", KerasModelSerializationProtocol()) \
            (Adaround.apply_adaround)

        model = _apply_adaround_cached(model, # pylint: disable=protected-access
                                       self.adaround_params,
                                       path=self.results_dir,
                                       filename_prefix=filename_prefix,
                                       default_param_bw=self._quantsim_params["param_bw"],
                                       default_quant_scheme=self._quantsim_params.get("quant_scheme").param_quant_scheme,
                                       config_file=self._quantsim_params["config_file"])

        return model, adaround_encoding_path

    def _optimize_helper(
            self,
            optimize_fn: Callable,
            allowed_accuracy_drop: float) -> Tuple[tf.keras.Model, float, str]:
        """
        Integrate and apply post-training quantization techniques.

        :param allowed_accuracy_drop: Maximum allowed accuracy drop
        :return: Tuple of (best model, eval score, encoding path)
        """
        allowed_accuracy_drop = float(allowed_accuracy_drop)
        if allowed_accuracy_drop < 0:
            raise ValueError(
                "`allowed_accuracy_drop` must be a positive value. Got {:.2f}"
                .format(allowed_accuracy_drop)
            )

        self.eval_manager.clear()

        try:
            with cache.enable(self.cache_dir):
                _logger.info("Starting AutoQuant")
                fp32_acc = self._evaluate_model_performance(self.fp32_model)
                target_acc = fp32_acc - allowed_accuracy_drop
                _logger.info("Target eval score: %f", target_acc)
                _logger.info("FP32 eval score (W32A32): %f", fp32_acc)

                ret = optimize_fn(self.fp32_model, target_acc)

                acc = ret["accuracy"]
                if acc is not None:
                    _logger.info("Best eval score: %f", acc)

                    # Save the best model with "best_model_" as prefix
                    best_res = self.eval_manager.get_best_ptq_result()
                    best_res.save_result_as("best_model")

                    if acc < target_acc:
                        _logger.info(
                            "AutoQuant is unable to match the target accuracy. "
                            "Consider Quantization Aware Training."
                        )

                return ret
        finally:
            self.eval_manager.export_diagnostics()

    def get_quant_scheme_candidates(self) -> Tuple[_QuantSchemePair, ...]:
        """
        Return the candidates for quant scheme search.
        During :meth:`~AutoQuant.optimize`, the candidate with the highest accuracy
        will be selected among them.

        :return: Candidates for quant scheme search
        """
        return self._quant_scheme_candidates

    def set_quant_scheme_candidates(self, candidates: Tuple[_QuantSchemePair, ...]):
        """
        Set candidates for quant scheme search.
        During :meth:`~AutoQuant.optimize`, the candidate with the highest accuracy
        will be selected among them.

        :param candidates: Candidates for quant scheme search
        """
        self._quant_scheme_candidates = copy.copy(candidates) # pylint: disable=E1101

    def _choose_default_quant_scheme(self):
        def eval_fn(pair: _QuantSchemePair):
            sim = self._create_quantsim_and_encodings(
                self.fp32_model,
                param_quant_scheme=pair.param_quant_scheme,
                param_percentile=pair.param_percentile,
                output_quant_scheme=pair.output_quant_scheme,
                output_percentile=pair.output_percentile,
            )
            return self._evaluate_model_performance(sim.model)

        param_bw = self._quantsim_params["param_bw"]
        output_bw = self._quantsim_params["output_bw"]

        candidates = self.get_quant_scheme_candidates()

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


    def _optimize_main(self, fp32_model: tf.keras.Model, target_acc: float):
        """
        Helper function of apply().

        :param fp32_model: Model to apply PTQ techniques.
        :param target_acc: Target eval score.

        :raises RuntimeError: If none of the PTQ techniques were finished successfully.

        :return: The best ptq result as a dictionary.
        """

        # Choose quant scheme automatically.
        with self.eval_manager.session("QuantScheme Selection") as sess:
            self._quantsim_params["quant_scheme"] = self._choose_default_quant_scheme()

        with self.eval_manager.session(f"W32 Evaluation") as sess:
            w32_eval_score = sess.eval(model=fp32_model, param_bw=32)
            _logger.info("Evaluation finished: W32A%d (eval score: %f)",
                         self._quantsim_params["output_bw"], w32_eval_score)

            # Early exit
            if w32_eval_score < target_acc:
                _logger.info(
                    "W32A%d eval score (%f) is lower "
                    "than the target eval score (%f). This means it is unlikely that "
                    "the target eval score can be met using PTQ techniques. "
                    "Please consider finetuning the model using range learning.",
                    self._quantsim_params["output_bw"], w32_eval_score, target_acc
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
        with self.eval_manager.session("Batchnorm Folding", ptq=True) as sess:
            model, _ = self._apply_batchnorm_folding(fp32_model)
            if sess.ptq_result is None:
                sess.set_ptq_result(model=model,
                                    applied_techniques=["batchnorm_folding"])

        best_result = self.eval_manager.get_best_ptq_result()
        if best_result and best_result.accuracy >= target_acc:
            sess.result["target_satisfied"] = True
            return best_result.as_dict()

        # Cross-Layer Equalization
        with self.eval_manager.session("Cross-Layer Equalization", ptq=True) as sess:
            model = self._apply_cross_layer_equalization(fp32_model)
            if sess.ptq_result is None:
                sess.set_ptq_result(model=model,
                                    applied_techniques=["cross_layer_equalization"])

        best_result = self.eval_manager.get_best_ptq_result()
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
        with self.eval_manager.session("AdaRound", ptq=True) as sess:
            model, encoding_path = self._apply_adaround(model)
            if sess.ptq_result is None:
                sess.set_ptq_result(model=model,
                                    encoding_path=encoding_path,
                                    applied_techniques=[*applied_techniques, "adaround"])

        best_result = self.eval_manager.get_best_ptq_result()
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
    Evaluation results
    :param model_path: Path to the serialized model.
    :param encoding_path: Path to the encoding file.
    :param accuracy: Accuracy of the model.
    :param applied_techniques: Applied ptq techniques.
    """
    model_path: str
    encoding_path: str
    accuracy: float
    applied_techniques: List[str]

    def load_model(self):
        """
        Load model
        :return: Loaded model
        """
        return tf.keras.models.load_model(self.model_path)
    def save_result_as(self, prefix: str = "best_model"):
        """
        Creates the copy of the PTQ result files with the given prefix.
        :param prefix: prefix to be added to the file's basename
        """
        src_files = [self.model_path+".h5", self.model_path+"_converted.pb", self.encoding_path]
        for file in src_files:
            name = os.path.basename(file)
            dirname = os.path.dirname(file)
            dest = os.path.join(dirname, prefix + "_" + name)
            if os.path.exists(file):
                if os.path.exists(dest):
                    os.remove(dest)
                shutil.copyfile(file, dest)
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
                 eval_func: Callable[[tf.keras.Model], float],
                 results_dir: str,
                 strict_validation: bool):
        """
        :param quantsim_factory: A factory function that returns QuantizationSimModel.
        :param eval_func: Evaluation function.
        :param results_dir: Base directory to save the temporary serialized model.
        :param strict_validation: Flag set to True by default.
            When False, AutoQuant will proceed with execution and
            handle errors internally if possible.
            This may produce unideal or unintuitive results.
        """
        self._quantsim_factory = quantsim_factory
        self._eval_func = eval_func
        self._results_dir = results_dir
        self._strict_validation = strict_validation
        os.makedirs(self._results_dir, exist_ok=True)

        self._all_sessions = OrderedDict()

    def clear(self):
        """
        Clear all the session status saved in the previous run
        """
        for sess in self._all_sessions.values():
            sess.reset_status()

    def get_best_ptq_result(self) -> PtqResult:
        """
        Get the results with the highest evaluation score among the ptq results evaluated so far.
        :return: The best evaluation result so far.
        """
        ptq_results = [sess.ptq_result for sess in self._all_sessions.values()
                       if sess.ptq_result is not None]
        if not ptq_results:
            return None

        return max(ptq_results, key=lambda ptq_result: ptq_result.accuracy)

    def session(self, title: str, ptq: bool = False):
        """
        Session factory.
        :param title: Title of the session.
        :param ptq: True if this session is a ptq session
        :return: Session object.
        """
        if title not in self._all_sessions:
            session = _EvalSession(title,
                                   self._quantsim_factory,
                                   self._eval_func,
                                   results_dir=os.path.join(self._results_dir, ".trace"),
                                   strict_validation=self._strict_validation,
                                   ptq=ptq)
            self._all_sessions[title] = session
        return self._all_sessions[title]

    HTML_TEMPLATE_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "auto_quant_v2_diagnostics_template.html",
    )

    def export_diagnostics(self) -> str:
        """
        Export diagnostics in html format.
        :return: Diagnostics string in html format.
        """
        loader = jinja2.FileSystemLoader(os.path.dirname(os.path.abspath(self.HTML_TEMPLATE_FILE)))
        env = jinja2.Environment(loader=loader)
        template = env.get_template(os.path.basename(self.HTML_TEMPLATE_FILE))

        if any(sess.diagnostics.contains_bokeh() for sess in self._all_sessions.values()):
            from bokeh.resources import CDN
            head = CDN.render()
        else:
            head = ""

        log = io.StringIO()
        for sess in self._all_sessions.values():
            if sess.diagnostics.is_empty():
                continue
            log.write(
                f"<h1> {sess.title} </h1>\n"
            )
            content = "\n".join(
                line.get_html_elem() for line in sess.diagnostics
            )
            log.write(f"{content}\n")

        result = OrderedDict()
        result["ptq_techniques"] = OrderedDict()

        for sess in self._all_sessions.values():
            if sess.is_ptq_session():
                result["ptq_techniques"][sess.title_lowercase] = sess.result
            else:
                result[sess.title_lowercase] = sess.result

        flowchart_metadata = _build_flowchart_metadata(result)

        html = template.render(head=head, log=log.getvalue(), **flowchart_metadata)

        filename = os.path.join(self._results_dir, "diagnostics.html")
        with open(filename, "w") as file_name:
            file_name.write(html)
        return html


class _EvalSession: # pylint: disable=too-many-instance-attributes
    """
    Evaluation session for AutoQuant.

    Each session object contains a title and diagnostics produced during the session.
    The collected diagnostics will be exported into a html file by _EvalManager.
    """
    def __init__(self, # pylint: disable=too-many-arguments
                 title: str,
                 quantsim_factory: Callable,
                 eval_func: Callable[[tf.keras.Model], float],
                 results_dir: str,
                 strict_validation: bool,
                 ptq: bool,):
        """
        :param title: Title of the session.
        :param quantsim_factory: A factory function that returns QuantizationSimModel.
        :param eval_func: Evaluation function.
        :param results_dir: Base directory to save the temporary serialized model.
        :param strict_validation: Flag set to True by default.
            When False, AutoQuant will proceed with execution
            and handle errors internally if possible.
            This may produce unideal or unintuitive results.
        :param ptq: True if this session is a ptq session
        """
        self.title = title
        self._quantsim_factory = quantsim_factory
        self._eval_func = eval_func
        self._results_dir = results_dir
        self._strict_validation = strict_validation
        self._ptq = ptq
        self._spinner = None

        self.result = {
            "status": None,
            "error": None,
            "target_satisfied": False,
            "effective": True,
        }

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
        self._ptq_result = None
        self._cached_result = None

    def is_ptq_session(self):
        """
        Getter method of self._ptq flag
        """
        return self._ptq

    def reset_status(self):
        """
        Reset the session status saved in the previous run
        """
        self.result = {
            "status": None,
            "error": None,
            "target_satisfied": False,
            "effective": True,
        }

    def eval(self, model: tf.keras.Model, **kwargs):
        """
        Evaluate the model
        :param model: Model to evaluate.
        :param kwargs: Additional arguments to the quantsim factory.
        :return: Eval score
        """
        sim = self._quantsim_factory(model, **kwargs)
        acc = self._eval_func(sim.model)
        return acc

    def __enter__(self):
        self._spinner = Spinner(self.title)
        self._spinner.__enter__()
        self._stdout_redirect.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._ptq_result is not None:
            _logger.info("Session finished: %s. (eval score: %f)",
                         self.title, self._ptq_result.accuracy)
        self._spinner.__exit__(exc_type, exc_val, exc_tb)

        if exc_val:
            buffer = io.StringIO()
            traceback.print_exception(exc_type, exc_val, exc_tb, file=buffer)

            if self._strict_validation:
                print(buffer.getvalue())
            else:
                print(
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
    @property
    def ptq_result(self) -> PtqResult:
        """Getter of self._ptq_result."""
        return self._ptq_result

    def set_ptq_result(self,
                       applied_techniques: List[str],
                       model: tf.keras.Model = None,
                       sim: QuantizationSimModel = None,
                       acc: float = None,
                       **kwargs):
        """
        Set the result of PTQ. Should be called exactly once inside a with-as block

        Exactly one among model and (sim, acc) pair should be specified
        1) If sim and acc is specified, save them as the result of this session
        2) If model is specified, evaluate the quantized accuracy of the model and save the result
        :param applied_techniques: List of applied technique names
        :param model: Result of PTQ
        :param sim: Result of PTQ. The quantization encoding (compute_encodings()) is
                    assumed to have been computed in advance
        :param acc: Eval score
        :param kwargs: Additional arguments to the quantsim factory
        """
        if sim is None:
            assert acc is None
            assert model is not None
            sim = self._quantsim_factory(model, **kwargs)
            acc = self._eval_func(sim.model)
        else:
            assert acc is not None
            assert model is None

        self._set_ptq_result(sim, acc, applied_techniques)

    def _set_ptq_result(self,
                        sim: QuantizationSimModel,
                        acc: float,
                        applied_techniques: List[str]) -> PtqResult:
        """
        Set the result of PTQ. Should be called exactly once inside a with-as block
        :param sim: Result of PTQ. The quantization encoding (compute_encodings()) is
                    assumed to have been computed in advance
        :param acc: Eval score
        :param applied_techniques: List of applied technique names
        :return: PtqResult object
        """
        if self._ptq_result is not None:
            raise RuntimeError(
                "sess.eval() can be called only once per each _EvalSession instance."
            )
        model_path, encoding_path = self._export(sim)
        self._ptq_result = PtqResult(model_path=model_path,
                                     encoding_path=encoding_path,
                                     accuracy=acc,
                                     applied_techniques=applied_techniques)
        _logger.info(self._ptq_result)
        return self._ptq_result

    def _export(self, sim: QuantizationSimModel) -> Tuple[str, str]:
        """
        Export quantsim
        :param sim: QuantizationSimModel object to export
        :return: The paths where model and encoding are saved
        """
        sim.export(path=self._results_dir, filename_prefix=self.title_lowercase)
        model_path = os.path.join(self._results_dir, f"{self.title_lowercase}")
        encoding_path = os.path.join(self._results_dir, f"{self.title_lowercase}.encodings")
        _logger.info("The results of %s is saved in %s and %s.",
                     self.title, model_path, encoding_path)
        return model_path, encoding_path

def _build_flowchart_metadata(result: Mapping) -> Dict: # pylint: disable=too-many-return-statements
    """
    Build flowchart metadata for the html template of summary report

    :param result: Result of AutoQuant with the following format:

        result := {
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
        edge_quant_scheme_selection_in='data-visited="true"',
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
