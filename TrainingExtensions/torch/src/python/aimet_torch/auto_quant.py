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
import abc
import contextlib
import copy
from dataclasses import dataclass
import functools
import os
import pickle
from typing import Any, Collection, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader
import bokeh.model
import bokeh.embed
import jinja2

from aimet_torch import utils
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.quantsim import QuantizationSimModel

from aimet_common.defs import QuantScheme
from aimet_common.utils import AimetLogger


_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AutoQuant)


@contextlib.contextmanager
def _in_eval_mode(model):
    training = model.training
    try:
        model.eval()
        yield
    finally:
        model.train(mode=training)


# The number of samples to be used for performance evaluation.
# NOTE: None means "all".
NUM_SAMPLES_FOR_PERFORMANCE_EVALUATION = None


class AutoQuant:
    """
    Integrate and apply post-training quantization techniques.

    AutoQuant includes 1) batchnorm folding, 2) cross-layer equalization,
    and 3) Adaround.
    These techniques will be applied in a best-effort manner until the model
    meets the evaluation goal given as allowed_accuracy_drop.
    """
    class Cache:
        """Cache of the outputs of PTQ functions."""

        _cached_funcs: Dict[str, str] = {}

        @classmethod
        def mark(cls, cache_key: str):
            """
            Mark functions that are subject to caching.
            The functions decorated with this mark will save/load the outputs
            to/from the cache directory if caching is enabled.

            :param cache_key: Used as a prefix of the name of the file that
                caches the results of the decorated function.
            :return: A decorator that registers the decorated functions to `_cached_funcs`.
            """
            def decorator(fn):
                if cache_key in cls._cached_funcs:
                    raise RuntimeError
                cls._cached_funcs[cache_key] = fn.__name__
                return fn
            return decorator

        def __init__(self, auto_quant: "AutoQuant", cache_dir: str = None):
            """
            :param auto_quant: AutoQuant object to apply caching.
            :param cache_dir: Cache directory. If None, caching is disabled.
            """
            self._auto_quant = auto_quant
            self._cache_dir = cache_dir
            self._original_funcs = []

            if self._cache_dir:
                os.makedirs(self._cache_dir, exist_ok=True)
                _logger.info("AutoQuant caching is enabled. Cache directory: %s", self._cache_dir)
            else:
                _logger.info("AutoQuant caching is disabled.")

        def __enter__(self):
            """Enable caching if self._cache_dir is specified."""

            def _wrap(fn: Callable, cache_key: str):
                """
                Wrap a function so that it load/save its results from/to the cache file.
                :param fn: Function to wrap
                :param cache_key: Used as a prefix of the name of the cache file.
                :return: A wrapper function which wraps `fn`.
                """
                @functools.wraps(fn)
                def caching_helper(*args, **kwargs):
                    # If caching is disabled, Evalaute the result.
                    if self._cache_dir is None:
                        return fn(*args, **kwargs)

                    cache_file = os.path.join(self._cache_dir, f"{cache_key}.pkl")
                    if os.path.exists(cache_file):
                        _logger.info("Loading result of %s from %s", cache_key, cache_file)
                        # Cached file exists (cache hit). Load from cache.
                        with open(cache_file, "rb") as f:
                            ret = pickle.load(f)
                    else:
                        # No cached file (cache miss). Evaluate the result.
                        ret = fn(*args, **kwargs)
                        _logger.info("Caching result of %s to %s", cache_key, cache_file)

                        # Save results to the cache.
                        with open(cache_file, "wb") as f:
                            pickle.dump(ret, f)
                    return ret
                return caching_helper

            for cache_key, attr_name in self._cached_funcs.items():
                original_fn = getattr(self._auto_quant, attr_name)
                self._original_funcs.append(original_fn)
                setattr(self._auto_quant, attr_name, _wrap(original_fn, cache_key))

        def __exit__(self, *_):
            while self._original_funcs:
                original_fn = self._original_funcs.pop()
                setattr(self, original_fn.__name__, original_fn)

    def __init__( # pylint: disable=too-many-arguments
            self,
            allowed_accuracy_drop: float,
            unlabeled_dataset_iterable: Union[DataLoader, Collection],
            eval_callback: Callable[[torch.nn.Module, Optional[int]], float],
            default_param_bw: int = 8,
            default_output_bw: int = 8,
            default_quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
            default_rounding_mode: str = 'nearest',
            default_config_file: str = None,
    ) -> None:
        """
        :param allowed_accuracy_drop: Maximum allowed accuracy drop.
        :param unlabeled_dataset_iterable: A collection (i.e. iterable with `__len__`)
                that iterates over an unlabeled dataset used for encoding computation.
                The values yielded by this iterable are expected to be able to be
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
        if allowed_accuracy_drop < 0:
            raise ValueError(
                "`allowed_accuracy_drop` must be a positive value. Got {:.2f}"
                .format(allowed_accuracy_drop)
            )

        QuantizationSimModel._validate_quantsim_inputs(default_quant_scheme, # pylint: disable=protected-access
                                                       default_rounding_mode,
                                                       default_output_bw,
                                                       default_param_bw)

        @functools.wraps(eval_callback)
        def eval_callback_wrapper(model: torch.nn.Module,
                                  num_samples: Optional[int]) -> float:
            """
            Wrapper to ensure that model is in eval mode before entering eval_callback.
            """
            with _in_eval_mode(model), torch.no_grad():
                return eval_callback(model, num_samples)

        self._allowed_accuracy_drop = allowed_accuracy_drop
        self._eval_callback = eval_callback_wrapper
        self._default_param_bw = default_param_bw
        self._default_output_bw = default_output_bw
        self._default_quant_scheme = default_quant_scheme
        self._default_rounding_mode = default_rounding_mode
        self._default_config_file = default_config_file

        def forward_pass_callback(model, _: Any = None):
            device = utils.get_device(model)
            with _in_eval_mode(model), torch.no_grad():
                for input_data in unlabeled_dataset_iterable:
                    input_data = utils.change_tensor_device_placement(input_data, device)
                    if isinstance(input_data, torch.Tensor):
                        model(input_data)
                    else:
                        assert isinstance(input_data, (tuple, list))
                        model(*input_data)

        self._forward_pass_callback = forward_pass_callback

        self._adaround_params = AdaroundParameters(unlabeled_dataset_iterable,
                                                   len(unlabeled_dataset_iterable))

    def _evaluate_model_performance(self, model) -> float:
        """
        Evaluate the model performance.
        """
        return self._eval_callback(model, NUM_SAMPLES_FOR_PERFORMANCE_EVALUATION)

    def set_adaround_params(self, adaround_params: AdaroundParameters) -> None:
        """
        Set Adaround parameters.
        If this method is not called explicitly by the user, AutoQuant will use
        `unlabeled_dataset_iterable` (passed to `__init__`) for Adaround.

        :param adaround_params: Adaround parameters.
        """
        self._adaround_params = adaround_params

    def _create_quantsim_and_encodings( # pylint: disable=too-many-arguments
            self,
            model: torch.nn.Module,
            dummy_input: Union[torch.Tensor, Tuple],
            quant_scheme: QuantScheme = None,
            rounding_mode: str = None,
            default_output_bw: int = None,
            default_param_bw: int = None,
            config_file: str = None,
            encoding_path: str = None,
    ) -> QuantizationSimModel:
        """
        Create a QuantizationSimModel and compute encoding. If `encoding_path` is not None,
        it is prioritized over other arguments (`default_output_bw`, `defalt_param_bw`, ...).

        :param model: Model to quantize.
        :param dummy_input: Dummy input to the model.
        :param quant_scheme: Quantization scheme. Defaults to self._default_quant_scheme.
        :param rounding_mode: Rounding mode. Defaults to self._default_rounding_mode.
        :param default_output_bw: Default bitwidth (4-31) to use for quantizing layer inputs andoutputs.
                                  Defaults to self._default_output_bw.
        :param default_param_bw: Default bitwidth (4-31) to use for quantizing layer parameters.
                                 Defaults to self._default_param_bw.
        :param config_file: Path to configuration file for model quantizers.
                            Defaults to self._default_config_file.
        :param encoding_path: Path to parameter encodings file.
        :return: Quantsim model.
        """
        kwargs = dict(
            quant_scheme=(quant_scheme or self._default_quant_scheme),
            rounding_mode=(rounding_mode or self._default_rounding_mode),
            default_output_bw=(default_output_bw or self._default_output_bw),
            default_param_bw=(default_param_bw or self._default_param_bw),
            config_file=(config_file or self._default_config_file),
        )
        sim = QuantizationSimModel(model, dummy_input, **kwargs)

        if encoding_path:
            sim.set_and_freeze_param_encodings(encoding_path)

        sim.compute_encodings(self._forward_pass_callback, None)

        return sim

    # pylint: disable=no-self-use
    @Cache.mark("batchnorm_folding")
    def _apply_batchnorm_folding(
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

    # pylint: disable=no-self-use
    @Cache.mark("cle")
    def _apply_cross_layer_equalization(
            self,
            model: torch.nn.Module,
            dummy_input: Union[torch.Tensor, Tuple],
    ) -> torch.nn.Module:
        """
        Apply cross-layer equalization.

        NOTE1: Input model is not mutated.
        NOTE2: We assume that input input model is already batchnorm-folded.

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

    @Cache.mark("adaround")
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
        model = Adaround.apply_adaround(model,
                                        dummy_input,
                                        self._adaround_params,
                                        path=results_dir,
                                        filename_prefix=filename_prefix,
                                        default_param_bw=self._default_param_bw,
                                        param_bw_override_list=None,
                                        ignore_quant_ops_list=None,
                                        default_quant_scheme=self._default_quant_scheme,
                                        default_config_file=self._default_config_file)

        return model, adaround_encoding_path

    def apply( # pylint: disable=protected-access, too-many-locals, too-many-statements
            self,
            fp32_model: torch.nn.Module,
            dummy_input_on_cpu: Union[torch.Tensor, Tuple],
            dummy_input_on_gpu: Optional[Union[torch.Tensor, Tuple]] = None,
            results_dir: str = "/tmp",
            cache_id: str = None,
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
        :return: Tuple of  (best model, eval score, encoding path front).
        :raises:
            - ValueError if the model is on GPU and dummy_input_on_gpu is not specified.
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

        with _in_eval_mode(fp32_model),\
                 AutoQuant.Cache(self, cache_dir):
            _logger.info("Starting AutoQuant")

            fp32_acc = self._evaluate_model_performance(fp32_model)
            target_acc = fp32_acc - self._allowed_accuracy_drop

            _logger.info("Target eval score: %.02f", target_acc)
            _logger.info("FP32 eval score (W32A32): %.02f", fp32_acc)

            eval_manager = _EvalManager(
                quantsim_factory=self._create_quantsim_and_encodings,
                eval_func=self._evaluate_model_performance,
                dummy_input=dummy_input,
                dummy_input_on_cpu=dummy_input_on_cpu,
                results_dir=results_dir,
            )

            ret = self._do_apply(fp32_model, target_acc, dummy_input,
                                 eval_manager, results_dir)

            _, acc, *_ = ret
            _logger.info("Best eval score: %.02f", acc)

            if acc < target_acc:
                _logger.info(
                    "AutoQuant is unable to match the target accuracy. "
                    "Consider Quantization Aware Training."
                )

            eval_manager.export_diagnostics()

            return ret

    def _do_apply( # pylint: disable=protected-access, too-many-locals, too-many-statements
            self,
            fp32_model: torch.nn.Module,
            target_acc: float,
            dummy_input: Union[torch.Tensor, Tuple],
            eval_manager: "_EvalManager",
            results_dir: str = "/tmp",
    ) -> Tuple[torch.nn.Module, float, str]:
        """
        Helper function of apply().

        :param fp32_model: Model to apply PTQ techniques.
        :param target_acc: Target eval score.
        :param dummy_input: Dummy input to the model.
            The device of dumyy_input should be same as that of model.
        :param eval_manager: _Evalmanager object.
        :param results_dir: Directory to save the results.
        :return: Tuple of  (best model, eval score, encoding path).
        """
        with eval_manager.analysis_session("Weight Quantization Sensitivity") as sess:
            acc = sess.eval(fp32_model, default_output_bw=32)
            sess.diagnostics.add(
                f"Weight-quantized eval score (W{self._default_param_bw}A32): {acc:.02f}"
            )

        with eval_manager.analysis_session("Activation Quantization Sensitivity") as sess:
            acc = sess.eval(fp32_model, default_param_bw=32)
            sess.diagnostics.add(
                f"Activation-quantized eval score (W32A{self._default_output_bw}): {acc:.02f}"
            )

        # Batchnorm Folding
        with eval_manager.ptq_session("Batchnorm Folding") as sess:
            model, folded_pairs = self._apply_batchnorm_folding(fp32_model, dummy_input)
            for conv, bn in folded_pairs:
                sess.diagnostics.add(f"{conv} was merged with {bn}.")
            sess.set_ptq_result(model)

        _, model, encoding_path, acc = eval_manager.get_best_ptq_result()
        if acc >= target_acc:
            return model, acc, encoding_path

        # Cross-Layer Equalization
        with eval_manager.ptq_session("Cross-Layer Equalization") as sess:
            model = self._apply_cross_layer_equalization(fp32_model, dummy_input)
            sess.set_ptq_result(model)

        _, model, encoding_path, acc = eval_manager.get_best_ptq_result()
        if acc >= target_acc:
            return model, acc, encoding_path

        # AdaRound
        with eval_manager.ptq_session("AdaRound") as sess:
            model, encoding_path = self._apply_adaround(model, dummy_input, results_dir)
            sess.set_ptq_result(model, encoding_path=encoding_path)

        _, model, encoding_path, acc = eval_manager.get_best_ptq_result()

        return model, acc, encoding_path


class _EvalManager:
    """
    Evaluation manager for AutoQuant.
    """
    def __init__(self,
                 quantsim_factory: Callable,
                 eval_func: Callable[[torch.nn.Module], float],
                 dummy_input: Union[torch.Tensor, Tuple],
                 dummy_input_on_cpu: Union[torch.Tensor, Tuple],
                 results_dir: str):
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

        os.makedirs(self._results_dir, exist_ok=True)

        self._all_sessions: List[_EvalSession] = []
        self._ptq_sessions: List[_PtqSession] = []

    def get_best_ptq_result(self) -> Tuple[str, torch.nn.Module, str, float]:
        """
        Get the results with the highest evaluation score among the ptq results evaluated so far.
        :return: The best evaluation result so far, including the tag, model object,
                 encodings path, and accuracy.
        """
        if not self._ptq_sessions:
            raise RuntimeError

        ptq_results = [sess.ptq_result for sess in self._ptq_sessions]
        best_ptq_result = max(ptq_results, key=lambda ptq_result: ptq_result.accuracy)
        return best_ptq_result.tag,\
               best_ptq_result.load_model(),\
               best_ptq_result.encoding_path,\
               best_ptq_result.accuracy

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
                              results_dir=os.path.join(self._results_dir, ".trace"))
        self._all_sessions.append(session)
        return session

    def export_diagnostics(self) -> str:
        """
        Export diagnostics in html format.
        :return: Diagnostics string in html format.
        """
        loader = jinja2.FileSystemLoader(os.path.dirname(os.path.abspath(__file__)))
        env = jinja2.Environment(loader=loader)
        template = env.get_template("auto_quant_diagnostics_template.html")

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
        filename = os.path.join(self._results_dir, "diagnostics.html")
        with open(filename, "w") as f:
            f.write(html)
        return html


class _EvalSession:
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
            results_dir: str
    ):
        """
        :param title: Title of the session.
        :param quantsim_factory: A factory function that returns QuantizationSimModel.
        :param eval_func: Evaluation function.
        :param dummy_input: Dummy input to the model. Assumed to be located on the same device as the model.
        :param dummy_input_on_cpu: Dummy input to the model in CPU memory.
        :param results_dir: Base directory to save the temporary serialized model.
        """
        self._title = title
        self._quantsim_factory = quantsim_factory
        self._eval_func = eval_func
        self._dummy_input = dummy_input
        self._dummy_input_on_cpu = dummy_input_on_cpu
        self._results_dir = results_dir

        os.makedirs(self._results_dir, exist_ok=True)

        self._diagnostics = _Diagnostics()

        # Map session title to file name.
        # e.g. title: "Cross-Layer Equalization" -> filename: "cross_layer_equalization"
        self._filename = self._title.lower().replace("-", " ")
        self._filename = "_".join(self._filename.split())

    @property
    def title(self):
        """Getter of self._title."""
        return self._title

    @property
    def diagnostics(self):
        """Getter of self._diagnostics."""
        return self._diagnostics

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
        _logger.info("Session start: %s", self._title)
        return self

    def __exit__(self, *_):
        _logger.info("Session finished: %s.", self._title)


class _PtqSession(_EvalSession):
    """
    PTQ session.

    Each PTQ session object should call `set_ptq_result` exactly once
    inside a with-as block.
    """

    @dataclass
    class PtqResult:
        """
        Evaluation results.
        :param tag: Identifier string of the evaluation result.
        :param model_path: Path to the serialized model.
        :param encoding_path: Path to the encoding file.
        :param accuracy: Accuracy of the model.
        """
        tag: str
        model_path: str
        device: torch.device
        encoding_path: str
        accuracy: float

        def load_model(self):
            """
            Load model.
            :return: Loaded model.
            """
            return torch.load(self.model_path).to(self.device)

    def __init__(self, *args, **kwargs):
        super(_PtqSession, self).__init__(*args, **kwargs)
        self._ptq_result = None

    @property
    def ptq_result(self) -> "_PtqSession.PtqResult":
        """Getter of self._ptq_result."""
        if self._ptq_result is None:
            raise RuntimeError
        return self._ptq_result

    def set_ptq_result(
            self,
            model: torch.nn.Module = None,
            sim: QuantizationSimModel = None,
            acc: float = None,
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
        if sim is None:
            assert acc is None
            assert model is not None
            sim = self._quantsim_factory(model, self._dummy_input, **kwargs)
            acc = self._eval_func(sim.model)
        else:
            assert acc is not None
            assert model is None

        self._set_ptq_result(sim, acc)

    def _set_ptq_result(self, sim: QuantizationSimModel, acc: float) -> "_PtqSession.PtqResult":
        """
        Set the result of PTQ. Should be called exactly once inside a with-as block.

        :param sim: Result of PTQ. The quamtization encoding (compute_encodings()) is
                    assumed to have been computed in advance.
        :param acc: Eval score.
        :return: PtqResult object.
        """
        if self._ptq_result is not None:
            raise RuntimeError(
                "sess.eval() can be called only once per each _EvalSession instance."
            )

        device = utils.get_device(sim.model)
        model_path, encoding_path = self._export(sim)
        self._ptq_result = _PtqSession.PtqResult(
            tag=self._filename,
            model_path=model_path,
            device=device,
            encoding_path=encoding_path,
            accuracy=acc
        )
        return self._ptq_result

    def _export(self, sim: QuantizationSimModel) -> Tuple[str, str]:
        """
        Export quantsim.
        :param sim: QuantizationSimModel object to export.
        :return: The paths where model and encoding are saved
        """
        sim.export(path=self._results_dir,
                   filename_prefix=self._filename,
                   dummy_input=self._dummy_input_on_cpu)
        model_path = os.path.join(self._results_dir, f"{self._filename}.pth")
        encoding_path = os.path.join(self._results_dir, f"{self._filename}.encodings")
        _logger.info("The results of %s is saved in %s and %s.",
                     self._title, model_path, encoding_path)
        return model_path, encoding_path


    def __exit__(self, *_):
        """Raises error if set_ptq_result is not called."""
        if self._ptq_result is None:
            raise RuntimeError

        _logger.info("Session finished: %s. (eval score: %.02f)",
                     self._title, self._ptq_result.accuracy)


class _Diagnostics:
    """Diagnostics produced by _EvalSession"""

    def __init__(self):
        self._contents: List[_DiagnosticsContent] = []

    def add(self, content: Union[str, bokeh.model.Model]) -> None:
        """
        Add content of diagnostics.
        :param content: Content of diagnostics.
        :return: None
        """
        if isinstance(content, str):
            c = _PlainTextContent(content)
        elif isinstance(content, bokeh.model.Model):
            c = _BokehModelContent(content)
        else:
            raise RuntimeError
        self._contents.append(c)

    def is_empty(self) -> bool:
        """Return True if and only if the diagnostics is empty."""
        return not bool(self._contents)

    def contains_bokeh(self):
        """Return True if and only if the diagnostics contains a bokeh model object."""
        return any(
            isinstance(content, _BokehModelContent)
            for content in self._contents
        )

    def __bool__(self):
        return not self.is_empty()

    def __iter__(self):
        return iter(self._contents)


class _DiagnosticsContent(abc.ABC):
    """Content of diagnostics."""
    def __init__(self, content: Any):
        self._content = content

    @abc.abstractmethod
    def get_html_elem(self) -> str:
        """
        Render content as an html element.
        :return: Content rendered as an html element.
        """


class _PlainTextContent(_DiagnosticsContent):
    """Content of diagnostics in plain text."""
    def get_html_elem(self) -> str:
        return f"<div> {self._content} </div>"


class _BokehModelContent(_DiagnosticsContent):
    """Content of diagnostics as a bokeh model object."""
    def get_html_elem(self) -> str:
        bokeh_model = self._content
        script, div = bokeh.embed.components(bokeh_model)
        return f"{script}{div}"
