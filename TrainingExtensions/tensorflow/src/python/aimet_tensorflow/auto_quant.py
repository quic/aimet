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
import contextlib
from dataclasses import dataclass
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import tensorflow as tf
from tqdm import tqdm

import jinja2

from aimet_tensorflow.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_tensorflow.cross_layer_equalization import equalize_model
from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.utils.graph_saver import load_model_from_meta
from aimet_tensorflow.utils.common import (
    create_input_feed_dict,
    deepcopy_tf_session,
    iterate_tf_dataset,
)
from aimet_tensorflow.cache import TfSessionSerializationProtocol

from aimet_common.auto_quant import Diagnostics
from aimet_common.cache import Cache
from aimet_common.defs import QuantScheme
from aimet_common.utils import AimetLogger, Spinner
from aimet_common.quantsim import validate_quantsim_inputs


tf.compat.v1.disable_eager_execution()

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AutoQuant)


cache = Cache()


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

    def __init__( # pylint: disable=too-many-arguments
            self,
            allowed_accuracy_drop: float,
            unlabeled_dataset: tf.compat.v1.data.Dataset,
            eval_callback: Callable[[tf.compat.v1.Session, Optional[int]], float],
            default_param_bw: int = 8,
            default_output_bw: int = 8,
            default_quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
            default_rounding_mode: str = 'nearest',
            default_config_file: str = None,
    ) -> None:
        """
        :param allowed_accuracy_drop: Maximum allowed accuracy drop.
        :param unlabeled_dataset: An unlabeled dataset for encoding computation.
                By default, this dataset will be also used for Adaround unless
                otherwise specified by `self.set_adaround_params`.
        :param eval_callback: A function that maps a tf session and the number of samples
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

        validate_quantsim_inputs(default_quant_scheme,
                                 default_rounding_mode,
                                 default_output_bw,
                                 default_param_bw)

        self.allowed_accuracy_drop = allowed_accuracy_drop
        self.eval_callback = eval_callback
        self.default_param_bw = default_param_bw
        self.default_output_bw = default_output_bw
        self.default_quant_scheme = default_quant_scheme
        self.default_rounding_mode = default_rounding_mode
        self.default_config_file = default_config_file

        self._unlabeled_dataset = unlabeled_dataset
        self._unlabled_dataset_length = None

        self._adaround_params = None

    @property
    def adaround_params(self):
        """Returns the adaround parameter."""
        # If adaround_params is manually set, return it.
        if self._adaround_params is not None:
            return self._adaround_params
        # Otherwise, return the default adaround params if the length of the
        # dataset if known.
        if self._unlabled_dataset_length is not None:
            return AdaroundParameters(self._unlabeled_dataset,
                                      self._unlabled_dataset_length)
        return None

    def _evaluate_model_performance(self, sess: tf.compat.v1.Session) -> float:
        """
        Evaluate the model performance.

        :param sess: tf.Session associated with the model to evaluate.
        :return: Evaluation score.
        """
        return self.eval_callback(sess, NUM_SAMPLES_FOR_PERFORMANCE_EVALUATION)

    def set_adaround_params(self, adaround_params: AdaroundParameters) -> None:
        """
        Set Adaround parameters.
        If this method is not called explicitly by the user, AutoQuant will use
        `unlabeled_dataset` (passed to `__init__`) for Adaround.

        :param adaround_params: Adaround parameters.
        """
        self._adaround_params = adaround_params

    def _create_quantsim_and_encodings( # pylint: disable=too-many-arguments
            self,
            sess: tf.compat.v1.Session,
            starting_op_names: List[str],
            output_op_names: List[str],
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

        NOTE: Input session is not mutated.

        :param sess: The input model as session to add quantize ops to.
        :param starting_op_names: List of starting op names of the model.
        :param output_op_names: List of output op names of the model.
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
        kwargs = dict(
            quant_scheme=(quant_scheme or self.default_quant_scheme),
            rounding_mode=(rounding_mode or self.default_rounding_mode),
            default_output_bw=(default_output_bw or self.default_output_bw),
            default_param_bw=(default_param_bw or self.default_param_bw),
            config_file=(config_file or self.default_config_file),
        )
        with deepcopy_tf_session(sess) as sess: # pylint: disable=redefined-argument-from-local
            sim = QuantizationSimModel(sess, starting_op_names, output_op_names, **kwargs)

        if encoding_path:
            sim.set_and_freeze_param_encodings(encoding_path)

        def forward_pass_callback(sess: tf.compat.v1.Session, _: Any = None):
            output_ops = [
                sess.graph.get_operation_by_name(op_name)
                for op_name in output_op_names
            ]

            count = 0
            iterator = iterate_tf_dataset(self._unlabeled_dataset)
            for inputs in tqdm(iterator, total=self._unlabled_dataset_length):
                feed_dict = create_input_feed_dict(sess.graph, starting_op_names, inputs)
                sess.run(output_ops, feed_dict=feed_dict)
                count += 1

            self._unlabled_dataset_length = count

        sim.compute_encodings(forward_pass_callback, None)

        return sim

    def _apply_batchnorm_folding( # pylint: disable=no-self-use
            self,
            sess: tf.compat.v1.Session,
            starting_op_names: List[str],
            output_op_names: List[str],
    ) -> Tuple[tf.compat.v1.Session, List[Tuple[tf.Operation, tf.Operation]]]:
        """
        Apply batchnorm folding.

        NOTE: Input session is not mutated.

        :param sess: tf.Session associated with the model to apply cle.
        :param starting_op_names: List of starting op names of the model.
        :param output_op_names: List of output op names of the model.
        :return: Output session and folded pairs.
        """
        # NOTE: We don't apply caching to batchnorm folding because caching is
        #       likely going to have an adverse effect on the performance.
        #       Since a tf.Operation contains a reference to the graph it belongs
        #       to, serializing a subset of operations of a tf.Graph requires
        #       serializing the whole graph, making the serialization cost very
        #       likely to exceed the evaluation cost.
        with deepcopy_tf_session(sess) as sess: # pylint: disable=redefined-argument-from-local
            return fold_all_batch_norms(sess, starting_op_names, output_op_names)

    @cache.mark("cle", TfSessionSerializationProtocol())
    def _apply_cross_layer_equalization( # pylint: disable=no-self-use
            self,
            sess: tf.compat.v1.Session,
            starting_op_names: List[str],
            output_op_names: List[str],
    ) -> tf.compat.v1.Session:
        """
        Apply cross-layer equalization.

        NOTE: Input session is not mutated.

        :param sess: tf.Session associated with the model to apply batchnorm folding.
        :param starting_op_names: List of starting op names of the model.
        :param output_op_names: List of output op names of the model.
        :return: Output session.
        """
        with deepcopy_tf_session(sess) as sess: # pylint: disable=redefined-argument-from-local
            return equalize_model(sess, starting_op_names, output_op_names)

    def _apply_adaround(
            self,
            sess: tf.compat.v1.Session,
            starting_op_names: List[str],
            output_op_names: List[str],
            results_dir: str,
    ) -> Tuple[tf.compat.v1.Session, str]:
        """
        Apply adaround.

        :param sess: tf.Session associated with the model to apply adaround.
        :param starting_op_names: List of starting op names of the model.
        :param output_op_names: List of output op names of the model.
        :param results_dir: Directory to save the results of AdaRound.
        :return: Output session and the path to the parameter encoding file.
        """
        # NOTE: We dont need to make a deepcopy of model here, since Adaround.apply_adaround
        # internally creates and returns a deepcopy of model.
        if self.adaround_params is None:
            raise RuntimeError

        filename_prefix = "adaround"
        adaround_encoding_path = os.path.join(results_dir,
                                              "{}.encodings".format(filename_prefix))
        _apply_adaround_cached =\
            cache.mark("adaround", TfSessionSerializationProtocol())\
            (Adaround.apply_adaround)

        sess = _apply_adaround_cached(sess,
                                      starting_op_names,
                                      output_op_names,
                                      self.adaround_params,
                                      path=results_dir,
                                      filename_prefix=filename_prefix,
                                      default_param_bw=self.default_param_bw,
                                      default_quant_scheme=self.default_quant_scheme,
                                      default_config_file=self.default_config_file)

        return sess, adaround_encoding_path

    def apply(
            self,
            fp32_sess: tf.compat.v1.Session,
            starting_op_names: List[str],
            output_op_names: List[str],
            results_dir: str = "/tmp",
            cache_id: str = None,
    ) -> Tuple[tf.compat.v1.Session, float, str]:
        """
        Apply post-training quantization techniques.

        :param fp32_sess: tf.Session associated with the model to apply PTQ techniques.
        :param starting_op_names: List of starting op names of the model.
        :param output_op_names: List of output op names of the model.
        :param results_dir: Directory to save the results.
        :return: Tuple of (best session, eval score, encoding path).
        """
        result = self._apply_helper(self._auto_quant_main,
                                    fp32_sess,
                                    starting_op_names,
                                    output_op_names,
                                    results_dir,
                                    cache_id)
        return result["model"],\
               result["accuracy"],\
               result["encoding_path"]

    def _apply_helper(
            self,
            auto_quant_main_fn: Callable,
            fp32_sess: tf.compat.v1.Session,
            starting_op_names: List[str],
            output_op_names: List[str],
            results_dir: str = "/tmp",
            cache_id: str = None,
    ) -> Dict[str, Any]:
        """
        Helper for self.apply().

        :param fp32_sess: tf.Session associated with the model to apply PTQ techniques.
        :param starting_op_names: List of starting op names of the model.
        :param output_op_names: List of output op names of the model.
        :param results_dir: Directory to save the results.
        :return: The best ptq result as a dictionary.
        """
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        if cache_id is None:
            cache_dir = None
        else:
            cache_dir = os.path.join(results_dir, ".auto_quant_cache", cache_id)

        with cache.enable(cache_dir):
            _logger.info("Starting AutoQuant")

            fp32_acc = self._evaluate_model_performance(fp32_sess)
            target_acc = fp32_acc - self.allowed_accuracy_drop

            _logger.info("Target eval score: %f", target_acc)
            _logger.info("FP32 eval score (W32A32): %f", fp32_acc)

            eval_manager = _EvalManager(
                quantsim_factory=self._create_quantsim_and_encodings,
                eval_func=self._evaluate_model_performance,
                starting_op_names=starting_op_names,
                output_op_names=output_op_names,
                results_dir=results_dir,
            )

            ret = auto_quant_main_fn(fp32_sess, target_acc,
                                     starting_op_names, output_op_names,
                                     eval_manager, results_dir)

            acc = ret["accuracy"]
            _logger.info("Best eval score: %f", acc)

            if acc < target_acc:
                _logger.info(
                    "AutoQuant is unable to match the target accuracy. "
                    "Consider Quantization Aware Training."
                )

            eval_manager.export_diagnostics()

            return ret

    def _auto_quant_main(
            self,
            fp32_sess: tf.compat.v1.Session,
            target_acc: float,
            starting_op_names: List[str],
            output_op_names: List[str],
            eval_manager: "_EvalManager",
            results_dir: str = "/tmp",
    ) -> Dict[str, Any]:
        """
        Helper function of apply().

        :param fp32_sess: Model to apply PTQ techniques.
        :param target_acc: Target eval score.
        :param starting_op_names: List of starting op names of the model.
        :param output_op_names: List of output op names of the model.
        :param eval_manager: _Evalmanager object.
        :param results_dir: Directory to save the results.
        :return: The best ptq result as a dictionary.
        """
        with eval_manager.analysis_session("Weight Quantization Sensitivity") as s:
            acc = s.eval(fp32_sess, default_output_bw=32)
            s.diagnostics.add(
                f"Weight-quantized eval score (W{self.default_param_bw}A32): {acc:f}"
            )

        with eval_manager.analysis_session("Activation Quantization Sensitivity") as s:
            acc = s.eval(fp32_sess, default_param_bw=32)
            s.diagnostics.add(
                f"Activation-quantized eval score (W32A{self.default_output_bw}): {acc:f}"
            )

        # Batchnorm Folding
        with eval_manager.ptq_session("Batchnorm Folding") as s:
            sess, folded_pairs = self._apply_batchnorm_folding(fp32_sess,
                                                               starting_op_names,
                                                               output_op_names)
            for conv, bn in folded_pairs:
                s.diagnostics.add(f"{conv} was merged with {bn}.")
            s.set_ptq_result(sess=sess, applied_techniques=["batchnorm_folding"])

        best_result = eval_manager.get_best_ptq_result()
        if best_result.accuracy >= target_acc:
            return best_result.as_dict()

        # Cross-Layer Equalization
        with eval_manager.ptq_session("Cross-Layer Equalization") as s:
            sess = self._apply_cross_layer_equalization(fp32_sess,
                                                        starting_op_names,
                                                        output_op_names)
            s.set_ptq_result(sess=sess, applied_techniques=["cross_layer_equalization"])

        best_result = eval_manager.get_best_ptq_result()
        if best_result.accuracy >= target_acc:
            return best_result.as_dict()

        # AdaRound
        with eval_manager.ptq_session("AdaRound") as s:
            sess, encoding_path = self._apply_adaround(best_result.load_model(),
                                                       starting_op_names,
                                                       output_op_names,
                                                       results_dir)
            s.set_ptq_result(sess=sess,
                             encoding_path=encoding_path,
                             applied_techniques=[*best_result.applied_techniques, "adaround"])

        return eval_manager.get_best_ptq_result().as_dict()


@dataclass
class PtqResult:
    """
    Evaluation results.
    :param tag: Identifier string of the evaluation result.
    :param model_path: Path to the serialized model.
    :param encoding_path: Path to the encoding file.
    :param accuracy: Accuracy of the model.
    """
    meta_path: str
    checkpoint_path: str
    encoding_path: str
    accuracy: float
    applied_techniques: List[str]

    def load_model(self) -> tf.compat.v1.Session:
        """
        Load model.
        :return: Loaded model.
        """
        return load_model_from_meta(self.meta_path, self.checkpoint_path)

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
                 eval_func: Callable[[tf.compat.v1.Session], float],
                 starting_op_names: List[str],
                 output_op_names: List[str],
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
        self._starting_op_names = starting_op_names
        self._output_op_names = output_op_names
        self._results_dir = results_dir

        os.makedirs(self._results_dir, exist_ok=True)

        self._all_sessions: List[_EvalSession] = []
        self._ptq_sessions: List[_PtqSession] = []

    def get_best_ptq_result(self) -> PtqResult:
        """
        Get the results with the highest evaluation score among the ptq results evaluated so far.
        :return: The best evaluation result so far.
        """
        if not self._ptq_sessions:
            raise RuntimeError

        ptq_results = [sess.ptq_result for sess in self._ptq_sessions]
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
                              self._starting_op_names,
                              self._output_op_names,
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
            eval_func: Callable[[tf.compat.v1.Session], float],
            starting_op_names: List[str],
            output_op_names: List[str],
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
        self._starting_op_names = starting_op_names
        self._output_op_names = output_op_names
        self._results_dir = results_dir
        self._spinner = None

        os.makedirs(self._results_dir, exist_ok=True)

        self._diagnostics = Diagnostics()

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

    def eval(self, sess: tf.compat.v1.Session, **kwargs):
        """
        Evaluate the model.
        :param sess: tf.Session associated with the model to evaluate.
        :param **kwargs: Additional arguments to the quantsim factory.
        :return: Eval score.
        """
        sim = self._quantsim_factory(sess,
                                     self._starting_op_names,
                                     self._output_op_names,
                                     **kwargs)
        acc = self._eval_func(sim.session)
        return acc

    def __enter__(self):
        self._spinner = Spinner(self._title)
        self._spinner.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._spinner is not None:
                self._spinner.__exit__(exc_type, exc_val, exc_tb)
        finally:
            if exc_val is not None:
                raise exc_val


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
    def ptq_result(self) -> PtqResult:
        """Getter of self._ptq_result."""
        if self._ptq_result is None:
            raise RuntimeError
        return self._ptq_result

    def set_ptq_result(
            self,
            applied_techniques: List[str],
            sess: tf.compat.v1.Session = None,
            sim: QuantizationSimModel = None,
            acc: float = None,
            **kwargs
    ) -> None:
        """
        Set the result of PTQ. Should be called exactly once inside a with-as block.

        Exactly one among model and (sim, acc) pair should be specified.
        1) If sim and acc is specified, save them as the result of this session.
        2) If model is specified, evaluate the quantized accuracy of the model and save the result.

        :param sess: Result of PTQ.
        :param sim: Result of PTQ. The quamtization encoding (compute_encodings()) is
                    assumed to have been computed in advance.
        :param acc: Eval score.
        :param **kwargs: Additional arguments to the quantsim factory.
        :return: None
        """
        if sim is None:
            assert acc is None
            assert sess is not None
            sim = self._quantsim_factory(sess,
                                         self._starting_op_names,
                                         self._output_op_names,
                                         **kwargs)
            acc = self._eval_func(sim.session)
        else:
            assert acc is not None
            assert sess is None

        self._set_ptq_result(sim, acc, applied_techniques)

    def _set_ptq_result(
            self,
            sim: QuantizationSimModel,
            acc: float,
            applied_techniques: List[str],
    ) -> PtqResult:
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

        meta_path, checkpoint_path, encoding_path = self._export(sim)
        self._ptq_result = PtqResult(
            meta_path=meta_path,
            checkpoint_path=checkpoint_path,
            encoding_path=encoding_path,
            accuracy=acc,
            applied_techniques=applied_techniques,
        )
        return self._ptq_result

    def _export(self, sim: QuantizationSimModel) -> Tuple[str, str, str]:
        """
        Export quantsim.
        :param sim: QuantizationSimModel object to export.
        :return: The paths where model and encoding are saved
        """
        sim.export(path=self._results_dir, filename_prefix=self._filename)
        checkpoint_path = os.path.join(self._results_dir, self._filename)
        meta_path = f"{checkpoint_path}.meta"
        encoding_path = f"{checkpoint_path}.encodings"
        _logger.info("The results of %s is saved in %s, %s, and %s.",
                     self._title, checkpoint_path, meta_path, encoding_path)
        return meta_path, checkpoint_path, encoding_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Raises error if set_ptq_result is not called."""
        super(_PtqSession, self).__exit__(exc_type, exc_val, exc_tb)

        if self._ptq_result is None:
            raise RuntimeError

        _logger.info("Session finished: %s. (eval score: %f)",
                     self._title, self._ptq_result.accuracy)


@contextlib.contextmanager
def spy_auto_quant(auto_quant: AutoQuant):
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

    def _auto_quant_main_wrapper(fp32_sess, target_acc, starting_op_names,
                                 output_op_names, eval_manager, results_dir="/tmp"):
        spy._eval_manager = eval_manager
        return _auto_quant_main(fp32_sess, target_acc, starting_op_names,
                                output_op_names, eval_manager, results_dir)

    try:
        setattr(auto_quant, "_auto_quant_main", _auto_quant_main_wrapper)
        yield spy
    finally:
        setattr(auto_quant, "_auto_quant_main", _auto_quant_main)
