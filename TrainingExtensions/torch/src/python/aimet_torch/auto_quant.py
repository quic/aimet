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

""" Concrete implementation of AutoQuant for v1 quantsim """
import itertools

from aimet_torch._auto_quant import AutoQuantBase, PtqResult, spy_auto_quant, _logger, _EvalSession, _EvalManager, \
    cache, _QuantSchemePair # pylint: disable=unused-import

from aimet_torch import utils
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.utils import get_all_quantizers
from aimet_common.defs import QuantScheme


# The number of samples to be used for performance evaluation.
# NOTE: None means "all".
NUM_SAMPLES_FOR_PERFORMANCE_EVALUATION = None


class AutoQuant(AutoQuantBase): # pylint: disable=too-many-instance-attributes
    """
    Integrate and apply post-training quantization techniques.

    AutoQuant includes 1) batchnorm folding, 2) cross-layer equalization,
    and 3) Adaround.
    These techniques will be applied in a best-effort manner until the model
    meets the evaluation goal given as allowed_accuracy_drop.
    """

    Adaround = Adaround
    AdaroundParameters = AdaroundParameters

    @staticmethod
    def _get_adaround_parameters(data_loader, num_batches):
        return AdaroundParameters(data_loader, num_batches)

    def _evaluate_model_performance(self, model) -> float:
        """
        Evaluate the model performance.
        """
        return self.eval_callback(model, NUM_SAMPLES_FOR_PERFORMANCE_EVALUATION)

    @staticmethod
    def _get_quantsim(model, dummy_input, **kwargs):
        return QuantizationSimModel(model, dummy_input, **kwargs)

    def _configure_quantsim(self, # pylint: disable=too-many-arguments
                            sim,
                            output_bw,
                            output_quant_scheme,
                            output_percentile,
                            param_bw,
                            param_quant_scheme,
                            param_percentile,
                            encoding_path):

        param_quantizers, input_quantizers, output_quantizers = utils.get_all_quantizers(sim.model)

        # Set input/output quantizers' quant schemes
        for quantizer in itertools.chain(input_quantizers, output_quantizers):
            quantizer.quant_scheme = output_quant_scheme
            if quantizer.quant_scheme == QuantScheme.post_training_percentile and\
                    output_percentile is not None:
                quantizer.set_percentile_value(output_percentile)

        # Set param quantizers' quant schemes
        for quantizer in param_quantizers:
            quantizer.quant_scheme = param_quant_scheme
            if quantizer.quant_scheme == QuantScheme.post_training_percentile and\
                    param_percentile is not None:
                quantizer.set_percentile_value(param_percentile)

        if encoding_path:
            sim.set_and_freeze_param_encodings(encoding_path)

        param_quantizers, input_quantizers, output_quantizers = utils.get_all_quantizers(sim.model)

        # Disable input/output quantizers, using fp32 to simulate int32.
        if output_bw == 32:
            for quantizer in input_quantizers + output_quantizers:
                quantizer.enabled = False

        # Disable param quantizers, using fp32 to simulate int32.
        if param_bw == 32:
            for quantizer in param_quantizers:
                quantizer.enabled = False

    @staticmethod
    def _has_enabled_quantizers(sim):
        param_quantizers, input_quantizers, output_quantizers = utils.get_all_quantizers(sim.model)
        return any(quantizer.enabled for quantizer in param_quantizers +\
                                                      input_quantizers +\
                                                      output_quantizers)

    @staticmethod
    def _disable_activation_quantizers(sim):
        _, input_quantizers, output_quantizers = get_all_quantizers(sim.model)
        for quantizer in itertools.chain(input_quantizers, output_quantizers):
            quantizer.enabled = False
