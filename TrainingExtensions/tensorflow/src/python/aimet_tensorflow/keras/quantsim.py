# /usr/bin/env python3.5
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
""" Quantsim for Keras """

from typing import Union
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import model_transformer

from aimet_common.defs import QuantScheme
from aimet_tensorflow.keras.quant_sim.qc_quantize_wrapper import QcQuantizeWrapper, QuantizeWrapperTransform, \
    QuantizerSettings

unquantizable_modules = ('InputLayer', 'QcQuantizeWrapper')

class QuantizationSimModel:
    """
    Implements mechanism to add quantization simulations ops to a model. This allows for off-target simulation of
    inference accuracy. Also allows the model to be fine-tuned to counter the effects of quantization.
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=unused-argument
    def __init__(self, model, quant_scheme: Union[QuantScheme, str] = 'tf_enhanced', rounding_mode: str = 'nearest',
                 default_output_bw: int = 8, default_param_bw: int = 8, config_file: str = None):
        self.model = self._add_quantization_wrappers(model, quant_scheme, rounding_mode, default_output_bw,
                                                     default_param_bw)

    @staticmethod
    def _add_quantization_wrappers(model, quant_scheme, rounding_mode, default_output_bw, default_param_bw):
        """
        Add quantization wrappers to the model and return a new model with the wrappers inserted.
        :param model: Model to add quantization wrappers for
        :param quant_scheme: Quantization scheme to use
        :param rounding_mode: Rounding mode to use
        :param default_output_bw: Default bitwidth for activation quantizers
        :param default_param_bw: Default bitwidth for param quantizers
        """
        layer_type_set = set()
        name_to_layer_map = {}
        for layer in model.layers:
            layer_type = layer.__class__.__name__
            if layer_type not in unquantizable_modules:
                layer_type_set.add(layer.__class__.__name__)
            name_to_layer_map[layer.name] = layer
        transforms = []
        for layer_type in layer_type_set:
            transforms.append(
                QuantizeWrapperTransform(layer_type,
                                         activation_quant_settings=QuantizerSettings(default_output_bw, rounding_mode,
                                                                                     quant_scheme, False, False, False),
                                         param_quant_settings=QuantizerSettings(default_param_bw, rounding_mode,
                                                                                quant_scheme, False, False, False),
                                         name_to_module_map=name_to_layer_map))
        model, _ = model_transformer.ModelTransformer(model, transforms).transform()
        return model

    def compute_encodings(self, forward_pass_callback, forward_pass_callback_args):
        """
        Computes encodings for all quantization sim nodes in the model.
        :param forward_pass_callback: A callback function that is expected to runs forward passes on a model.
               This callback function should use representative data for the forward pass, so the calculated
               encodings work for all data samples.
        :param forward_pass_callback_args: These argument(s) are passed to the forward_pass_callback as-is. Up to
               the user to determine the type of this parameter. E.g. could be simply an integer representing the number
               of data samples to use. Or could be a tuple of parameters or an object representing something more
               complex.
        """
        forward_pass_callback(self.model, forward_pass_callback_args)
        for quant_wrapper in self.quant_wrappers():
            quant_wrapper.compute_encoding()

    def quant_wrappers(self):
        """
        Generator for yielding all quantization wrappers
        """
        for layer in self.model.layers:
            if isinstance(layer, QcQuantizeWrapper):
                yield layer
