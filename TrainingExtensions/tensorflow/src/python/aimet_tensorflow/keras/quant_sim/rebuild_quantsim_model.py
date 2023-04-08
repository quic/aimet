# /usr/bin/env python3.8
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
""" Rebuilt QuantSim Model For Keras """
from typing import Dict, List, Union
from collections import OrderedDict
import tensorflow as tf

from aimet_common.utils import AimetLogger

from aimet_tensorflow.keras.quant_sim.tensor_quantizer import TensorQuantizer
from aimet_tensorflow.keras.quant_sim.qc_quantize_wrapper import QcQuantizeWrapper

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

_SPECIAL_PARSE_CHAR = "@"
_VARIATIONS_OF_TENSOR_QUANTIZERS_TYPES = Union[TensorQuantizer, List[TensorQuantizer]]

class RebuiltQuantSimModelFactory:
    """
    Rebuilt QuantSim Model For Keras. Specifically used to rebuild a QuantSim model after it has been converted to a frozen pb.
    This use case occurs during exporting of the model.
    """

    def __init__(self, original_quantsim: tf.keras.Model): # UPDATE
        """
        :param original_quantsim_model: The original QuantizationSimModel
        """
        self.original_quantsim_model         = original_quantsim.model
        self.original_quantsim_params        = original_quantsim._params
        self.original_model_without_wrappers = original_quantsim._model_without_wrappers
        self.original_quantsim_model_weights = original_quantsim.model.get_weights()

        self.original_quantsim_model_weight_names_to_weights = \
            RebuiltQuantSimModelFactory._get_model_weight_names_to_weight_values_dict(original_quantsim.model)
        self.rebuilt_model = None

    def _assert_rebuilt_model_is_not_none(self):
        """
        :return: True if the rebuilt model is None
        """
        assert self.rebuilt_model is not None, "rebuilt_model is None. Please call rebuild_model first."

    @staticmethod
    def _get_model_weight_names_to_weight_values_dict(model: tf.keras.Model) -> Dict[str, tf.Variable]:
        """
        There could be some weights that are added on later on that are not attached to the model itself. For example,
        like the auto-quant tests. This function is used to get all the weights and their weight names. Unfortunately,
        unattached weights names do not have to be unique. For example, there could be multiple 'Variable:0's in the weights
        This function adds a special parsing character so that sets can be used to speed up computation for identifying missing
        weights when rebuilding.

        :param model: The model to get the weights from
        :return: A dictionary of weight names to weights
        """
        dict_with_weights_to_return = OrderedDict()
        found_duplicates = OrderedDict()
        for w in model.weights:
            if w.name in dict_with_weights_to_return:
                dict_with_weights_to_return[
                    f"{w.name}{_SPECIAL_PARSE_CHAR}{found_duplicates.get(w.name, 0) + 1}"] = w
                found_duplicates[w.name] = found_duplicates.get(w.name, 0) + 1
            else:
                dict_with_weights_to_return[w.name] = w
        return dict_with_weights_to_return

    def _copy_original_models_quantizer_properties_to_rebuilt_models_quantizers(self):
        """
        Moves over the original models quantizer properties and sets them on the rebuilt model. Specifically, the enabled,
        and is_encoding_valid properties are copied over for both the python and C++ side of the quantizer.
        """
        self._assert_rebuilt_model_is_not_none()

        def get_tensor_quantizer_in_list(quantizer: _VARIATIONS_OF_TENSOR_QUANTIZERS_TYPES) -> List[TensorQuantizer]:
            """
            Helper function to get the tensor quantizer(s) in a list

            :param quantizer: The quantizer to return as is or in a list
            :return: list of tensor quantizer(s)
            """
            return quantizer.tensor_quantizer if isinstance(quantizer.tensor_quantizer, list) else [quantizer.tensor_quantizer]

        def assign_quantizers_properties(original_quantizers: _VARIATIONS_OF_TENSOR_QUANTIZERS_TYPES,
                                         rebuilt_quantizers: _VARIATIONS_OF_TENSOR_QUANTIZERS_TYPES):
            """
            Helper function to assign the quantizer properties

            :param original_quantizers: The original quantizers
            :param rebuilt_quantizers: The rebuilt quantizers
            """
            for original_quantizer, rebuilt_quantizer in zip(original_quantizers, rebuilt_quantizers):
                rebuilt_quantizer.enabled = original_quantizer.is_enabled()
                rebuilt_quantizer._is_encoding_valid = original_quantizer.is_encoding_valid()  # pylint: disable=protected-access

                rebuilt_tensor_quantizer = get_tensor_quantizer_in_list(rebuilt_quantizer)
                original_tensor_quantizer = get_tensor_quantizer_in_list(original_quantizer)

                for rebuilt_tq, original_tq in zip(rebuilt_tensor_quantizer, original_tensor_quantizer):
                    rebuilt_tq.isEncodingValid = original_tq.isEncodingValid

        def assert_number_of_quantizers_match(which_quantizer: str, original_layer: QcQuantizeWrapper,
                                              rebuilt_layer: QcQuantizeWrapper):
            """
            Helper function to assert the number of quantizers match

            :param which_quantizer: The quantizer type
            :param original_layer: The original layer
            :param rebuilt_layer: The rebuilt layer
            """
            assert len(original_layer.output_quantizers) == len(rebuilt_layer.output_quantizers), \
                f"Number of {which_quantizer} quantizers for layer {original_layer.name} does not match " \
                f"number of {which_quantizer} quantizers for rebuilt layer {rebuilt_layer.name}"

        for original_layer, rebuilt_layer in zip(self.original_quantsim_model.layers, self.rebuilt_model.layers):
            if isinstance(original_layer, QcQuantizeWrapper):
                _logger.debug("Copying quantizer properties for layer %s", original_layer.name)
                assert_number_of_quantizers_match('input', original_layer, rebuilt_layer)
                assign_quantizers_properties(original_layer.input_quantizers, rebuilt_layer.input_quantizers)

                assert_number_of_quantizers_match('output', original_layer, rebuilt_layer)
                assign_quantizers_properties(original_layer.output_quantizers, rebuilt_layer.output_quantizers)

                assert_number_of_quantizers_match('param', original_layer, rebuilt_layer)
                assign_quantizers_properties(original_layer.param_quantizers, rebuilt_layer.param_quantizers)

    def _assign_weights_not_connected_to_model(self):
        """
        This function adds weights that are not connected to the model to the rebuilt model
        """
        self._assert_rebuilt_model_is_not_none()

        if len(self.rebuilt_model.weights) != len(self.original_quantsim_model_weights):
            model_weight_names = RebuiltQuantSimModelFactory._get_model_weight_names_to_weight_values_dict(self.rebuilt_model)

            missing_weights = set(self.original_quantsim_model_weight_names_to_weights.keys()) - set(model_weight_names.keys())
            _logger.debug('Found %d weights that are not connected to the model', len(missing_weights))

            for weight in missing_weights:
                if not self.original_quantsim_model_weight_names_to_weights[weight].trainable:
                    self.rebuilt_model.add_weight(''.join(weight.split(_SPECIAL_PARSE_CHAR)[:-1]),  # To get original name back
                                                  self.original_quantsim_model_weight_names_to_weights[weight].shape,
                                                  self.original_quantsim_model_weight_names_to_weights[weight].dtype,
                                                  self.original_quantsim_model_weight_names_to_weights[weight].initializer)
            _logger.debug('Added %d weights to the rebuilt model', len(missing_weights))

    def rebuild_model(self) -> tf.keras.Model:
        """
        Rebuild the model after converting to a frozen pb. Freezing to a pb invalidates all Keras graphs.
        Meaning that the model needs to be rebuilt to allow users to still use sim.model after exporting.
        We do this by passing the original model and the parameters used to create the sim model again. This will
        rebuild the model with the same wrapping. We then set the weights of the rebuilt model to the weights of the
        original model. This will ensure that the weights are the same as the original model. We then copy over the
        quantizer properties from the original model to the rebuilt model. This will ensure that the quantizer
        properties are the same as the original model.

        :param original_quantsim_model_weights: The weights of the original model
        :return: The rebuilt model
        """
        from aimet_tensorflow.keras.quantsim import QuantizationSimModel # Import here to avoid circular import
        tf.keras.backend.clear_session()  # Clear the session to avoid naming issues

        self.rebuilt_model = QuantizationSimModel(self.original_model_without_wrappers,
                                                  **self.original_quantsim_params.__dict__).model

        self._assign_weights_not_connected_to_model()
        self.rebuilt_model.set_weights(self.original_quantsim_model_weights)
        self._copy_original_models_quantizer_properties_to_rebuilt_models_quantizers()

        _logger.debug("QuantSim model successfully rebuilt.")
        return self.rebuilt_model
