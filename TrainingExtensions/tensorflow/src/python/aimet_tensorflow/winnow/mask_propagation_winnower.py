# /usr/bin/env python3.5
# -*- mode: python -*-
#  =============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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
#
#  =============================================================================

""" Winnow the API provided input channels from the modules in a tf model. """

from typing import List, Tuple, Dict
import tensorflow as tf
from aimet_common.utils import AimetLogger, ModelApi
from aimet_common.winnow.mask import Mask
from aimet_common.winnow.mask_propagation_winnower import MaskPropagationWinnower as aimetCommonMaskPropagationWinnower
from aimet_common.winnow.mask_propagator import MaskPropagator
from aimet_tensorflow.common.connectedgraph import ConnectedGraph
from aimet_tensorflow.winnow.module_reducer import ModuleReducer


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Winnow)


class MaskPropagationWinnower(aimetCommonMaskPropagationWinnower):
    """
    The MaskPropagationWinnower class implements winnowing based on propagating masks corresponding to
    each module's input channels identified to be winnowed.
    """

    def __init__(self, sess: tf.compat.v1.Session, input_op_names: List[str], output_op_names: List[str],
                 list_of_modules_to_winnow: List[Tuple[tf.Operation, List]] = None, reshape=True,
                 in_place=False, verbose=False):
        """
        MaskPropagationWinnower object initialization.

        :param sess: The model to be winnowed.
        :param input_op_names: Input operations to the model.
        :param output_op_names: List of output op names of the model, used to help ConnectedGraph determine valid ops
        (to ignore training ops for example).
        :param list_of_modules_to_winnow: A list of Tuples with each Tuple containing a module and a list of
                    channels to be winnowed for that module.
        :param reshape: f set to True a Down Sample Layer is added between modules to match the number of channels.
                    If set to False, the modules that need a Down Sample Layer will not be winnowed.
        :param in_place: If set to True, the model will be winnowed in place.
                     If set to False, a copy of the model will be winnowed.
        :param verbose: If set to True, logs detailed winnowing log messages.
        """

        super().__init__(list_of_modules_to_winnow, reshape, in_place, verbose)

        debug_level = logger.getEffectiveLevel()
        logger.debug("Current log level: %s", debug_level)

        self._conn_graph = ConnectedGraph(sess.graph, input_op_names, output_op_names)
        self._modules_by_name = None
        self._mask_propagator = MaskPropagator(self._conn_graph, model_api=ModelApi.tensorflow)
        self._module_reducer = ModuleReducer(self._conn_graph, sess, using_cuda=False, reshape=False,
                                             op_to_mask_dict=self._mask_propagator.op_to_mask_dict)

    def propagate_masks_and_winnow(self):
        """  For the modules to be winnowed, create and propagate the masks.
        Once mask propagation is completed, winnow the model. """

        # Propagate the masks
        self._propagate_masks()

        modified_op_list = self._mask_propagator.get_ops_with_non_default_ip_op_masks()
        for name in modified_op_list:
            logger.info("Modified Op: %s", name)

        new_sess, modified_modules_dict = self._module_reducer.reduce_modules()

        if modified_modules_dict:
            ordered_module_list = self._create_modified_modules_list(modified_modules_dict)
        else:
            ordered_module_list = None
            logger.info("No modules were winnowed. Original model is returned.")

        return new_sess, ordered_module_list

    def _propagate_masks(self):
        """  For the modules to be winnowed, set the channels to winnow and propagate the masks."""

        for tf_op, list_of_channels_to_winnow in self._list_of_modules_to_winnow:
            self.validate_winnow_api_parameters(tf_op, list_of_channels_to_winnow)

            input_channels_to_winnow = list_of_channels_to_winnow
            output_channels_to_winnow = None
            if tf_op.type in ['Conv2D', 'Dense']:
                self._mask_propagator.update_channels_to_winnow(tf_op.name, self._reshape, input_channels_to_winnow,
                                                                output_channels_to_winnow)

        # The channels to winnow have been updated
        # Propagate the masks.
        self._mask_propagator.propagate_masks()

    @staticmethod
    def _create_modified_modules_list(modified_modules: Dict[str, Tuple[tf.Operation, Mask]]):
        """
        Creates and returns a list of tuples with each tuple containing the original module and its replacement module
        :param modified_modules: Dictionary mapping names of ops before winnow to a tuple of
        (name after winnow, op mask)

        """

        modified_module_list = []

        for orig_module_name, (new_module, op_mask) in modified_modules.items():
            # Remove prefix of the model name
            # E.g. the module_name maybe Net.layer1.conv1, we only want layer1.conv1
            first_dot_position = orig_module_name.find('.')
            if first_dot_position != -1:
                orig_module_name = orig_module_name[first_dot_position + 1:]
            modified_module_list.append((orig_module_name, new_module, op_mask.input_channel_masks,
                                         op_mask.output_channel_masks))

        return modified_module_list

    def validate_winnow_api_parameters(self, tf_op: tf.Operation, list_of_channels_to_winnow: List[int]):
        """

        For a given module, validate Winnow API parameters.

        :param tf_op: tf operation whose channel numbers are being validated.
        :param list_of_channels_to_winnow: list of channels to be winnowed.
        :return:
        """

        if not tf_op.type == "Conv2D":
            logger.critical("Winnowing is currently only supported for Conv2d modules. Attempting to winnow "
                            "module of type %s",
                            tf_op.type)
            raise NotImplementedError(tf_op.type)

        # Validate the list of channels.
        num_channels_to_winnow = len(list_of_channels_to_winnow)
        if num_channels_to_winnow == 0:
            raise ValueError("The list of channels to winnow is empty for the module: %s" % tf_op.name)

        module_op = self._conn_graph.get_op_from_module_name(tf_op.name)

        max_channel_num = max(list_of_channels_to_winnow)
        max_in_channel_index = module_op.num_in_channels - 1
        if max_channel_num > max_in_channel_index:
            raise ValueError("Channel number: %s exceeds module's max channel number index: %s for module: %s" %
                             (max_channel_num, max_in_channel_index, tf_op.name))

        if num_channels_to_winnow == module_op.num_in_channels:
            raise ValueError("Winnowing all the input channels is not allowed, module: %s" % tf_op.name)

        input_index = 0     # Using op index 0 to examine input to op
        if module_op.inputs[input_index].is_model_input:
            error_msg = (f'Winnowing the first module of a model is NOT supported. Please ignore the first '
                         f'module and try again. First module: {module_op.dotted_name}, shape '
                         f'{module_op.inputs[input_index].shape}, channels to winnow: {list_of_channels_to_winnow}')
            logger.error(error_msg)
            raise NotImplementedError(error_msg)
