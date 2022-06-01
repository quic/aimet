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

""" Sample input activation to quantized op and output activation from original op for Adaround feature """

from typing import List, Tuple
import numpy as np
import tensorflow as tf

# Import AIMET specific modules
from aimet_tensorflow.utils.common import create_input_feed_dict, iterate_tf_dataset
from aimet_tensorflow.utils.op.conv import BiasUtils


class ActivationSampler:
    """
    Collect op's output activation data from unquantized model and input activation data from quantized model with
    all the preceding op's weights are quantized
    """
    def __init__(self, data_set: tf.compat.v1.data.Dataset):
        """
        :param data_set: Data set
        """
        # pylint: disable=protected-access
        self._dataset = data_set

    def sample_activation(self, orig_op: tf.Operation, quant_op: tf.Operation,
                          orig_session: tf.compat.v1.Session, quant_session: tf.compat.v1.Session,
                          inp_op_names: List, num_batches: int) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        From the original op, collect output activations and input activations to corresponding quantized op.
        :param orig_op: Single un quantized op from the original session
        :param quant_op: Corresponding quant op from the Quant sim session
        :param orig_session: Session with the original model
        :param quant_session: Session with the model with quantization simulations ops
        :param inp_op_names : input Op names, should be same in both models
        :param num_batches: Number of batches
        :return: Input data to quant op, Output data from original op
        """
        # pylint: disable=too-many-locals
        all_inp_data = []
        all_out_data = []

        # Initialize the iterator
        dataset_iterator = iterate_tf_dataset(self._dataset)

        for batch_index in range(num_batches):

            try:
                model_inputs = next(dataset_iterator)

                # batch is of shape (model_inputs, labels)
                if isinstance(model_inputs, (tuple, list)):
                    model_inputs, _ = model_inputs

                # Collect output activation data from original op
                feed_dict = create_input_feed_dict(orig_session.graph, inp_op_names, model_inputs)
                output_tensor = self._get_output_tensor(orig_op)
                out_data = orig_session.run(output_tensor, feed_dict=feed_dict)

                # Collect input activation data to quant sim op
                feed_dict = create_input_feed_dict(quant_session.graph, inp_op_names, model_inputs)
                input_tensor = self._get_input_tensor(quant_op)
                inp_data = quant_session.run(input_tensor, feed_dict=feed_dict)

                all_inp_data.append(inp_data)
                all_out_data.append(out_data)

                if batch_index == num_batches - 1:
                    break

            except tf.errors.OutOfRangeError:
                raise StopIteration("Can not fetch {} batches from dataset.".format(num_batches))

        all_inp_data = np.vstack(all_inp_data)
        all_out_data = np.vstack(all_out_data)

        return all_inp_data, all_out_data

    @staticmethod
    def _get_output_tensor(op: tf.Operation) -> tf.Tensor:
        """
        Output tensor for given op
        :param op: Tf operation
        :return: Output tensor
        """
        output_tensor = op.outputs[0]

        # If followed by bias op
        if not BiasUtils.is_bias_none(op):
            output_tensor = op.outputs[0].consumers()[0].outputs[0]

        return output_tensor

    @staticmethod
    def _get_input_tensor(op: tf.Operation) -> tf.Tensor:
        """
        Input tensor for given op
        :param op: Tf operation
        :return: Input tensor
        """
        input_tensor = op.inputs[0]
        if op.type == 'Conv2DBackpropInput':
            input_tensor = op.inputs[2]
        return input_tensor
