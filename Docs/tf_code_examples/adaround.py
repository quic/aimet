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
# pylint: skip-file

""" AdaRound code example to be used for documentation generation. """

# AdaRound imports

import logging
import numpy as np
import tensorflow as tf

from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_tensorflow.examples.test_models import keras_model
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.adaround.adaround_weight import Adaround, AdaroundParameters

# End of import statements
tf.compat.v1.disable_eager_execution()

def dummy_forward_pass(session: tf.compat.v1.Session, _):
    """
    This is intended to be the user-defined model evaluation function.
    AIMET requires the above signature. So if the user's eval function does not
    match this signature, please create a simple wrapper.
    :param session: Session with model to be evaluated
    :param _: These argument(s) are passed to the forward_pass_callback as-is. Up to
            the user to determine the type of this parameter. E.g. could be simply an integer representing the number
            of data samples to use. Or could be a tuple of parameters or an object representing something more complex.
            If set to None, forward_pass_callback will be invoked with no parameters.
    :return: single float number (accuracy) representing model's performance
    """
    input_data = np.random.rand(32, 16, 16, 3)
    input_tensor = session.graph.get_tensor_by_name('conv2d_input:0')
    output_tensor = session.graph.get_tensor_by_name('keras_model/Softmax:0')
    output = session.run(output_tensor, feed_dict={input_tensor: input_data})
    return output


def apply_adaround_example():

    AimetLogger.set_level_for_all_areas(logging.DEBUG)
    tf.compat.v1.reset_default_graph()

    _ = keras_model()
    init = tf.compat.v1.global_variables_initializer()
    dataset_size = 32
    batch_size = 16
    possible_batches = dataset_size // batch_size
    input_data = np.random.rand(dataset_size, 16, 16, 3)
    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    dataset = dataset.batch(batch_size=batch_size)

    session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
    session.run(init)

    params = AdaroundParameters(data_set=dataset, num_batches=possible_batches, default_num_iterations=10)
    starting_op_names = ['conv2d_input']
    output_op_names = ['keras_model/Softmax']

    # W4A8
    param_bw = 4
    output_bw = 8
    quant_scheme = QuantScheme.post_training_tf_enhanced

    # Returns session with adarounded weights and their corresponding encodings
    adarounded_session = Adaround.apply_adaround(session, starting_op_names, output_op_names, params, path='./',
                                                 filename_prefix='dummy', default_param_bw=param_bw,
                                                 default_quant_scheme=quant_scheme, default_config_file=None)

    # Create QuantSim using adarounded_session
    sim = QuantizationSimModel(adarounded_session, starting_op_names, output_op_names, quant_scheme,
                               default_output_bw=output_bw, default_param_bw=param_bw, use_cuda=False)

    # Set and freeze encodings to use same quantization grid and then invoke compute encodings
    sim.set_and_freeze_param_encodings(encoding_path='./dummy.encodings')
    sim.compute_encodings(dummy_forward_pass, None)

    session.close()
    adarounded_session.close()

if __name__ == '__main__':
    apply_adaround_example()
