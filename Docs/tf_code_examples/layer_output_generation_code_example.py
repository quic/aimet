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

""" Code example to generate intermediate layer outputs of a model """

# Step 0. Import statements
import numpy as np
import tensorflow as tf

from aimet_tensorflow.examples.test_models import keras_model
from aimet_tensorflow.quantsim import QuantizationSimModel
from aimet_tensorflow.layer_output_utils import LayerOutputUtil
# End step 0

# Step 1. Obtain original or quantsim model
# Load original model into session
def cpu_session():
    tf.compat.v1.reset_default_graph()
    with tf.device('/cpu:0'):
        model = keras_model()
        init = tf.compat.v1.global_variables_initializer()
    session = tf.compat.v1.Session()
    session.run(init)
    return session

session = cpu_session()

# Obtain quantsim model session
def quantsim_forward_pass_callback(session, dummy_input):
    model_input = session.graph.get_tensor_by_name('conv2d_input:0')
    model_output = session.graph.get_tensor_by_name('keras_model/Softmax_quantized:0')
    return session.run(model_output, feed_dict={model_input: dummy_input})

dummy_input = np.random.randn(1, 16, 16, 3)

quantsim = QuantizationSimModel(session, ['conv2d_input'], ['keras_model/Softmax'], use_cuda=False)
quantsim.compute_encodings(quantsim_forward_pass_callback, dummy_input)
# End step 1

# Step 2. Obtain pre-processed inputs
# Get the inputs that are pre-processed using the same manner while computing quantsim encodings
input_batches = get_pre_processed_inputs()
# End step 2

# Step 3. Generate outputs
# Generate layer-outputs
layer_output_util = LayerOutputUtil(session=quantsim.session, starting_op_names=['conv2d_input'],
                                    output_op_names=['keras_model/Softmax'], dir_path='./layer_output_dump')
for input_batch in input_batches:
    layer_output_util.generate_layer_outputs(input_batch)
# End step 3
