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
# pylint: skip-file
""" Code example to generate intermediate layer outputs of a model """

# Step 0. Import statements
import numpy as np
import tensorflow as tf

from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.layer_output_utils import LayerOutputUtil
# End step 0

# Step 1. Obtain original or quantsim model
def quantsim_forward_pass_callback(model, dummy_input):
    _ = model.predict(dummy_input)

# Load the baseline/original (FP32) model
base_model = load_baseline_model()

dummy_input = np.random.rand(1, 16, 16, 3)

# Create QuantizationSim Object
quantsim_obj = QuantizationSimModel(
    model=base_model,
    quant_scheme='tf_enhanced',
    rounding_mode="nearest",
    default_output_bw=8,
    default_param_bw=8,
    in_place=False,
    config_file=None
)

# Compute encodings
quantsim_obj.compute_encodings(quantsim_forward_pass_callback,
                      forward_pass_callback_args=dummy_input
                      )
# End step 1

# Step 2. Obtain pre-processed inputs
# Get the inputs that are pre-processed using the same manner while computing quantsim encodings
input_batches = get_pre_processed_inputs()
# End step 2

# Step 3. Generate outputs
# Generate layer-outputs
layer_output_util = LayerOutputUtil(model=quantsim_obj.model, save_dir="./KerasLayerOutput")
for input_batch in input_batches:
    layer_output_util.generate_layer_outputs(input_batch=input_batch)
# End step 3
