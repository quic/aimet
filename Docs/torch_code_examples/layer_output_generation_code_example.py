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
import torch
from torchvision import models

from aimet_torch.onnx_utils import OnnxExportApiArgs
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.layer_output_utils import LayerOutputUtil, NamingScheme
# End step 0

# Step 1. Obtain original or quantsim model
# Obtain original model
original_model = models.resnet18()
original_model.eval()
original_model = prepare_model(original_model)

# Obtain quantsim model
dummy_input = torch.rand(1, 3, 224, 224)

def forward_pass(model: torch.nn.Module, input_batch: torch.Tensor):
    model.eval()
    with torch.no_grad():
        _ = model(input_batch)

quantsim = QuantizationSimModel(model=original_model, quant_scheme='tf_enhanced',
                                dummy_input=dummy_input, rounding_mode='nearest',
                                default_output_bw=8, default_param_bw=8, in_place=False)

quantsim.compute_encodings(forward_pass_callback=forward_pass,
                           forward_pass_callback_args=dummy_input)
# End step 1

# Step 2. Obtain pre-processed inputs
# Get the inputs that are pre-processed using the same manner while computing quantsim encodings
input_batches = get_pre_processed_inputs()
# End step 2

# Step 3. Generate outputs
# Generate layer-outputs
layer_output_util = LayerOutputUtil(model=quantsim.model, dir_path='./layer_output_dump', naming_scheme=NamingScheme.ONNX,
                                    dummy_input=dummy_input, onnx_export_args=OnnxExportApiArgs())
for input_batch in input_batches:
    layer_output_util.generate_layer_outputs(input_batch)
# End step 3
