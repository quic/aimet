# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Code example for mixed precision """

# Step 0. Import statements
import torch
from aimet_common.defs import QuantizationDataType, CallbackFunc
from aimet_torch.mixed_precision import choose_mixed_precision
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.amp.mixed_precision_algo import GreedyMixedPrecisionAlgo
# End step 0


def quantize_with_mixed_precision(model):
    """
    Code example showing the call flow for Auto Mixed Precision
    """
    # Define parameters to pass to mixed precision algo
    dummy_input = torch.randn(1, 1, 28, 28).cuda()
    default_bitwidth = 16
    # ((activation bitwidth, activation data type), (param bitwidth, param data type))
    candidates = [((16, QuantizationDataType.int), (16, QuantizationDataType.int)),
                 ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
                 ((8, QuantizationDataType.int), (16, QuantizationDataType.int))]
    # Allowed accuracy drop in absolute value
    allowed_accuracy_drop = 0.5 # Implies 50% drop

    eval_callback_for_phase_1 = CallbackFunc(eval_callback_func, func_callback_args=5000)
    eval_callback_for_phase_2 = CallbackFunc(eval_callback_func, func_callback_args=None)

    forward_pass_call_back = CallbackFunc(forward_pass_callback, func_callback_args=dummy_input)

    # Create quant sim
    sim = QuantizationSimModel(model, default_param_bw=default_bitwidth, default_output_bw=default_bitwidth,
                               dummy_input=dummy_input)
    sim.compute_encodings(forward_pass_callback, forward_pass_callback_args=None)

    # Enable phase-3 (optional)
    # GreedyMixedPrecisionAlgo.ENABLE_CONVERT_OP_REDUCTION = True
    # Note: supported candidates ((8,int), (8,int)) & ((16,int), (8,int))

    # Call the mixed precision algo with clean start = True i.e. new accuracy list and pareto list will be generated
    # If set to False then pareto front list and accuracy list will be loaded from the provided directory path
    pareto_front_list = choose_mixed_precision(sim, dummy_input, candidates, eval_callback_for_phase_1,
                                               eval_callback_for_phase_2, allowed_accuracy_drop, results_dir='./data',
                                               clean_start=True, forward_pass_callback=forward_pass_call_back)

    print(pareto_front_list)
    sim.export("./data", str(allowed_accuracy_drop), dummy_input)


def quantize_with_mixed_precision_start_from_existing_cache(model):
    """
    Code example shows how to start from an existing cache when using the API of Auto Mixed Precision
    """
    # Define parameters to pass to mixed precision algo
    dummy_input = torch.randn(1, 1, 28, 28).cuda()
    default_bitwidth = 16
    # ((activation bitwidth, activation data type), (param bitwidth, param data type))
    candidates = [((16, QuantizationDataType.int), (16, QuantizationDataType.int)),
                 ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
                 ((8, QuantizationDataType.int), (16, QuantizationDataType.int))]
    # Allowed accuracy drop in absolute value
    allowed_accuracy_drop = 0.5 # Implies 50% drop

    eval_callback_for_phase_1 = CallbackFunc(eval_callback_func, func_callback_args=5000)
    eval_callback_for_phase_2 = CallbackFunc(eval_callback_func, func_callback_args=None)

    forward_pass_call_back = CallbackFunc(forward_pass_callback, func_callback_args=dummy_input)

    # Create quant sim
    sim = QuantizationSimModel(model, default_param_bw=default_bitwidth, default_output_bw=default_bitwidth,
                               dummy_input=dummy_input)
    sim.compute_encodings(forward_pass_callback, forward_pass_callback_args=None)

    # Enable phase-3 (optional)
    GreedyMixedPrecisionAlgo.ENABLE_CONVERT_OP_REDUCTION = True

    # Call the mixed precision algo with clean start = True i.e. new accuracy list and pareto list will be generated
    # If set to False then pareto front list and accuracy list will be loaded from the provided directory path
    # A allowed_accuracy_drop can be specified to export the final model with reference to the pareto list
    pareto_front_list = choose_mixed_precision(sim, dummy_input, candidates, eval_callback_for_phase_1,
                                               eval_callback_for_phase_2, allowed_accuracy_drop, results_dir='./data',
                                               clean_start=True, forward_pass_callback=forward_pass_call_back)

    print(pareto_front_list)
    sim.export("./data", str(allowed_accuracy_drop), dummy_input)

    # Set clean_start to False to start from an existing cache
    # Set allowed_accuracy_drop to 0.9 to export the 90% drop point in pareto list
    allowed_accuracy_drop = 0.9
    pareto_front_list = choose_mixed_precision(sim, dummy_input, candidates, eval_callback_for_phase_1,
                                               eval_callback_for_phase_2, allowed_accuracy_drop, results_dir='./data',
                                               clean_start=False, forward_pass_callback=forward_pass_call_back)
    print(pareto_front_list)
    sim.export("./data", str(allowed_accuracy_drop), dummy_input)


def forward_pass_callback(model, input_tensor):
    """ Call forward pass of model """
    # Note: A user can populate this function as per their model. This is a toy example to show how the API
    # for the function can look like
    return model(input_tensor)


def eval_callback_func(model, number_of_samples):
    """ Call eval function for model """
    # Note: A user can populate this function as per their model. This is a toy example to show how the API
    # for the function can look like
    model.perform_eval(number_of_samples)
