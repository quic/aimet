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
import torch
import torch.cuda
from torchvision import models

from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_torch.utils import create_fake_data_loader
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters

# End of import statements


def dummy_forward_pass(model: torch.nn.Module, forward_pass_callback_args) -> float:
    """
    This is intended to be the user-defined model evaluation function.
    AIMET requires the above signature. So if the user's eval function does not
    match this signature, please create a simple wrapper.

    :param model: Model to evaluate
    :param forward_pass_callback_args: These argument(s) are passed to the forward_pass_callback as-is. Up to
            the user to determine the type of this parameter. E.g. could be simply an integer representing the number
            of data samples to use. Or could be a tuple of parameters or an object representing something more complex.
            If set to None, forward_pass_callback will be invoked with no parameters.
    :return: single float number (accuracy) representing model's performance
    """
    return .5


def apply_adaround_example():

    AimetLogger.set_level_for_all_areas(logging.DEBUG)
    torch.cuda.empty_cache()

    model = models.resnet18(pretrained=True).eval()
    model = model.to(torch.device('cuda'))
    input_shape = (1, 3, 224, 224)
    dummy_input = torch.randn(input_shape).to(torch.device('cuda'))

    # As an illustrating example, a fake data loader is used here.
    # For AdaRound, the user should provide the training data loader.
    data_loader = create_fake_data_loader(dataset_size=64, batch_size=16, image_size=input_shape[1:])

    params = AdaroundParameters(data_loader=data_loader, num_batches=4, default_num_iterations=50,
                                default_reg_param=0.01, default_beta_range=(20, 2))

    # Returns model with adarounded weights and their corresponding encodings
    adarounded_model = Adaround.apply_adaround(model, dummy_input, params, path='./',
                                               filename_prefix='resnet18', default_param_bw=4,
                                               default_quant_scheme=QuantScheme.post_training_tf_enhanced,
                                               default_config_file=None)

    # Create QuantSim using adarounded_model
    sim = QuantizationSimModel(adarounded_model, quant_scheme=quant_scheme, default_param_bw=param_bw,
                               default_output_bw=output_bw, dummy_input=dummy_input)

    # Set and freeze encodings to use same quantization grid and then invoke compute encodings
    sim.set_and_freeze_param_encodings(encoding_path='./resnet18.encodings')
    sim.compute_encodings(dummy_forward_pass, forward_pass_callback_args=None)

if __name__ == '__main__':
    apply_adaround_example()
