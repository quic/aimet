# /usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" QuantSim and QAT code example to be used for documentation generation. """

# Quantsim imports

import logging
import torch
import torch.cuda
from torch.utils.data import DataLoader
from torchvision import models
from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_torch.utils import create_fake_data_loader
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantsim import QuantizationSimModel

# End of import statements


def train(model: torch.nn.Module, data_loader: DataLoader) -> torch.Tensor:
    """
    This is intended to be the user-defined model train function.
    :param model: torch model
    :param data_loader: torch data loader
    :return: total loss
    """
    total_loss = 0
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for (data, labels) in data_loader:
        optimizer.zero_grad()
        data = data.cuda()
        labels = labels.cuda()
        predicted = model(data)
        loss = criterion(predicted, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss

    return total_loss


def evaluate(model: torch.nn.Module, forward_pass_callback_args):
    """
     This is intended to be the user-defined model evaluation function. AIMET requires the above signature. So if the
     user's eval function does not match this signature, please create a simple wrapper.
     Use representative dataset that covers diversity in training data to compute optimal encodings.

    :param model: Model to evaluate
    :param forward_pass_callback_args: These argument(s) are passed to the forward_pass_callback as-is. Up to
            the user to determine the type of this parameter. E.g. could be simply an integer representing the number
            of data samples to use. Or could be a tuple of parameters or an object representing something more complex.
            If set to None, forward_pass_callback will be invoked with no parameters.
    """
    dummy_input = torch.randn(1, 3, 224, 224).to(torch.device('cuda'))
    model.eval()
    with torch.no_grad():
        model(dummy_input)


def quantsim_example():

    AimetLogger.set_level_for_all_areas(logging.INFO)
    model = models.resnet18().eval()
    model.cuda()
    input_shape = (1, 3, 224, 224)
    dummy_input = torch.randn(input_shape).cuda()

    # Prepare model for Quantization SIM. This will automate some changes required in model definition for example
    # create independent modules for torch.nn.functional and reused modules
    prepared_model = prepare_model(model)

    # Instantiate Quantization SIM. This will insert simulation nodes in the model
    quant_sim = QuantizationSimModel(prepared_model, dummy_input=dummy_input,
                                     quant_scheme=QuantScheme.post_training_tf_enhanced,
                                     default_param_bw=8, default_output_bw=8,
                                     config_file='../../TrainingExtensions/common/src/python/aimet_common/quantsim_config/'
                                                 'default_config.json')

    # Compute encodings (min, max, delta, offset) for activations and parameters. Use representative dataset
    # roughly ~1000 examples
    quant_sim.compute_encodings(evaluate, forward_pass_callback_args=None)

    # QAT - Quantization Aware Training - Fine-tune the model fore few epochs to retain accuracy using train loop
    data_loader = create_fake_data_loader(dataset_size=32, batch_size=16, image_size=input_shape[1:])
    _ = train(quant_sim.model, data_loader)

    # Export the model which saves pytorch model without any simulation nodes and saves encodings file for both
    # activations and parameters in JSON format
    quant_sim.export(path='./', filename_prefix='quantized_resnet18', dummy_input=dummy_input.cpu())


if __name__ == '__main__':
    quantsim_example()
