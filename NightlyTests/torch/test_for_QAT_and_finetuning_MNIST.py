#/usr/bin/env python2.7
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#  
#  Copyright (c) 2017-2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

import unittest
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from aimet_torch.quantsim import QuantizationSimModel, load_checkpoint, save_checkpoint
from aimet_torch.examples import mnist_torch_model
import aimet_torch.examples.mnist_torch_model as mnist_model
from aimet_torch.qc_quantize_op import QcPostTrainingWrapper
from aimet_common.defs import QuantScheme

path = str('../data')
filename_prefix = 'quantized_mnist'


def check_if_layer_weights_are_updating(model, batch_idx):
    """
    helper function to check if weights are updating during training
    :param model: quantized MNIST model, expects first layer to be conv1
    :param batch_idx: batch id
    :return: None
    """
    # Creating an alias for easier reference
    f = check_if_layer_weights_are_updating

    # get the initial values of some layers
    conv1_w_value = model.conv1._module_to_wrap.weight

    if batch_idx != 0:
        assert not np.allclose(conv1_w_value.cpu().detach().numpy(), f.conv1_w_value_old.cpu().detach().numpy())
    else:
        f.conv1_w_value_old = conv1_w_value.clone()


class QuantizationSimAcceptanceTests(unittest.TestCase):

    @staticmethod
    def forward_pass(model, iterations):
        mnist_torch_model.evaluate(model=model, iterations=iterations, use_cuda=True)

    def test_with_finetuning(self):

        torch.cuda.empty_cache()

        model = mnist_model.Net().to(torch.device('cuda'))
        mnist_torch_model.evaluate(model=model, iterations=None, use_cuda=True)

        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 28, 28).cuda())

        # Quantize the untrained MNIST model
        sim.compute_encodings(self.forward_pass, forward_pass_callback_args=5)

        # Run some inferences
        mnist_torch_model.evaluate(model=sim.model, iterations=None, use_cuda=True)

        # train the model again
        mnist_model.train(sim.model, epochs=1, num_batches=3,
                          batch_callback=check_if_layer_weights_are_updating, use_cuda=True)

    def test_retraining_on_quantized_model_first_step(self):

        torch.cuda.empty_cache()

        model = mnist_model.Net().to(torch.device('cuda'))

        sim = QuantizationSimModel(model,
                                   default_output_bw=4,
                                   default_param_bw=4,
                                   dummy_input=torch.rand(1, 1, 28, 28).cuda())

        # Quantize the untrained MNIST model
        sim.compute_encodings(self.forward_pass, forward_pass_callback_args=5)

        # train the model for entire one epoch
        mnist_model.train(model=sim.model, epochs=1, num_batches=3,
                          batch_callback=check_if_layer_weights_are_updating, use_cuda=True)

        # Checkpoint the model
        save_checkpoint(sim, os.path.join(path, 'checkpoint.pt'))

    def test_retraining_on_quantized_model_second_step(self):

        torch.cuda.empty_cache()

        sim = load_checkpoint(os.path.join(path, 'checkpoint.pt'))

        # re-train the model for entire one epoch
        mnist_model.train(model=sim.model, epochs=1, num_batches=3,
                          batch_callback=check_if_layer_weights_are_updating, use_cuda=True)

    # TODO: Revisit to resolve error when wrapping stand alone op
    @unittest.skip
    def test_manual_mode(self):

        torch.cuda.empty_cache()

        net = mnist_model.Net()

        model = net.to(torch.device('cuda'))
        # Adding wrapper to first convolution layer
        for module_name, module_ref in model.named_children():
            if module_name is 'conv1':
                quantized_module = QcPostTrainingWrapper(module_ref, weight_bw=8, activation_bw=8, round_mode='nearest',
                                                         quant_scheme=QuantScheme.post_training_tf)
                setattr(model, module_name, quantized_module)

        sim = QuantizationSimModel(model, dummy_input=torch.rand(1, 1, 28, 28).cuda())

        # Quantize the untrained MNIST model
        sim.compute_encodings(self.forward_pass, forward_pass_callback_args=5)

        # Run some inferences
        mnist_torch_model.evaluate(model=sim.model, iterations=100, use_cuda=True)

        # train the model again
        mnist_model.train(model=sim.model, epochs=1, num_batches=3,
                          batch_callback=check_if_layer_weights_are_updating, use_cuda=True)
