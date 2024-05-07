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
"""

.. _tutorials-quickstart-guide:

Quickstart Guide
================

In this tutorial, we will go through the end-to-end process of using AIMET and PyTorch to create, calibrate, and export
a simple quantized model. Note that this is intended to show the most basic workflow in AIMET. It is *not* meant to
demonstrate the most state-of-the-art techniques available in AIMET.

Overall flow
------------

1. Define the basic floating-point PyTorch model, training, and eval loops
2. Prepare the trained model for quantization
3. Create quantization simulation (quantsim) model in AIMET to simulate the effects of quantization
4. Calibrate the quantsim model on training data and evaluate the quantized accuracy
5. Fine-tune the quantized model to improve the quantized accuracy
6. Export the quantized model


PyTorch prerequisites
---------------------
To see clearly what happens inside AIMET, let's first start with some simple PyTorch code for defining, training, and
evaluating a model. The code below is adapted from PyTorch's
`basic optimization tutorial <https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html>`_.
Note that AIMET does not have any special requirement on what these training/eval loops look like.

"""

import torch
import torchvision
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 1) Start with some data loaders to train, evaluate, and calibrate the model

cifar10_train_data = torchvision.datasets.FashionMNIST('/tmp/cifar10', train=True, download=True, transform=torchvision.transforms.ToTensor())
cifar10_test_data = torchvision.datasets.FashionMNIST('/tmp/cifar10', train=True, download=True, transform=torchvision.transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(cifar10_train_data, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(cifar10_train_data, batch_size=128, shuffle=True)

# 2) Define a simple model to train on this dataset

class Network(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.bn_1 = torch.nn.BatchNorm2d(128)
        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.bn_2 = torch.nn.BatchNorm2d(256)
        self.linear = torch.nn.Linear(in_features=7*7*256, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn_1(x))
        x = self.conv2(x)
        x = F.relu(self.bn_2(x))
        x = self.linear(x.view(x.shape[0], -1))
        return F.softmax(x, dim=-1)


# 3) Define an evaluation loop for the model

def evaluate(model, data_loader):
    model.eval()
    correct = total = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        correct += (torch.argmax(output, dim=1) == y).sum()
        total += x.shape[0]

    accuracy = correct / total * 100.
    return accuracy

###############################################################################
# Now, let's instantiate a network and train for a few epochs on our dataset to establish a baseline floating-point model

# Create a model
model = Network()

# Send the model to the desired device (optional)
model.to(device)

# Define some loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train for 4 epochs
model.train()
for epoch in range(4):
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Evaluate the floating-point model
model.eval()
fp_accuracy = evaluate(model, test_loader)
print(f"Floating point accuracy: {fp_accuracy}")

###############################################################################
# Prepare the floating point model for quantization
# -------------------------------------------------
#
# Before we can (accurately) simulate quantization, there are a couple important steps to take care of:
#
#
# 1) Model preparation
# ^^^^^^^^^^^^^^^^^^^^
# AIMET's quantization simulation tool (:class:`QuantizationSimModel`) expects the floating point model to conform to some
# specific guidelines. For example, :class:`QuantizationSimModel` is only able to quantize math operations performed by
# :class:`torch.nn.Module` objects, whereas :mod:`torch.nn.functional` calls will be (incorrectly) ignored.
#
# If we look back at our previous model definition, we see it calls :func:`F.relu` and :func:`F.softmax` in the forward
# function. Does this mean we need to completely redefine our model to use AIMET? Thankfully, no. AIMET provides the
# :mod:`model_preparer` API to transform our incompatible model into a new fully-compatible model.

from aimet_torch import model_preparer

prepared_model = model_preparer.prepare_model(model)
print(prepared_model)

# Note: This transformation should not change the model's forward function at all
fp_accuracy_prepared = evaluate(prepared_model, test_loader)
assert fp_accuracy_prepared == fp_accuracy

###############################################################################
# Note how the prepared model now contains distinct modules for the :func:`relu` and :func:`softmax` operations.
#
# 2) BatchNorm fold
# ^^^^^^^^^^^^^^^^^
#
# When models are executed in a quantized runtime, batchnorm layers are typically folded into the weight and bias of
# an adjacent convolution layer whenever possible in order to remove unnecessary computations. To accurately simulate
# inference in these runtimes, it is generally a good idea to perform this batchnorm folding on the floating point model
# before applying quantization. AIMET provides the :mod:`batch_norm_fold` tool to do this.

from aimet_torch import batch_norm_fold

sample_input, _ = next(iter(train_loader))
batch_norm_fold.fold_all_batch_norms(prepared_model, input_shapes=sample_input.shape)

print(prepared_model)

###############################################################################
# Note that the model now has :class:`Identity` (passthrough) layers where it previously had :class:`BatchNorm2d` layers. Like the
# :mod:`model_preparer` step, this operation should not impact the model's accuracy.
#
# Quantize the model
# ------------------
#
# Now, we are ready to use AIMET's :class:`QuantizationSimModel` to simulate quantizing the floating point model. This
# involves two steps:
#
# 1) Add quantizers to simulate quantization noise during the model's forward pass
# 2) Calibrate the quantizer encodings (e.g., min/max ranges) on some sample inputs
#
# Calibration is necessary to determine the range of values each activation quantizer is likely to encounter in the
# model's forward pass, and should therefore be able to represent. Theoretically, we could pass the entire training
# dataset through the model for calibration, but in practice we usually only need about 500-1000 representative samples
# to accurately estimate the ranges.

import aimet_torch.v2 as aimet
from aimet_torch.v2 import quantsim

# QuantizationSimModel will convert each nn.Module in prepared_model into a quantized equivalent module and configure the module's quantizers
# In this case, we will quantize all parameters to 4 bits and all activations to 8 bits.
sim = quantsim.QuantizationSimModel(prepared_model,
                                    dummy_input=sample_input.to(device),
                                    default_output_bw=8,                                # Simulate 8-bit activations
                                    default_param_bw=4)                                 # Simulate 4-bit weights

# Inside the compute_encodings context, quantizers will observe the statistics of the activations passing through them. These statistics will be used
# to compute properly calibrated encodings upon exiting the context.
with aimet.nn.compute_encodings(sim.model):
    for idx, (x, _) in enumerate(train_loader):
        x = x.to(device)
        sim.model(x)
        if idx >= 10:
            break

# Compare the accuracy before and after quantization:
quantized_accuracy = evaluate(sim.model, test_loader)

print(sim.model)

print(f"Floating point model accuracy: {fp_accuracy} %\n"
      f"Quantized model accuracy: {quantized_accuracy} %")

###############################################################################
# Here, we can see that ``sim.model`` is nothing more than the ``prepared_model`` with every layer replaced with a
# quantized version of the layer. The quantization behavior of each module is determined by the configuration of its
# held quantizers.
#
# For example, we can see that ``sim.model.conv2`` has a 4-bit weight quantizer and an 8-bit output quantizer as specified
# during construction. We will discuss more advanced ways to configure these quantizers to optimize performance and
# accuracy in a later tutorial.
#
# Fine-tune the model with quantization aware training
# ----------------------------------------------------
#
# If we're not satisfied with our accuracy after applying quantization, there are some steps we can take to further
# optimize the quantized accuracy. One such step is quantization aware training (QAT), during which the model is trained
# with the fake-quantization ops present.
#
# Let's repeat our floating-point training loop for one more epoch, but this time use the quantized model.

# Define some loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(sim.model.parameters(), lr=1e-4)

# Train for one more epoch on the quantsim model
for epoch in range(1):
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        output = sim.model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


# Compare the accuracy before and after QAT:
post_QAT_accuracy = evaluate(sim.model, test_loader)

print(f"Original quantized model accuracy: {quantized_accuracy} %\n"
      f"Post-QAT model accuracy: {post_QAT_accuracy} %")

###############################################################################
# Export the quantsim model
# -------------------------
#
# Now that we are happy with our quantized model's accuracy, we are ready to export the model with its quantization parameters.

export_path = "/tmp/"
model_name = "fashion_mnist_model"
sample_input, _ = next(iter(train_loader))

sim.export(export_path, model_name, dummy_input=sample_input)

###############################################################################
# This export method will save the model with quantization nodes removed, along with an encodings file containing
# quantization parameters for each activation and weight tensor in the model. These artifacts can then be sent to a
# quantized runtime such as Qualcomm\ |reg| Neural Processing SDK.
#
# .. |reg|    unicode:: U+000AE .. REGISTERED SIGN

