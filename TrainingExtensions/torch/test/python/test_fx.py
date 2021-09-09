# /usr/bin/env python3.5
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

import pytest
import copy
from packaging import version
import torch
from torchvision import models

from aimet_torch import elementwise_ops
from aimet_torch.examples.test_models import ModelWithFunctionalReLU, SingleResidual, ModelWithDuplicateReLU


class TestFX:


    def test_fx_with_relu(self):
        """
        """
        if version.parse(torch.__version__) >= version.parse("1.8"):
            from aimet_torch.fx import replace_functional_by_module
            input_shape = (1, 3, 32, 32)
            input_tensor = torch.randn(*input_shape)
            model = ModelWithFunctionalReLU().eval()
            model_copy = copy.deepcopy(model)

            model_transformed = replace_functional_by_module(model_copy)
            print(model_transformed)

            assert isinstance(model_transformed.module_relu, torch.nn.ReLU)
            assert isinstance(model_transformed.module_relu_1, torch.nn.ReLU)
            assert isinstance(model_transformed.module_relu_2, torch.nn.ReLU)
            assert isinstance(model_transformed.module_relu_3, torch.nn.ReLU)

            assert torch.allclose(model(input_tensor), model_transformed(input_tensor))

    def test_fx_with_add(self):
        """
        """
        if version.parse(torch.__version__) >= version.parse("1.8"):
            from aimet_torch.fx import replace_functional_by_module
            input_shape = (1, 3, 32, 32)
            input_tensor = torch.randn(*input_shape)
            model = SingleResidual().eval()
            model_copy = copy.deepcopy(model)

            model_transformed = replace_functional_by_module(model_copy)
            print(model_transformed)

            assert isinstance(model_transformed.module_add, elementwise_ops.Add)
            assert torch.allclose(model(input_tensor), model_transformed(input_tensor))

    def test_fx_with_duplicate_relu(self):
        """
        """
        if version.parse(torch.__version__) >= version.parse("1.8"):
            from aimet_torch.fx import replace_functional_by_module
            input_shape = (1, 3, 32, 32)
            input_tensor = torch.randn(*input_shape)
            model = ModelWithDuplicateReLU().eval()
            model_copy = copy.deepcopy(model)

            model_transformed = replace_functional_by_module(model_copy)

            assert isinstance(model_transformed.relu, torch.nn.ReLU)
            assert isinstance(model_transformed.module_relu_1, torch.nn.ReLU)
            assert isinstance(model_transformed.module_relu_2, torch.nn.ReLU)
            assert isinstance(model_transformed.module_relu_3, torch.nn.ReLU)

            assert torch.allclose(model(input_tensor), model_transformed(input_tensor))

    def test_fx_with_resnet18(self):
        """
        """
        if version.parse(torch.__version__) >= version.parse("1.8"):
            from aimet_torch.fx import replace_functional_by_module
            input_shape = (1, 3, 224, 224)
            input_tensor = torch.randn(*input_shape)
            model = models.resnet18().eval()
            model_copy = copy.deepcopy(model)

            model_transformed = replace_functional_by_module(model_copy)
            print(model_transformed)

            assert torch.allclose(model(input_tensor), model_transformed(input_tensor))

    @pytest.mark.cuda
    def test_fx_with_resnet18_with_cuda(self):
        """
        """
        if version.parse(torch.__version__) >= version.parse("1.8"):
            from aimet_torch.fx import replace_functional_by_module
            input_shape = (1, 3, 224, 224)
            input_tensor = torch.randn(*input_shape).cuda()
            model = models.resnet18().cuda().eval()
            model_copy = copy.deepcopy(model)

            model_transformed = replace_functional_by_module(model_copy)
            print(model_transformed)

            assert torch.allclose(model(input_tensor), model_transformed(input_tensor))
