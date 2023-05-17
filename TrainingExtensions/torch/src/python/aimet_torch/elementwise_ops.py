# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Custom modules for functional operations defined under torch and torch.nn.functional packages """

from typing import Callable, Any
import torch
import torch.nn


def forward_function_wrapper(functional: Callable) -> Any:
    """
    Wrapper function returning forward method for given functional operation.

    :param functional: torch.nn.functional
    :return: forward method
    """
    @staticmethod
    def forward(*args, **kwargs) -> Any:
        """
        Forward-pass routine for the functional operation.
        """
        return functional(*args, **kwargs)
    return forward

def create_wrapper_module(class_name: str, functional: Callable) -> Callable:
    """
    Dynamically create wrapper module for a functional operation.

    :param class_name: Name of the class.
    :param functional: Functional operation.
    :return: Module.
    """
    wrapped_module = type(class_name, (torch.nn.Module,), {'forward': forward_function_wrapper(functional)})
    return wrapped_module


# modules for functional operations under torch package
Subtract = create_wrapper_module('Subtract', torch.sub)
Divide = create_wrapper_module('Divide', torch.div)
FloorDivide = create_wrapper_module('FloorDivide', torch.floor_divide)
MatMul = create_wrapper_module('MatMul', torch.matmul)
Norm = create_wrapper_module('Norm', torch.norm)
Exponential = create_wrapper_module('Exponential', torch.exp)
Erf = create_wrapper_module('Erf', torch.erf)
Sqrt = create_wrapper_module('Sqrt', torch.sqrt)
Maximum = create_wrapper_module('Maximum', torch.maximum)
Minimum = create_wrapper_module('Minimum', torch.minimum)
Where = create_wrapper_module('Where', torch.where)
Greater = create_wrapper_module('Greater', torch.gt)
Less = create_wrapper_module('Less', torch.lt)
GreaterEqual = create_wrapper_module('GreaterEqual', torch.ge)
LessEqual = create_wrapper_module('LessEqual', torch.le)
NotEqual = create_wrapper_module('NotEqual', torch.ne)
Equal = create_wrapper_module('Equal', torch.eq)
Bmm = create_wrapper_module('Bmm', torch.bmm)
CumSum = create_wrapper_module('CumSum', torch.cumsum)
MaskedFill = create_wrapper_module('MaskedFill', torch.Tensor.masked_fill_)
Mean = create_wrapper_module('Mean', torch.mean)
Sum = create_wrapper_module('Sum', torch.sum)
Prod = create_wrapper_module('Prod', torch.prod)
Log = create_wrapper_module('Log', torch.log)
Abs = create_wrapper_module('Abs', torch.abs)
Neg = create_wrapper_module('Neg', torch.neg)


# modules for functional operations defined under torch.nn.functional package
Interpolate = create_wrapper_module('Interpolate', torch.nn.functional.interpolate)
MaxPool2d = create_wrapper_module('MaxPool2d', torch.nn.functional.max_pool2d)
AdaptiveAvgPool2d = create_wrapper_module('AdaptiveAvgPool2d', torch.nn.functional.adaptive_avg_pool2d)
AvgPool2d = create_wrapper_module('AvgPool2d', torch.nn.functional.avg_pool2d)
BatchNorm = create_wrapper_module('BatchNorm', torch.nn.functional.batch_norm)
GroupNorm = create_wrapper_module('GroupNorm', torch.nn.functional.group_norm)


# following modules are for overloaded operators like + and *,
# which can operate other than torch.Tensor datatype.
class Add(torch.nn.Module):
    """ Add module for a functional add"""
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: Any, y: Any) -> Any:
        """
        Forward-pass routine for add op
        """
        return x + y


class Multiply(torch.nn.Module):
    """ Multiply module for a functional multiply"""
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: Any, y: Any) -> Any:
        """
        Forward-pass routine for multiply op
        """
        return x * y


# modules for functional requiring special handling
class Concat(torch.nn.Module):
    """ Concat module for a functional concat"""
    def __init__(self, axis: int = 0):
        super(Concat, self).__init__()
        self._axis = axis

    # pylint:disable=arguments-differ
    def forward(self, *x) -> torch.Tensor:
        """
        Forward-pass routine for cat op
        """
        return torch.cat(x, dim=self._axis)


class DynamicConv2d(torch.nn.Module):
    """ Conv2d module for a functional conv2d"""
    def __init__(self, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        """
        Forward-pass routine for conv2d op
        """
        return torch.nn.functional.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class Pow(torch.nn.Module):
    """ Pow module for a functional pow """
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: Any, y: Any) -> Any:
        """
        Forward-pass routine for Pow op
        """
        return x ** y


class CustomSiLU(torch.nn.Module):
    """ SiLU as Sigmoid + mul """
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.mul = Multiply()

    def forward(self, x: torch.Tensor) -> Any:
        """
        Forward-pass routine for custom SiLU
        """
        return self.mul(x, self.sigmoid(x))


class Baddbmm(torch.nn.Module):
    """Custom module for a functional baddbmm"""
    @staticmethod
    def forward(*args) -> torch.Tensor:
        """
        Forward-pass routine for torch.baddbmm
        """
        tensor, batch1, batch2, beta, alpha = args
        return tensor.baddbmm(batch1, batch2, beta=beta, alpha=alpha)


class Addmm(torch.nn.Module):
    """Custom module for a functional baddbmm"""
    @staticmethod
    def forward(*args) -> torch.Tensor:
        """
        Forward-pass routine for torch.baddbmm
        """
        tensor, mat1, mat2, beta, alpha = args
        return tensor.addmm(mat1, mat2, beta=beta, alpha=alpha)
