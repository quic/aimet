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

""" Modules for functional elementwise ops """
import torch
import torch.nn


class Add(torch.nn.Module):
    """ Add module for a functional add"""

    # pylint:disable=arguments-differ
    def forward(self, x, y):
        """
        Forward-pass routine for add op
        """
        return x + y


class Subtract(torch.nn.Module):
    """ Subtract module for a functional subtract"""

    # pylint:disable=arguments-differ
    def forward(self, x, y):
        """
        Forward-pass routine for subtact op
        """
        return x - y


class Multiply(torch.nn.Module):
    """ Multiply module for a functional multiply"""

    # pylint:disable=arguments-differ
    def forward(self, x, y):
        """
        Forward-pass routine for multiply op
        """
        return x * y


class Divide(torch.nn.Module):
    """ Divide module for a functional divide"""

    # pylint:disable=arguments-differ
    def forward(self, x, y):
        """
        Forward-pass routine for divide op
        """
        return torch.div(x, y)


class Concat(torch.nn.Module):
    """ Concat module for a functional concat"""
    def __init__(self, axis: int = 0):
        super(Concat, self).__init__()
        self.axis = axis

    # pylint:disable=arguments-differ
    def forward(self, x, y):
        """
        Forward-pass routine for divide op
        """
        return torch.cat((x, y), self.axis)
