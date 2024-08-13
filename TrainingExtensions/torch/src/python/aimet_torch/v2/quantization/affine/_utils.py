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
# pylint: disable=missing-module-docstring
from typing import Union
import math


class _GridMixin:
    """
    Collection of helper functions to manage quantzation grid parameters.
    """
    qmin: int
    qmax: int

    def _get_num_steps(self) -> int:
        return self.qmax - self.qmin

    def _set_num_steps(self, num_steps):
        self.qmin -= math.ceil(num_steps / 2)
        self.qmax += math.floor(num_steps / 2)

    def _get_centroid(self) -> int:
        return math.ceil((self.qmax + self.qmin) / 2)

    def _set_centroid(self, centroid: int):
        num_steps = self._get_num_steps()
        self.qmin = centroid - math.ceil(num_steps / 2)
        self.qmax = centroid + math.floor(num_steps / 2)

    def _get_bitwidth(self) -> Union[int, float]:
        r"""
        Returns bitwidth of the quantizer.

        Bitwidth is a hardware concept which is defined as below

        .. math::
            bitwidth=
            \begin{cases}
                B,         & \text{if}\quad \exists_{B \geq 1} \quad (qmin=0, qmax=2^B-1)
                                            \text{ or } (qmin=-2^{B-1}, qmax=2^{B-1}-1)\\
                undefined, & \text{otherwise}
            \end{cases}

        where :math:`qmin` and :math:`qmax` denotes the minimum and maximum values of the quantization grid.

        If bitwidth can't be defined as such, this function will throw a runtime error
        """
        if self.qmin + self.qmax == -1 or self.qmin == 0:
            bitwidth = math.log2(self.qmax - self.qmin + 1)
            if bitwidth == int(bitwidth):
                bitwidth = int(bitwidth)
            return bitwidth

        msg = self._invalid_bitwidth_error_msg()
        raise RuntimeError(msg)

    def _set_bitwidth(self, bitwidth: int):
        if bitwidth == int(bitwidth):
            bitwidth = int(bitwidth)

        if not isinstance(bitwidth, int):
            clsname = type(self).__qualname__
            msg = "Setting bitwidth to a non-integer value {.3f:bitwidth} is not supported. "\
                  "To modify quantization grid with finer granularity, "\
                 f"please consider setting {clsname}.qmin and .qmax to the desired values."
            raise TypeError(msg)

        if bitwidth < 1:
            raise ValueError(f"Bitwidth can't be smaller than 1. Got {bitwidth}")

        if self.qmin + self.qmax == -1:
            # signed
            self.qmin = -2**(bitwidth - 1)
            self.qmax = 2**(bitwidth - 1) - 1
        elif self.qmin == 0:
            # unsigned
            self.qmin =  0
            self.qmax = 2**bitwidth - 1
        else:
            clsname = type(self).__qualname__
            msg = ' '.join([
                self._invalid_bitwidth_error_msg(),
                 "To modify quantization grid with finer granularity, "\
                f"please consider setting {clsname}.qmin and .qmax to the desired values."
            ])
            raise RuntimeError(msg)

    def _invalid_bitwidth_error_msg(self):
        clsname = type(self).__qualname__
        qmin = self.qmin
        qmax = self.qmax
        return f"{clsname}.bitwidth is undefined in the quantization grid [{qmin}, {qmax}]. "\
               f"Bitwidth can be defined if and only if there exists some non-negative number `B` such that "\
               f"the quantization grid can be expressed in the form of [-2**(B-1), 2**(B-1) - 1] or [0, 2**B - 1]."
