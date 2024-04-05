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


from aimet_torch.v2.nn.function_selector import FunctionSelector
import pytest


class Lib:

    def true_pred(self, *args, **kwargs):
        return True

    def false_pred(self, *args, **kwargs):
        return False

    def kernel_1(self, *args, **kwargs):
        pass

    def kernel_2(self, *args, **kwargs):
        pass

    def get_kernel(self, op_key):
        if op_key == "kernel":
            return [(self.true_pred, self.kernel_1), (self.true_pred, self.kernel_2)]
        else:
            return []

class OtherLib(Lib):

    def get_kernel(self, op_key):
        if op_key == "kernel":
            return [(self.false_pred, self.kernel_1), (self.true_pred, self.kernel_2)]
        if op_key == "unknown":
            return [( self.true_pred, self.kernel_1)]

class TestFunctionSelector:

    def test_kernel_retrieval(self):

        lib = Lib()
        selector = FunctionSelector(lib, strict=True)
        assert selector.get_impl("kernel") == lib.kernel_1
        lib_2 = Lib()
        selector.set_libraries(lib_2, strict=True)
        assert selector.get_impl("kernel") == lib_2.kernel_1
        with pytest.raises(RuntimeError):
            selector.get_impl("unknown")

        selector.set_libraries([lib, lib_2], strict=False)
        assert selector.get_impl("kernel") == lib.kernel_1
        assert selector.get_impl("unknown") is None

        lib_3 = OtherLib()
        selector.set_libraries([lib_3, lib, lib_2], strict=True)
        assert selector.get_impl("kernel") == lib_3.kernel_2
        assert selector.get_impl("unknown") == lib_3.kernel_1
