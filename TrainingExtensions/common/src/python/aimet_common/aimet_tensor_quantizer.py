# /usr/bin/env python3
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

""" Conditionally imports to use AIMET features using MO and python-only implementations """

# pylint: disable=unused-wildcard-import, wildcard-import, protected-access
try:
    from aimet_common.AimetTensorQuantizer import *
except ImportError as err:
    ERROR_MESSAGE = f"AimetTensorQuantizer import failed with the following error:\n\n{err}\n\n" \
                    "Please check that AimetTensorQuantizer has been built and is compatible with your " \
                    "current environment."


    class _MetaUnavailableClass(type):
        @classmethod
        def __getattr__(mcs, name):
            raise RuntimeError(f"Unable to access attribute {name} of class AimetTensorQuantizer: {ERROR_MESSAGE}")


    class AimetTensorQuantizer(metaclass=_MetaUnavailableClass):
        """
        Placeholder class for raising errors when using this class
        """
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"Unable to initialize class AimetTensorQuantizer: {ERROR_MESSAGE}")

        def __getattr__(self, name):
            raise RuntimeError(f"Unable to access attribute {name} of class AimetTensorQuantizer: {ERROR_MESSAGE}")
