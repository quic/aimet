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

""" utils associated with transformer quantization handling """
# current implementation sets mask to -6 by default.
# user can register for override on mask add op in an attention head.
MASK_OVERRIDE_VALUE = -6

# default attention types supported
# {attention block type name : mask_op_name}
SUPPORTED_ATTENTION_MASK_OVERRIDE_DICT = {'BertSelfAttention': 'mask_add'}


def register_attention_mask_override(attention_type_name: str = None,
                                     mask_op_name: str = None):

    """
    Registers attention type and op within it to be clamped
    :param attention_type_name: Attention type name, as string
    :param mask_op_name: Mask op identifier within attention head, as string
    :return:
    """
    if attention_type_name is not None and mask_op_name is not None:
        SUPPORTED_ATTENTION_MASK_OVERRIDE_DICT[attention_type_name] = mask_op_name


def get_supported_attention_types():
    """
    returns dictionary of supported attention types with corresponding mask op name
    :return:
    """
    return SUPPORTED_ATTENTION_MASK_OVERRIDE_DICT
