# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Constants for arch checker. """

# pylint: disable=too-few-public-methods
class ArchCheckerReportConstants:
    """ Constants for arch checker report. """
    OP_STRUCT_OP_TYPE = "OpStructure"
    DF_GRAPH_NODENAME = "Graph/Layer_name"
    DF_ISSUE = "Issue"
    DF_RECOMM = "Recommendation"

    UNDEFINED_ISSUE = "Undefined issue from check: {}"
    UNDEFINED_RECOMM = "Undefined recommendation from check: {}"

    OUTPUT_CSV_HEADER = [DF_GRAPH_NODENAME, DF_ISSUE, DF_RECOMM]

    ERR_MSG_DICT = {
        "_check_conv_channel_32_base": {DF_ISSUE: "The channel size of input/output tensor of this convolution is not a multiple of 32",
                                        DF_RECOMM: "Try adjusting the channels to multiple of 32 to get better performance."},
        "_check_conv_channel_larger_than_32":{DF_ISSUE: "The channel size of input/output tensor of this convolution is smaller than 32",
                                              DF_RECOMM: "Try adjusting the channels to multiple of 32 to get better performance."},
        "_activation_checks":{"PRelu": {DF_ISSUE:"PRelu activation function degenerates performance.",
                                        DF_RECOMM:"Try use Relu instead."},
                              "SiLU": {DF_ISSUE:"SiLU (Swish) activation function degenerates performance.",
                                       DF_RECOMM:"Try use Hard SiLU (hardswish) instaed."}},
        "_check_batch_norm_fold": {DF_ISSUE: "The batch norm layer cannot be folded to immediate conv/linear layer. Quantizing standalone BN can degenerate performance.",
                                   DF_RECOMM: "Try remove the standalone BN or move the BN adjacent to Conv."},
        "_check_intermediate_padding": {DF_ISSUE: "This convolution includes intermediate padding that degenerates performance.",
                                        DF_RECOMM: "Try move all padding to the first convolution in the sequence: [Conv -> Activation -> (Optionally) BN -> Conv]."},
        "_check_foldable_bn_with_split": {DF_ISSUE: "This structure: (conv1, conv2, ...) -> split_node(concat) -> BN degenerates performance",
                                          DF_RECOMM: "Try transform the structure so that BN can be folded to conv."}
    }
