//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  3. Neither the name of the copyright holder nor the names of its contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//  SPDX-License-Identifier: BSD-3-Clause
//
//  @@-COPYRIGHT-END-@@
//
//==============================================================================


#ifndef QUANTIZATION_UTILS_H_
#define QUANTIZATION_UTILS_H_

#include <math.h>
#include <stdint.h>

#include "DlQuantization/Quantization.hpp"

namespace DlQuantization
{

static constexpr double EPSILON = 1e-5;

TfEncoding getComputedEncodings(uint8_t bw, double min, double max, bool useSymmetricEncodings, bool useStrictSymmetric,
                                bool useUnsignedSymmetric);

// ensures min - max is not too close, by checking that max - min > epsilon
void gateMinMax(double& encodingMin, double& encodingMax);

void computeMinMaxRangeFromDeltaOffset(uint8_t bw, TfEncoding& encoding, bool useSymmetricEncodings, bool useUnsignedSymmetric,
                                       bool useStrictSymmetric);

void computeDeltaAndOffsetFromMinMax(uint8_t bw, TfEncoding& encoding, bool useSymmetricEncodings, bool useUnsignedSymmetric,
                                     bool useStrictSymmetric);

// Function to slice a tensor along an axis, allocate and populate output buffers. Output shape will be the same for each slice.
template <typename DTYPE>
void slice(const DTYPE* data, const std::vector<uint32_t>& inputShape, int32_t axis, std::vector<std::vector<DTYPE>>& output, std::vector<uint32_t>& splitShape);

// Function to concatenate from slice along an axis. Should be the same shape as the original input shape to slice.
template<typename DTYPE>
void concat(const std::vector<std::vector<DTYPE>>& data, const std::vector<uint32_t>& inputShape, int32_t axis, DTYPE* output, std::vector<uint32_t>& outputShape);

}   // End of namespace DlQuantization

#endif   // QUANTIZATION_UTILS_H_