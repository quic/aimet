//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

#include "QuantizeDequantizeUtils.hpp"

template <typename DTYPE>
void quantizeDequantizePerChannelCPU(const DTYPE* inTensor, DTYPE* outTensor, std::vector<int64_t>& dims, int axis,
                                     std::vector<DlQuantization::TfEncoding*>& encodings)
{
    int64_t channels = dims[axis];
    int64_t num_el   = 1;
    for (long dim: dims)
    {
        num_el *= dim;
    }

    int64_t innerDims = 1;
    for (int i = 0; i < dims.size(); i++)
    {
        if (i > axis)
        {
            innerDims *= dims[i];
        }
    }

    for (long idx = 0; idx < num_el; idx++)
    {
        int64_t chanIdx = (idx / innerDims) % channels;
        DTYPE delta = encodings[chanIdx]->delta;

        // Saturate
        DTYPE out = fmax(fmin(inTensor[idx], encodings[chanIdx]->max), encodings[chanIdx]->min);
        // Scale
        out = out / delta;
        // Round
        out = roundf(out) * delta;
        outTensor[idx] = out;
    }
}

template void quantizeDequantizePerChannelCPU(const float* inTensor, float* outTensor, std::vector<int64_t>& dims,
                                              int axis, std::vector<DlQuantization::TfEncoding*>& encodings);
