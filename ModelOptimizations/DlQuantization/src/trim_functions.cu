//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2016-2017, Qualcomm Innovation Center, Inc. All rights reserved.
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

#include <stdexcept>

#include "cuda_util.hpp"
#include "trim_functions.cuh"
#include "trim_functions.hpp"

namespace DlQuantization
{
template <typename DTYPE>
__global__ void QuantizeDequantize_kernel(const DTYPE* in, int cnt, const TfEncoding encoding, DTYPE* out,
                                     RoundingMode rounding_mode)
{
    CUDA_KERNEL_LOOP(i, cnt)
    {
        QuantizeToFxp_device<DTYPE>(in + i, i, encoding, out + i, rounding_mode);
        DequantizeFromFxp_device<DTYPE>(encoding, out + i);
    }
}

template <typename DTYPE>
__global__ void QuantizeToFxp_kernel(const DTYPE* in, int cnt, const TfEncoding encoding, DTYPE* out,
                                          RoundingMode rounding_mode)
{
    CUDA_KERNEL_LOOP(i, cnt)
    {
        QuantizeToFxp_device<DTYPE>(in + i, i, encoding, out + i, rounding_mode);
    }
}

template <typename DTYPE>
void QuantizeDequantize_GPU(const DTYPE* in, int cnt, const TfEncoding& encoding, DTYPE* out,
                            RoundingMode rounding_mode)
{
    QuantizeDequantize_kernel<<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS>>>(in, cnt, encoding, out, rounding_mode);
}

template <typename DTYPE>
void QuantizeToFxp_GPU(const DTYPE* in, int cnt, const TfEncoding& encoding, DTYPE* out, RoundingMode rounding_mode)
{
    QuantizeToFxp_kernel<<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS>>>(in, cnt, encoding, out, rounding_mode);
}

// Explicit instantiations
template void QuantizeDequantize_GPU(const double* in, int cnt, const TfEncoding& encoding, double* out,
                                     RoundingMode rounding_mode);

template void QuantizeDequantize_GPU(const float* in, int cnt, const TfEncoding& encoding, float* out,
                                     RoundingMode rounding_mode);

template void QuantizeToFxp_GPU(const double* in, int cnt, const TfEncoding& encoding, double* out,
                                RoundingMode rounding_mode);


template void QuantizeToFxp_GPU(const float* in, int cnt, const TfEncoding& encoding, float* out,
                                RoundingMode rounding_mode);
}   // End of namespace DlQuantization
