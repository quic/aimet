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

#include "cuda_fp16.h"
#include "cuda_util.hpp"
#include "trim_functions.cuh"
#include "trim_functions.hpp"

namespace DlQuantization
{
template <typename DTYPE>
__global__ void quantizeDequantizeKernel(const DTYPE* in, int cnt, DTYPE* out,
                                         DTYPE encoding_min, DTYPE encoding_max,
                                         DTYPE encoding_delta, DTYPE encoding_offset,
                                         RoundingMode rounding_mode)
{
    CUDA_KERNEL_LOOP(i, cnt)
    {
        quantizeToFxpDevice<DTYPE>(in + i, out + i,
                                   encoding_min, encoding_max,
                                   encoding_delta, encoding_offset,
                                   rounding_mode, i);
        dequantizeFromFxpDevice<DTYPE>(out + i, encoding_delta, encoding_offset);
    }
}

template <typename DTYPE>
__global__ void quantizeToFxpKernel(const DTYPE* in, int cnt, DTYPE* out,
                                    DTYPE encoding_min, DTYPE encoding_max,
                                    DTYPE encoding_delta, DTYPE encoding_offset,
                                    RoundingMode rounding_mode, unsigned int shift)
{
    CUDA_KERNEL_LOOP(i, cnt)
    {
        quantizeToFxpDevice<DTYPE>(in + i, out + i,
                                   encoding_min, encoding_max,
                                   encoding_delta, encoding_offset,
                                   rounding_mode, i);
        *(out + i) -= shift;
    }
}

template <typename DTYPE>
__global__ void quantizeDequantizePerChannelKernel(const DTYPE* in, int numChannel, int numElement, int numElementPerChannel,
                                                   DTYPE* out, DTYPE* encodingMin, DTYPE* encodingMax, DTYPE* encodingDelta,
                                                   DTYPE* encodingOffset, RoundingMode roundingMode)
{
    CUDA_KERNEL_LOOP(i, numElement)
    {
        int channelIdx = (i / numElementPerChannel) % numChannel;
        quantizeToFxpDevice<DTYPE>(in + i, out + i,
                                   *(encodingMin + channelIdx), *(encodingMax + channelIdx),
                                   *(encodingDelta + channelIdx), *(encodingOffset + channelIdx),
                                   roundingMode, i);
        dequantizeFromFxpDevice<DTYPE>(out + i, *(encodingDelta + channelIdx), *(encodingOffset + channelIdx));
    }
}

template <typename DTYPE>
void quantizeDequantizeGpu(const DTYPE* in, int cnt, const TfEncoding& encoding, DTYPE* out, RoundingMode rounding_mode,
                           void* stream)
{
    quantizeDequantizeKernel<DTYPE>
        <<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            in, cnt, out, encoding.min, encoding.max, encoding.delta, encoding.offset, rounding_mode);
}


__global__ void quantizeDequantizeFp16Kernel(const float* in, int cnt, float* out)
{
    CUDA_KERNEL_LOOP(i, cnt)
    {
        *(out + i) = __half2float(__float2half(*(in + i)));
    }
}


void quantizeDequantizeFp16ForGPU(const float* in, int cnt, float* out, void* stream)
{
    quantizeDequantizeFp16Kernel<<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        in, cnt, out);
}


template <typename DTYPE>
void quantizeToFxpGpu(const DTYPE* in, int cnt, const TfEncoding& encoding,
                      DTYPE* out, RoundingMode rounding_mode, bool shiftToSigned)
{
    unsigned int shift = 0;
    if (shiftToSigned) {
        shift = pow(2, encoding.bw - 1);
    }
    quantizeToFxpKernel<DTYPE><<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS>>>(
            in, cnt, out, encoding.min, encoding.max, encoding.delta,
            encoding.offset, rounding_mode, shift);
}

template <typename DTYPE>
void quantizeDequantizePerChannelGpu(const DTYPE* in, int numChannel, int numElement, int numElementPerChannel,
                                     DTYPE* out, DTYPE* encodingMin, DTYPE* encodingMax, DTYPE* encodingDelta,
                                     DTYPE* encodingOffset, RoundingMode roundingMode, void* stream)
{
    quantizeDequantizePerChannelKernel<DTYPE>
        <<<CUDA_NUM_BLOCKS(numElement), CUDA_NUM_THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            in, numChannel, numElement, numElementPerChannel, out, encodingMin, encodingMax, encodingDelta,
            encodingOffset, roundingMode);
}

// Explicit instantiations
template void quantizeDequantizeGpu(const double* in, int cnt, const TfEncoding& encoding, double* out,
                                    RoundingMode rounding_mode, void* stream);

template void quantizeDequantizeGpu(const float* in, int cnt, const TfEncoding& encoding, float* out,
                                    RoundingMode rounding_mode, void* stream);

template void quantizeToFxpGpu(const double* in, int cnt, const TfEncoding& encoding, double* out,
                               RoundingMode rounding_mode, bool shiftToSigned);


template void quantizeToFxpGpu(const float* in, int cnt, const TfEncoding& encoding, float* out,
                               RoundingMode rounding_mode, bool shiftToSigned);

template void quantizeDequantizePerChannelGpu(const float* in, int numChannel, int numElement, int numElementPerChannel,
                                              float* out, float* encodingMin, float* encodingMax, float* encodingDelta,
                                              float* encodingOffset, RoundingMode roundingMode, void* stream);

template void quantizeDequantizePerChannelGpu(const double* in, int numChannel, int numElement,
                                              int numElementPerChannel, double* out, double* encodingMin,
                                              double* encodingMax, double* encodingDelta, double* encodingOffset,
                                              RoundingMode roundingMode, void* stream);

}   // End of namespace DlQuantization
