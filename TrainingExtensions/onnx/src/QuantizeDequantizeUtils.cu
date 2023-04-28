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


const int CUDA_NUM_THREADS = 512;

// Compute the number of blocks based on the total number of threads.
inline int CUDA_NUM_BLOCKS(const int N)
{
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


template <typename T>
__global__ void sliceTensorChannelKernel(const T* inTensor, T* outTensor, long iters, long copyWidth,
                                         long inputStride, long outputStride, long inputOffset, long outputOffset)
{
    /*
     * Can be used to retrieve a specific channel of the input tensor into the output tensor
     * For example: 4-D array [in_chan, out_chan, k, k], to write outTensor <- inTensor[:, n, :, :],
     * iters = in_chan; copyWidth = k * k; inputStride = out_chan * k * k; outputStride = copyWidth;
     * inputOffset = n * k * k; outputOffset = 0;
     *
     */
    int64_t idx            = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t totalElements = iters * copyWidth;
    if (idx < totalElements)
    {
        int64_t sliceNumber  = idx / copyWidth;
        int64_t slicePos     = idx % copyWidth;
        int64_t inputIdx     = inputOffset + sliceNumber * inputStride + slicePos;
        int64_t outputIdx    = outputOffset + sliceNumber * outputStride + slicePos;
        outTensor[outputIdx] = inTensor[inputIdx];
    }
}

template <typename T>
void sliceTensorChannelGPU(const T* inTensor, T* outTensor, long iters, long copyWidth, long inputStride,
                           long outputStride, long inputOffset, long outputOffset)
{
    int64_t totalThreads = iters * copyWidth;
    int64_t gridSize     = CUDA_NUM_BLOCKS(totalThreads);
    sliceTensorChannelKernel<T><<<gridSize, CUDA_NUM_THREADS>>>(inTensor, outTensor, iters, copyWidth, inputStride,
                                                                 outputStride, inputOffset, outputOffset);
}


template void sliceTensorChannelGPU(const float* inTensor, float* outTensor, long iters, long copyWidth,
                                    long inputStride, long outputStride, long inputOffset, long outputOffset);
