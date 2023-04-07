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
__global__ void sliceTensorChannelKernel(const T* inTensor, T* outTensor, long iters, long copy_width,
                                         long input_stride, long output_stride, long input_offset, long output_offset)
{
    /*
     * Can be used to retrieve a specific channel of the input tensor into the output tensor
     * For example: 4-D array [in_chan, out_chan, k, k], to write outTensor <- inTensor[:, n, :, :],
     * iters = in_chan; copy_width = k * k; input_stride = out_chan * k * k; output_stride = copy_width;
     * input_offset = n * k * k; output_offset = 0;
     *
     */
    int64_t idx            = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = iters * copy_width;
    if (idx < total_elements)
    {
        int64_t slice_number = idx / copy_width;
        int64_t slice_pos = idx % copy_width;
        int64_t input_idx  = input_offset + slice_number * input_stride + slice_pos;
        int64_t output_idx = output_offset + slice_number * output_stride + slice_pos;
        outTensor[output_idx] = inTensor[input_idx];
    }
}

template <typename T>
void sliceTensorChannelGPU(const T* inTensor, T* outTensor, long iters, long copy_width, long input_stride,
                           long output_stride, long input_offset, long output_offset)
{
    int64_t total_threads = iters * copy_width;
    int64_t grid_size     = CUDA_NUM_BLOCKS(total_threads);
    sliceTensorChannelKernel<T><<<grid_size, CUDA_NUM_THREADS>>>(inTensor, outTensor, iters, copy_width, input_stride,
                                                                 output_stride, input_offset, output_offset);
}


template <typename DTYPE>
__global__ void quantizeDequantizePerChannelKernel(const DTYPE* inTensor, DTYPE* outTensor, DTYPE* encodingMin,
                                                   DTYPE* encodingMax, DTYPE* encodingDelta, uint num_el,
                                                   uint innerDims, uint num_channels)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_el)
    {
        uint chanIdx = (idx / innerDims) % num_channels;

        DTYPE delta = encodingDelta[chanIdx];

        // Saturate
        DTYPE out = fmaxf(fminf(inTensor[idx], encodingMax[chanIdx]), encodingMin[chanIdx]);
        // Scale
        out = out / delta;
        // Round, unscale
        out            = roundf(out) * delta;
        outTensor[idx] = out;
    }
}


template <typename DTYPE>
void quantizeDequantizePerChannelGPU(const DTYPE* inTensor, DTYPE* outTensor, std::vector<int64_t>& dims, int axis,
                                     std::vector<DlQuantization::TfEncoding*>& encodings,
                                     DlQuantization::IAllocator* allocator)
{
    int64_t channels = dims[axis];
    int num_el       = 1;
    for (long dim: dims)
    {
        num_el *= dim;
    }
    int64_t total_threads = num_el;
    int64_t grid_size     = CUDA_NUM_BLOCKS((int) total_threads);
    DTYPE enc_vec[3][channels];
    for (int i = 0; i < channels; i++)
    {
        enc_vec[0][i] = encodings[i]->min;
        enc_vec[1][i] = encodings[i]->max;
        enc_vec[2][i] = encodings[i]->delta;
    }
    auto encodingVectorDevice = (DTYPE*) allocator->allocateRaw(3 * channels * sizeof(DTYPE));
    cudaMemcpy(encodingVectorDevice, enc_vec, 3 * channels * sizeof(DTYPE), cudaMemcpyHostToDevice);
    DTYPE* encodingMin   = encodingVectorDevice;
    DTYPE* encodingMax   = encodingVectorDevice + channels;
    DTYPE* encodingDelta = encodingVectorDevice + 2 * channels;

    int innerDims = 1;
    for (int i = 0; i < dims.size(); i++)
    {
        if (i > axis)
        {
            innerDims *= dims[i];
        }
    }

    quantizeDequantizePerChannelKernel<<<grid_size, CUDA_NUM_THREADS>>>(inTensor, outTensor, encodingMin, encodingMax,
                                                                        encodingDelta, num_el, innerDims, channels);

    allocator->deleteRaw(encodingVectorDevice);
}


template void quantizeDequantizePerChannelGPU(const float* inTensor, float* outTensor, std::vector<int64_t>& dims,
                                              int axis, std::vector<DlQuantization::TfEncoding*>& encodings,
                                              DlQuantization::IAllocator* allocator);

template void sliceTensorChannelGPU(const float* inTensor, float* outTensor, long iters, long copy_width,
                                    long input_stride, long output_stride, long input_offset, long output_offset);
