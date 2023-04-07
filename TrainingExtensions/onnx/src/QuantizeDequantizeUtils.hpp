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

#ifndef AIMET_QUANTIZEDEQUANTIZEUTILS_HPP
#define AIMET_QUANTIZEDEQUANTIZEUTILS_HPP

#include "OnnxOpUtils.h"
#include "DlQuantization/TensorQuantizer.h"
#include <cstdint>
#include <stdexcept>
#include <vector>


template <typename DTYPE>
void quantizeDequantizePerChannelGPU(const DTYPE* inTensor, DTYPE* outTensor, std::vector<int64_t>& dims, int axis,
                                     std::vector<DlQuantization::TfEncoding*>& encodings,
                                     DlQuantization::IAllocator* allocator);

template <typename DTYPE>
void quantizeDequantizePerChannelCPU(const DTYPE* inTensor, DTYPE* outTensor, std::vector<int64_t>& dims, int axis,
                                     std::vector<DlQuantization::TfEncoding*>& encodings);

template <typename T>
void sliceTensorChannelGPU(const T* inTensor, T* outTensor, long iters, long copy_width, long input_stride,
                           long output_stride, long input_offset, long output_offset);


template <typename T>
void sliceTensorAlongAxis(const T* inTensor, std::vector<int64_t>& dims, size_t axis, size_t channel, T* outTensor,
                          bool useCuda)
{
    uint64_t copy_width = 1;
    uint64_t iter       = 1;
    for (int i = 0; i < dims.size(); i++)
    {
        if (i < axis)
        {
            iter *= dims[i];
        }
        else if (i > axis)
        {
            copy_width *= dims[i];
        }
    }
    uint64_t incr          = copy_width * dims[axis];
    uint64_t input_offset  = copy_width * channel;
    uint64_t output_stride = copy_width;
    if (useCuda)
    {
#ifdef ONNX_CUDA
        sliceTensorChannelGPU(inTensor, outTensor, iter, copy_width, incr, output_stride, input_offset, 0);
#else
        throw std::runtime_error("Not compiled for GPU mode.");
#endif
    }
    else
    {
        for (long i = 0; i < iter; i++)
        {
            std::copy(inTensor + input_offset, inTensor + input_offset + copy_width, outTensor + i * copy_width);
            input_offset += incr;
        }
    }
}


template <typename T>
void quantizeDequantizePerChannel(const T* inTensor, std::vector<int64_t>& shape, int axis, T* outTensor,
                                  std::vector<DlQuantization::TfEncoding*>& encodings,
                                  std::vector<DlQuantization::TensorQuantizer*>& tensorQuantizers, bool useCuda,
                                  DlQuantization::IAllocator* allocator)
{
    if (useCuda)
    {
#ifdef ONNX_CUDA
        quantizeDequantizePerChannelGPU(inTensor, outTensor, shape, axis, encodings, allocator);
#else
        throw std::runtime_error("Not compiled for GPU mode.");
#endif
    }
    else
    {
        quantizeDequantizePerChannelCPU(inTensor, outTensor, shape, axis, encodings);
    }
}


#endif   // AIMET_QUANTIZEDEQUANTIZEUTILS_HPP
