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

#include "DlQuantization/TensorQuantizer.h"
#include "OnnxOpUtils.h"
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef ONNX_CUDA
#include <cuda_runtime_api.h>
#endif


template <typename T>
void sliceTensorChannelGPU(const T* inTensor, T* outTensor, long iters, long copyWidth, long inputStride,
                           long outputStride, long inputOffset, long outputOffset);


template <typename T>
void sliceTensorAlongAxis(const T* inTensor, std::vector<int64_t>& dims, size_t axis, size_t channel, T* outTensor,
                          bool useCuda)
{
    uint64_t copyWidth = 1;
    uint64_t iter      = 1;
    for (int i = 0; i < dims.size(); i++)
    {
        if (i < axis)
        {
            iter *= dims[i];
        }
        else if (i > axis)
        {
            copyWidth *= dims[i];
        }
    }
    uint64_t incr         = copyWidth * dims[axis];
    uint64_t inputOffset  = copyWidth * channel;
    uint64_t outputStride = copyWidth;
    if (useCuda)
    {
#ifdef ONNX_CUDA
        sliceTensorChannelGPU(inTensor, outTensor, iter, copyWidth, incr, outputStride, inputOffset, 0);
#else
        throw std::runtime_error("Not compiled for GPU mode.");
#endif
    }
    else
    {
        for (long i = 0; i < iter; i++)
        {
            std::copy(inTensor + inputOffset, inTensor + inputOffset + copyWidth, outTensor + i * copyWidth);
            inputOffset += incr;
        }
    }
}


template <typename T>
void quantizeDequantizePerChannel(
    const T* inTensor, std::vector<int64_t>& shape, int axis, T* outTensor,
    std::vector<DlQuantization::TfEncoding*>& encodings,
    std::vector<DlQuantization::TensorQuantizer*>& tensorQuantizers, bool useCuda,
    DlQuantization::IAllocator* allocator, void* stream,
    std::unique_ptr<DlQuantization::ITensorQuantizationSim<float> >& tensorQuantizationSim)
{
    size_t channels   = shape[axis];
    size_t numElement = 1;
    size_t innerDims  = 1;
    for (int i = 0; i < shape.size(); i++)
    {
        numElement *= shape[i];
        if (i > axis)
        {
            innerDims *= shape[i];
        }
    }

    T encVec[4][channels];
    for (int i = 0; i < channels; i++)
    {
        encVec[0][i] = encodings[i]->min;
        encVec[1][i] = encodings[i]->max;
        encVec[2][i] = encodings[i]->delta;
        encVec[3][i] = encodings[i]->offset;
    }
    T* encodingVectorDevice;
    if (useCuda)
    {
#ifdef ONNX_CUDA
        encodingVectorDevice = (T*) allocator->allocateRaw(4 * channels * sizeof(T));
        cudaMemcpy(encodingVectorDevice, encVec, 4 * channels * sizeof(T), cudaMemcpyHostToDevice);
#else
        throw std::runtime_error("Not compiled for GPU mode.");
#endif
    }
    else
    {
        encodingVectorDevice = (T*) encVec;
    }

    T* encodingMin    = encodingVectorDevice;
    T* encodingMax    = encodingVectorDevice + channels;
    T* encodingDelta  = encodingVectorDevice + 2 * channels;
    T* encodingOffset = encodingVectorDevice + 3 * channels;

    tensorQuantizationSim->quantizeDequantizeTensorPerChannel(inTensor, channels, numElement, innerDims, outTensor,
                                                              encodingMin, encodingMax, encodingDelta, encodingOffset,
                                                              tensorQuantizers[0]->roundingMode, useCuda, stream);
    if (useCuda)
    {
        allocator->deleteRaw(encodingVectorDevice);
    }
}


#endif   // AIMET_QUANTIZEDEQUANTIZEUTILS_HPP
