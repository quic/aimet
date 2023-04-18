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

#ifndef AIMET_MAIN_AIMETOPUTILS_H
#define AIMET_MAIN_AIMETOPUTILS_H

#include "OnnxOpUtils.h"
#include "QuantizeDequantizeUtils.hpp"
#include "DlQuantization/TensorQuantizer.h"
#include "DlQuantization/TensorQuantizerOpFacade.h"
#include "DlQuantization/Quantization.hpp"
#include "DlQuantization/Fp16Quantization.hpp"
#include "Eigen/Core"
#include "Eigen/src/Core/arch/CUDA/Half.h"

#include <cstdint>
#include <stdexcept>
#ifdef ONNX_CUDA
#include <cuda_runtime_api.h>
#endif

#ifdef ONNX_CUDA
class OnnxCudaAllocator : public DlQuantization::IAllocator
{
public:
    void* allocateRaw(size_t bytes) override
    {
        void* ptr;
        cudaMalloc(&ptr, bytes);
        return ptr;
    }

    void deleteRaw(void* ptr) override
    {
        cudaFree(ptr);
    }
};
#endif

class OnnxCpuAllocator : public DlQuantization::IAllocator
{
public:
    void* allocateRaw(size_t bytes) override
    {
        void* ptr;
        ptr = malloc(bytes);
        return ptr;
    }

    void deleteRaw(void* ptr) override
    {
        free(ptr);
    }
};

template <typename T>
void copyInputTensorsToOutputTensors(const T* inTensor, size_t count, T* outTensor, bool useCuda);

void quantizeDequantizeFp16Cpu(const float* in, int cnt, float* out);


template <typename T>
void modeSpecificActionInt(const T* inTensor, size_t count, T* outTensor,
                           DlQuantization::TensorQuantizer* tensorQuantizer,
                           const DlQuantization::TensorQuantizerOpMode opMode, DlQuantization::TfEncoding* encoding,
                           const bool useSymmetricEncoding, DlQuantization::IAllocator* allocator, bool useCuda)
{

    switch (opMode)
    {
    case DlQuantization::TensorQuantizerOpMode::oneShotQuantizeDequantize:
    {
        tensorQuantizer->resetEncodingStats();
        tensorQuantizer->updateStats(inTensor, count, useCuda, allocator);
        DlQuantization::TfEncoding initial_encoding =
            tensorQuantizer->computeEncoding(encoding->bw, useSymmetricEncoding);
        tensorQuantizer->quantizeDequantize(inTensor, count, outTensor, initial_encoding.min, initial_encoding.max,
                                            encoding->bw, useCuda);
        // Update encoding object with computed encoding
        encoding->min    = initial_encoding.min;
        encoding->max    = initial_encoding.max;
        encoding->offset = initial_encoding.offset;
        encoding->delta  = initial_encoding.delta;
        break;
    }
    case DlQuantization::TensorQuantizerOpMode::updateStats:
    {
        tensorQuantizer->updateStats(inTensor, count, useCuda, allocator);
        copyInputTensorsToOutputTensors(inTensor, count, outTensor, useCuda);
        break;
    }
    case DlQuantization::TensorQuantizerOpMode::quantizeDequantize:
    {
        tensorQuantizer->quantizeDequantize(inTensor, count, outTensor, encoding->min, encoding->max, encoding->bw,
                                            useCuda);
        break;
    }
    case DlQuantization::TensorQuantizerOpMode::passThrough:
    {
        copyInputTensorsToOutputTensors(inTensor, count, outTensor, useCuda);
        break;
    }
    default:
    {
        throw std::exception();
    }
    }
}


template <typename T>
void modeSpecificActionPerChannelInt(const T* inTensor, size_t count, T* outTensor, int axis, OrtTensorDimensions& dims,
                                     std::vector<DlQuantization::TensorQuantizer*>& tensorQuantizers,
                                     const DlQuantization::TensorQuantizerOpMode opMode,
                                     std::vector<DlQuantization::TfEncoding*>& encodings,
                                     const bool useSymmetricEncoding, DlQuantization::IAllocator* allocator,
                                     bool useCuda)
{
    size_t num_channels = dims[axis];
    size_t channel_size = count / num_channels;
    T* channel_buffer;

    if (num_channels != encodings.size())
    {
        throw std::runtime_error(std::string("Channel dimensions do not match encoding vector size."));
    }

    switch (opMode)
    {
    case DlQuantization::TensorQuantizerOpMode::oneShotQuantizeDequantize:
    {
        channel_buffer = (T*) allocator->allocateRaw(sizeof(T) * channel_size);
        for (int ch = 0; ch < num_channels; ch++)
        {
            auto tensorQuantizer = tensorQuantizers[ch];

            sliceTensorAlongAxis(inTensor, dims, axis, ch, channel_buffer, useCuda);
            tensorQuantizer->resetEncodingStats();
            tensorQuantizer->updateStats(channel_buffer, channel_size, useCuda, allocator);
            DlQuantization::TfEncoding initial_encoding =
                tensorQuantizer->computeEncoding(encodings[ch]->bw, useSymmetricEncoding);
            encodings[ch]->min    = initial_encoding.min;
            encodings[ch]->max    = initial_encoding.max;
            encodings[ch]->offset = initial_encoding.offset;
            encodings[ch]->delta  = initial_encoding.delta;
        }
        quantizeDequantizePerChannel(inTensor, dims, axis, outTensor, encodings, tensorQuantizers, useCuda, allocator);
        allocator->deleteRaw(channel_buffer);
        break;
    }
    case DlQuantization::TensorQuantizerOpMode::updateStats:
    {
        channel_buffer = (T*) allocator->allocateRaw(sizeof(T) * channel_size);
        for (int ch = 0; ch < num_channels; ch++)
        {
            auto tensorQuantizer = tensorQuantizers[ch];
            sliceTensorAlongAxis(inTensor, dims, axis, ch, channel_buffer, useCuda);
            tensorQuantizer->updateStats(channel_buffer, channel_size, useCuda, allocator);
        }
        allocator->deleteRaw(channel_buffer);
        copyInputTensorsToOutputTensors(inTensor, count, outTensor, useCuda);
        break;
    }
    case DlQuantization::TensorQuantizerOpMode::quantizeDequantize:
    {
        quantizeDequantizePerChannel(inTensor, dims, axis, outTensor, encodings, tensorQuantizers, useCuda, allocator);
        break;
    }
    case DlQuantization::TensorQuantizerOpMode::passThrough:
    {
        copyInputTensorsToOutputTensors(inTensor, count, outTensor, useCuda);
        break;
    }
    default:
    {
        throw std::exception();
    }
    }
}

template <typename T>
void modeSpecificActionFloat(const T* inTensor, size_t count, T* outTensor,
                           const DlQuantization::TensorQuantizerOpMode opMode,
                           DlQuantization::IAllocator* allocator, bool useCuda)
{
    switch (opMode)
    {
    case DlQuantization::TensorQuantizerOpMode::oneShotQuantizeDequantize:
    case DlQuantization::TensorQuantizerOpMode::quantizeDequantize:
    {
        if(useCuda)
        {
           DlQuantization::quantizeDequantizeFp16Gpu(inTensor, count, outTensor);
        }
        else
            quantizeDequantizeFp16Cpu(inTensor, count, outTensor);
        break;
    }
    case DlQuantization::TensorQuantizerOpMode::updateStats:
    case DlQuantization::TensorQuantizerOpMode::passThrough:
    {
        copyInputTensorsToOutputTensors(inTensor, count, outTensor, useCuda);
        break;
    }
    default:
    {
        throw std::exception();
    }
    }
}

#endif   // AIMET_MAIN_AIMETOPUTILS_H
