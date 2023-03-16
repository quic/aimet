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


#include <DlQuantization/TensorQuantizerOpFacade.h>
#include <cstdint>
#include <stdexcept>
#ifdef ONNX_CUDA
#include <cuda_runtime_api.h>
#endif


template <typename T>
void copyInputTensorsToOutputTensors(const T* inTensor, size_t count, T* outTensor, bool useCuda);


template <typename T>
void modeSpecificActionInt(const T* inTensor, size_t count, T* outTensor,
                           DlQuantization::TensorQuantizerOpFacade* tensorQuantizer,
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
void modeSpecificActionFloat(const T* inTensor, size_t count, T* outTensor,
                           const DlQuantization::TensorQuantizerOpMode opMode,
                           DlQuantization::IAllocator* allocator, bool useCuda)
{
    switch (opMode)
    {
    case DlQuantization::TensorQuantizerOpMode::oneShotQuantizeDequantize:
    {
        copyInputTensorsToOutputTensors(inTensor, count, outTensor, useCuda);
        break;
    }
    case DlQuantization::TensorQuantizerOpMode::updateStats:
    {
        copyInputTensorsToOutputTensors(inTensor, count, outTensor, useCuda);
        break;
    }
    case DlQuantization::TensorQuantizerOpMode::quantizeDequantize:
    {
        copyInputTensorsToOutputTensors(inTensor, count, outTensor, useCuda);
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

#endif   // AIMET_MAIN_AIMETOPUTILS_H
