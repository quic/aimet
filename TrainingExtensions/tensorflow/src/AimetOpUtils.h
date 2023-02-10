//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2020-2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

#ifndef AIMET_OP_UTILS_H
#define AIMET_OP_UTILS_H

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/logging.h"
#include <cmath>

#include <DlQuantization/TensorQuantizerOpFacade.h>

// Declarations of the functors to be used
// to invoke Quantization APIs.
using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename T>
void copyInputTensorsToOutputTensors(const CPUDevice& d, const T* inTensor, size_t count, T* outTensor);

template <typename T>
T copyLiteralToHost(const CPUDevice& d, const T* deviceValue);

template <typename T>
void copyArrayToHost(const CPUDevice& d, const T* srcPtr, T* destPtr, int count);

template <typename T>
void copyInputTensorsToOutputTensors(const GPUDevice& d, const T* inTensor, size_t count, T* outTensor);

template <typename T>
T copyLiteralToHost(const GPUDevice& d, const T* deviceValue);

template <typename T>
void copyArrayToHost(const GPUDevice& d, const T* srcPtr, T* destPtr, int count);

void chipAndCopyPerChannelValues(const CPUDevice& d, Tensor tensorToCopyInto,
                                 TTypes<float>::ConstMatrix tensorToCopyFrom, int channel);

void chipAndCopyPerChannelValues(const GPUDevice& d, Tensor tensorToCopyInto,
                                 TTypes<float>::ConstMatrix tensorToCopyFrom, int channel);

void sliceAndStoreTensor(const CPUDevice& d, Tensor* slicedTensor, Tensor tensorToSlice, int channel);

void sliceAndStoreTensor(const GPUDevice& d, Tensor* slicedTensor, Tensor tensorToSlice, int channel);

void quantizeDequantize(const GPUDevice& d, TTypes<float>::ConstMatrix inputs,
                        DlQuantization::TfEncoding encodings, TTypes<float>::Matrix outputs, int channel);

void quantizeDequantize(const CPUDevice& d, TTypes<float>::ConstMatrix inputs,
                        DlQuantization::TfEncoding encodings, TTypes<float>::Matrix outputs, int channel);

void quantizeDequantizePerChannel(const GPUDevice& d, TTypes<float>::ConstMatrix inputs, TTypes<float>::Matrix outputs,
                           Tensor* encodingMin, Tensor* encodingMax, Tensor* encodingScale, Tensor* encodingOffset,
                           Tensor* encodingInvScaleTensor);

void quantizeDequantizePerChannel(const CPUDevice& d, TTypes<float>::ConstMatrix inputs, TTypes<float>::Matrix outputs,
                           Tensor* encodingMin, Tensor* encodingMax, Tensor* encodingScale, Tensor* encodingOffset,
                           Tensor* encodingInvScaleTensor);

#if GOOGLE_CUDA
class TensorFlowCudaAllocator: public DlQuantization::IAllocator
{
public:
    TensorFlowCudaAllocator(Allocator* allocator): allocator_(allocator) {}

    void* allocateRaw(size_t bytes) override
    {
        return allocator_->AllocateRaw(256, bytes);
    }

    void deleteRaw(void *ptr) override
    {
        allocator_->DeallocateRaw(ptr);
    }

protected:
    Allocator* allocator_;
};
#endif // GOOGLE_CUDA

template <typename D, typename T>
void modeSpecificActionInt(const D& d, const T* inTensor, size_t count, T* outTensor,
                        const uint64* tensorQuantizerRef, const int32* opMode,
                        const double* min, const double* max, const int8* bw,
                        const bool* useSymEncoding, DlQuantization::IAllocator* allocator)
{
    bool useCuda = false;
    if (std::is_same<D, GPUDevice>::value)
    {
        useCuda = true;
    }

    // Note that all of the pointers to data here could either be pointing to CPU memory or GPU memory
    // We first copy everything to CPU memory and then use them
    auto tensorQuantizerRefHost = copyLiteralToHost<uint64>(d, tensorQuantizerRef);
    auto opModeHost = copyLiteralToHost<int32>(d, opMode);
    auto opModeEnum = static_cast<const DlQuantization::TensorQuantizerOpMode>(opModeHost);
    auto encodingMin = copyLiteralToHost<double>(d, min);
    auto encodingMax = copyLiteralToHost<double>(d, max);
    auto tensorQuantizer = reinterpret_cast<DlQuantization::TensorQuantizerOpFacade*>(tensorQuantizerRefHost);
    auto bitwidth = copyLiteralToHost<int8>(d, bw);
    auto useSymmetricEncoding = copyLiteralToHost<bool>(d, useSymEncoding);

    switch (opModeEnum)
    {
    case DlQuantization::TensorQuantizerOpMode::oneShotQuantizeDequantize:
    {
        tensorQuantizer->resetEncodingStats();
        tensorQuantizer->updateStats(inTensor, count, useCuda, allocator);
        DlQuantization::TfEncoding initial_encoding = tensorQuantizer->computeEncoding(bitwidth, useSymmetricEncoding);
        tensorQuantizer->quantizeDequantize(inTensor, count, outTensor, initial_encoding.min, initial_encoding.max,
                                            bitwidth, useCuda);

        break;
    }

    case DlQuantization::TensorQuantizerOpMode::updateStats:
    {
        tensorQuantizer->updateStats(inTensor, count, useCuda, allocator);
        copyInputTensorsToOutputTensors(d, inTensor, count, outTensor);
        break;
    }
    case DlQuantization::TensorQuantizerOpMode::quantizeDequantize:
    {
        tensorQuantizer->quantizeDequantize(inTensor, count, outTensor, encodingMin, encodingMax, bitwidth, useCuda);
        break;
    }
    case DlQuantization::TensorQuantizerOpMode::passThrough:
    {
        copyInputTensorsToOutputTensors(d, inTensor, count, outTensor);
        break;
    }
    default:
    {
        assert(0);
    }
    }
}

#endif   // AIMET_OP_UTILS_H
