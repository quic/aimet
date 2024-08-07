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

#ifndef AIMET_MAIN_QCQUANTIZEOP_H
#define AIMET_MAIN_QCQUANTIZEOP_H

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "QcQuantizeInfo.h"

#ifdef ONNX_CUDA
// IMPORTANT: cuda_context.h needs to be included before onnxruntime_lite_custom_op.h
#include "core/providers/cuda/cuda_context.h"
#include <cuda_runtime_api.h>
#endif

#include "onnxruntime_lite_custom_op.h"


struct QcQuantizeOp
{
    QcQuantizeOp(const OrtApi* api, const OrtKernelInfo* info);

    void computeImpl(const Ort::Custom::Tensor<float>& input, Ort::Custom::Tensor<float>& output, void* stream,
                     bool useCuda, DlQuantization::IAllocator* allocator);

protected:
    struct QcQuantizeInfo* quantInfo;

private:
    std::unique_ptr<DlQuantization::ITensorQuantizationSim<float>> tensorQuantizationSim;
    const OrtKernelInfo* info_;
    OrtApi api_;
};


void RegisterOps(Ort::CustomOpDomain& domain);

#endif   // AIMET_MAIN_QCQUANTIZEOP_H
