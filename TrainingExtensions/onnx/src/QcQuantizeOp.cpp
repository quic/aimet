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

#include "QcQuantizeOp.h"
#include "AimetOpUtils.h"


#include <vector>


#ifdef ONNX_CUDA
static OnnxCudaAllocator cudaAllocator;
#endif
static OnnxCpuAllocator cpuAllocator;


QcQuantizeKernel::QcQuantizeKernel(const OrtApi* api, const OrtKernelInfo* info, bool useCuda) :
    api_(*api), info_(info), useCuda(useCuda)
{
    tensorQuantizationSim = DlQuantization::getTensorQuantizationSim<float>();
    quantInfo =
        reinterpret_cast<struct QcQuantizeInfo*>(api_.KernelInfoGetAttribute<std::int64_t>(info_, "quant_info"));
}


void QcQuantizeKernel::Compute(OrtKernelContext* context)
{
    // Setup inputs
    const OrtValue* input = api_.KernelContext_GetInput(context, 0);
    auto inputData       = api_.GetTensorData<float>(input);
    OrtTensorDimensions dimensions(api_, input);
    // Setup outputs
    OrtValue* output = api_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    auto result      = api_.GetTensorMutableData<float>(output);
    OrtTensorTypeAndShapeInfo* outputInfo = api_.GetTensorTypeAndShape(output);
    size_t size                            = api_.GetTensorShapeElementCount(outputInfo);

    std::vector<DlQuantization::TfEncoding*> encodings = quantInfo->encoding;

    DlQuantization::TensorQuantizerOpMode opMode = quantInfo->opMode;
    // Disable unused quantizers
    if (!quantInfo->enabled)
    {
        opMode = DlQuantization::TensorQuantizerOpMode::passThrough;
    }

    api_.ReleaseTensorTypeAndShapeInfo(outputInfo);

    DlQuantization::IAllocator* allocator = &cpuAllocator;
#ifdef ONNX_CUDA
    if (useCuda)
    {
        allocator = &cudaAllocator;
        cudaDeviceSynchronize();
    }
#endif

    if (quantInfo->isIntDataType)
    {
        if (quantInfo->usePerChannelMode)
        {
            int axis = quantInfo->channelAxis;
            modeSpecificActionPerChannelInt(inputData, size, result, axis, dimensions, quantInfo->tensorQuantizerRef,
                                            opMode, encodings, quantInfo->useSymmetricEncoding, allocator, useCuda,
                                            tensorQuantizationSim);
        }
        else
        {
            modeSpecificActionInt(inputData, size, result, quantInfo->tensorQuantizerRef[0], opMode, encodings[0],
                                  quantInfo->useSymmetricEncoding, allocator, useCuda);
        }
    }
    else
    {
        modeSpecificActionFloat(inputData, size, result, opMode, allocator, useCuda);
    }
}


void* QcQuantizeOp::CreateKernel(const OrtApi& api, const OrtKernelInfo* info)
{
    return new QcQuantizeKernel(&api, info, false);
};


const char* QcQuantizeOp::GetName()
{
    return "QcQuantizeOp";
};


size_t QcQuantizeOp::GetInputTypeCount()
{
    return 1;
};


ONNXTensorElementDataType QcQuantizeOp::GetInputType(size_t /*index*/)
{
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};


size_t QcQuantizeOp::GetOutputTypeCount()
{
    return 1;
};


ONNXTensorElementDataType QcQuantizeOp::GetOutputType(size_t /*index*/)
{
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

const char* QcQuantizeOp::GetExecutionProviderType() const
{
    return "CPUExecutionProvider";
};

#ifdef ONNX_CUDA
void* QcQuantizeOpGPU::CreateKernel(const OrtApi& api, const OrtKernelInfo* info)
{
    return new QcQuantizeKernel(&api, info, true);
};


const char* QcQuantizeOpGPU::GetName()
{
    return "QcQuantizeOp";
};


size_t QcQuantizeOpGPU::GetInputTypeCount()
{
    return 1;
};


ONNXTensorElementDataType QcQuantizeOpGPU::GetInputType(size_t /*index*/)
{
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};


size_t QcQuantizeOpGPU::GetOutputTypeCount()
{
    return 1;
};


ONNXTensorElementDataType QcQuantizeOpGPU::GetOutputType(size_t /*index*/)
{
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

const char* QcQuantizeOpGPU::GetExecutionProviderType() const
{
    return "CUDAExecutionProvider";
};
#endif
