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
static OnnxCudaAllocator _allocator;
#endif
static OnnxCpuAllocator cpu_allocator;


QcQuantizeKernel::QcQuantizeKernel(const OrtApi* api, const OrtKernelInfo* info, bool useCuda) :
    api_(*api), info_(info), useCuda(useCuda)
{
    tensorQuantizationSim = DlQuantization::getTensorQuantizationSim<float>();
    quant_info =
        reinterpret_cast<struct QcQuantizeInfo*>(api_.KernelInfoGetAttribute<std::int64_t>(info_, "quant_info"));
}


void QcQuantizeKernel::Compute(OrtKernelContext* context)
{
    // Setup inputs
    const OrtValue* input = api_.KernelContext_GetInput(context, 0);
    auto input_data       = api_.GetTensorData<float>(input);
    OrtTensorDimensions dimensions(api_, input);
    // Setup outputs
    OrtValue* output = api_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    auto result      = api_.GetTensorMutableData<float>(output);
    OrtTensorTypeAndShapeInfo* output_info = api_.GetTensorTypeAndShape(output);
    size_t size                            = api_.GetTensorShapeElementCount(output_info);

    std::vector<DlQuantization::TfEncoding*> encodings = quant_info->encoding;

    DlQuantization::TensorQuantizerOpMode op_mode = quant_info->opMode;
    // Disable unused quantizers
    if (!quant_info->enabled)
    {
        op_mode = DlQuantization::TensorQuantizerOpMode::passThrough;
    }

    api_.ReleaseTensorTypeAndShapeInfo(output_info);

    DlQuantization::IAllocator* allocator = &cpu_allocator;
#ifdef ONNX_CUDA
    if (useCuda)
    {
        allocator = &_allocator;
        cudaDeviceSynchronize();
    }
#endif

    if (quant_info->isIntDataType)
    {
        if (quant_info->usePerChannelMode)
        {
            int axis = quant_info->channelAxis;
            modeSpecificActionPerChannelInt(input_data, size, result, axis, dimensions, quant_info->tensorQuantizerRef,
                                            op_mode, encodings, quant_info->useSymmetricEncoding, allocator, useCuda,
                                            tensorQuantizationSim);
        }
        else
        {
            modeSpecificActionInt(input_data, size, result, quant_info->tensorQuantizerRef[0], op_mode, encodings[0],
                                  quant_info->useSymmetricEncoding, allocator, useCuda);
        }
    }
    else
    {
        modeSpecificActionFloat(input_data, size, result, op_mode, allocator, useCuda);
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
