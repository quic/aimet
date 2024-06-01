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

#include "onnx_custom_add.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

static const char* c_OpDomain = "my_ops";

using namespace Ort;

struct custom_add_kernel
{
    custom_add_kernel(const OrtApi& api, const OrtKernelInfo* info)
    {
    }

    void Compute(OrtKernelContext* context)
    {
        Ort::KernelContext ctx(context);
        Ort::ConstValue input_X = ctx.GetInput(0);
        Ort::ConstValue input_Y = ctx.GetInput(1);
        const float* X          = input_X.GetTensorData<float>();
        const float* Y          = input_Y.GetTensorData<float>();

        std::vector<int64_t> dimensions = input_X.GetTensorTypeAndShapeInfo().GetShape();
        Ort::UnownedValue output        = ctx.GetOutput(0, dimensions);
        float* out                      = output.GetTensorMutableData<float>();

        const size_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();

        for (size_t i = 0; i < size; i++)
        {
            out[i] = X[i] + Y[i];
        }
    }
};

struct custom_add : Ort::CustomOpBase<custom_add, custom_add_kernel>
{
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return std::make_unique<custom_add_kernel>(api, info).release();
    };

    const char* GetName() const
    {
        return "custom_add";
    };

    const char* GetExecutionProviderType() const
    {
        return "CPUExecutionProvider";
    };

    size_t GetInputTypeCount() const
    {
        return 2;
    };

    size_t GetOutputTypeCount() const
    {
        return 1;
    };

    ONNXTensorElementDataType GetInputType(size_t index) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    };

    ONNXTensorElementDataType GetOutputType(size_t index) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    };
};

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api)
{
    Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);
    OrtStatus* result       = nullptr;

    try
    {
        static custom_add c_CustomOpOne;
        static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
        Ort::CustomOpDomain domain {c_OpDomain};
        domain.Add(&c_CustomOpOne);

        Ort::UnownedSessionOptions session_options(options);
        session_options.Add(domain);
        ort_custom_op_domain_container.push_back(std::move(domain));
    }
    catch (const std::exception& e)
    {
        Ort::Status status {e};
        result = status.release();
    }
    return result;
}