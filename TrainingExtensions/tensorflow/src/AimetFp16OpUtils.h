//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

#ifndef AIMET_FP16_OP_UTILS_H
#define AIMET_FP16_OP_UTILS_H

#include "AimetOpUtils.h"

#define EIGEN_USE_THREADS

// below forward declaration is done to remove the dependency of "cast_op_impl.h" for TF1.15
namespace tensorflow {
typedef std::function<void(OpKernelContext*, const Tensor&, Tensor*,
                           bool trunc)> CastFunctorType;

CastFunctorType GetCpuCastFromFloat(DataType dst_dtype);
CastFunctorType GetCpuCastFromHalf(DataType dst_dtype);
CastFunctorType GetGpuCastFromHalf(DataType dst_dtype);
CastFunctorType GetGpuCastFromFloat(DataType dst_dtype);
}

template <typename Device>
class QuantizeDequantizeFp16Functor
{
    /*
    class to quantize the input tensor to fp16 and dequantize it to fp32. This class has a specialized implementation
    for GPUDevice and CPUDevice. Tensorflow internal functions are called for each of them.
    fp16 type supported as part of this operation is IEEE float16.
    */
};

template <>
class QuantizeDequantizeFp16Functor <CPUDevice>
{
    // truncate, if set to true would truncate the inputs before casting to fp16. If set to true, tensorflow backend
    // calls LSBZeroSetter which does the truncate operation
    bool _truncate = false;

    public:
    void operator()(OpKernelContext* context, const Tensor& inTensor, Tensor* outTensor)
    {
        Tensor tempTensorFp16;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_HALF, inTensor.shape(), &tempTensorFp16));

        GetCpuCastFromFloat(DT_HALF)(context, inTensor, &tempTensorFp16, _truncate);
        GetCpuCastFromHalf(DT_FLOAT)(context, tempTensorFp16, outTensor, _truncate);
    }
};

#ifdef GOOGLE_CUDA
template <>
class QuantizeDequantizeFp16Functor <GPUDevice>
{
    // truncate, if set to true would truncate the inputs before casting to fp16. If set to true, tensorflow backend
    // calls LSBZeroSetter which does the truncate operation
    bool _truncate = false;

    public:
    void operator()(OpKernelContext* context, const Tensor& inTensor, Tensor* outTensor)
    {
        Tensor tempTensorFp16;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_HALF, inTensor.shape(), &tempTensorFp16));

        GetGpuCastFromFloat(DT_HALF)(context, inTensor, &tempTensorFp16, _truncate);
        GetGpuCastFromHalf(DT_FLOAT)(context, tempTensorFp16, outTensor, _truncate);
    }
};
#endif // GOOGLE_CUDA

template <typename D, typename T>
void modeSpecificActionFp16(OpKernelContext* context, const Tensor& inTensor, const uint64* tensorQuantizerRef,
                            const int32* opMode, Tensor* outTensor)
{
    auto opModeHost = copyLiteralToHost<int32>(context->eigen_device<D>(), opMode);
    auto opModeEnum = static_cast<const DlQuantization::TensorQuantizerOpMode>(opModeHost);

    switch (opModeEnum)
    {
    case DlQuantization::TensorQuantizerOpMode::oneShotQuantizeDequantize:
    {
        QuantizeDequantizeFp16Functor<D> fp16Op;
        fp16Op(context, inTensor, outTensor);
        break;
    }
    case DlQuantization::TensorQuantizerOpMode::quantizeDequantize:
    {
        QuantizeDequantizeFp16Functor<D> fp16Op;
        fp16Op(context, inTensor, outTensor);
        break;
    }
    case DlQuantization::TensorQuantizerOpMode::updateStats:
    {
        copyInputTensorsToOutputTensors(context->eigen_device<D>(), inTensor.flat<T>().data(),
                                            inTensor.NumElements(), outTensor->flat<T>().data());
        break;
    }
    case DlQuantization::TensorQuantizerOpMode::passThrough:
    {
        copyInputTensorsToOutputTensors(context->eigen_device<D>(), inTensor.flat<T>().data(),
                                            inTensor.NumElements(), outTensor->flat<T>().data());
        break;
    }
    default:
    {
        std::cout << "encountered unknown TensorQuantizerOpMode" << std::endl;
        assert(0);
    }
    }
}

#endif   // AIMET_FP16_OP_UTILS_H
