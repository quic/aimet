//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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

#include "QcQuantizeOp.hpp"
#include "AimetOpUtils.h"
#include "AimetFp16OpUtils.h"

#include <type_traits>

#define EIGEN_USE_THREADS
using namespace tensorflow;

REGISTER_OP("QcQuantize")
    .Input("in_tensor: T")     // list of input tensors (weights/activations)
    .Input("op_mode: int32")   //{'ANALYSIS', 'ACTIVE', 'PASSTHROUGH'}")
    .Input("tensor_quantizer_reference: int64")
    .Input("encoding_min: double")
    .Input("encoding_max: double")
    .Input("bit_width: int8")
    .Input("use_symmetric_encoding: bool")
    .Input("is_int_data_type: bool")
    .Output("out_tensor: T")   // list of output tensors (weights/activations)

    .Attr("T: {float} = DT_FLOAT")   // attr 'T' specifies which template instantiation of op to use, default float
    .Doc(R"doc(QcQuantize custom op.)doc")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });


// OpKernel definition.
// 'Device is templated on the type of device.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class QcQuantizeOp : public OpKernel
{
public:
    explicit QcQuantizeOp(OpKernelConstruction* context) : OpKernel(context)
    {
    }

    void Compute(OpKernelContext* context) override
    {
        // Consume input tensor
        const Tensor& inTensor = context->input(0);
        auto inTensorFlat      = inTensor.flat<T>().data();

        // Read the op_mode
        const Tensor* opModeTensor;
        OP_REQUIRES_OK(context, context->input("op_mode", &opModeTensor));
        const int32* opMode = opModeTensor->flat<int32>().data();

        // Read the tensor quantizer ref
        const Tensor* quantizerRefTensor;
        OP_REQUIRES_OK(context, context->input("tensor_quantizer_reference", &quantizerRefTensor));
        uint64* quantizerAddr = (uint64*) quantizerRefTensor->flat<int64>().data();

        // Read the encoding_min
        const Tensor* encodingMinTensor;
        OP_REQUIRES_OK(context, context->input("encoding_min", &encodingMinTensor));
        const double* encodingMin = encodingMinTensor->flat<double>().data();

        // Read the encoding_max
        const Tensor* encodingMaxTensor;
        OP_REQUIRES_OK(context, context->input("encoding_max", &encodingMaxTensor));
        const double* encodingMax = encodingMaxTensor->flat<double>().data();

        // read bitwidth
        const Tensor* bitwidthTensor;
        OP_REQUIRES_OK(context, context->input("bit_width", &bitwidthTensor));
        const int8* bitwidth = bitwidthTensor->flat<int8>().data();

        // use symmetric encoding
        const Tensor* useSymmetricEncodingTensor;
        OP_REQUIRES_OK(context, context->input("use_symmetric_encoding", &useSymmetricEncodingTensor));
        auto useSymmetricEncoding = useSymmetricEncodingTensor->flat<bool>().data();

        // is_int_data_type
        const Tensor* isIntDataTypeTensor;
        OP_REQUIRES_OK(context, context->input("is_int_data_type", &isIntDataTypeTensor));
        auto isIntDataTypeFlat = isIntDataTypeTensor->flat<bool>().data();
        auto isIntDataType = copyLiteralToHost<bool>(context->eigen_device<Device>(), isIntDataTypeFlat);

        // allocate output tensors
        Tensor* outTensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, inTensor.shape(), &outTensor));
        auto outTensorFlat = outTensor->flat<T>().data();

        if(isIntDataType)
        {
            DlQuantization::IAllocator* allocator = nullptr;
#if GOOGLE_CUDA
            auto tf_allocator = context->get_allocator(context->output_alloc_attr(0));
            auto _allocator = TensorFlowCudaAllocator(tf_allocator);
            allocator = &_allocator;
#endif
            modeSpecificActionInt(context->eigen_device<Device>(), inTensorFlat, inTensor.NumElements(), outTensorFlat,
                           quantizerAddr, opMode, encodingMin, encodingMax, bitwidth, useSymmetricEncoding, allocator);
        }
        else
        {
            modeSpecificActionFp16<Device, T>(context, inTensor, quantizerAddr, opMode, outTensor);
        }
    }
};


#define REGISTER_CPU(T) \
    REGISTER_KERNEL_BUILDER(Name("QcQuantize").Device(DEVICE_CPU).TypeConstraint<T>("T"), QcQuantizeOp<CPUDevice, T>);

REGISTER_CPU(float);

// Register the GPU kernels.

#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T) \
    REGISTER_KERNEL_BUILDER(Name("QcQuantize").Device(DEVICE_GPU).TypeConstraint<T>("T"), QcQuantizeOp<GPUDevice, T>);
REGISTER_GPU(float);

#endif   // GOOGLE_CUDA
