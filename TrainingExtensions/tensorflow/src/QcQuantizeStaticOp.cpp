//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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

#include "QcQuantizeStaticOp.hpp"
#include "AimetOpUtils.h"

#include <memory>
#include <string>
#include <type_traits>

#define EIGEN_USE_THREADS
using namespace tensorflow;


REGISTER_OP("QcQuantizeStatic")
    .Input("in_tensor: T")     // list of input tensors (weights/activations)
    .Output("out_tensor: T")   // list of output tensors (weights/activations)

    .Attr("T: {float} = DT_FLOAT")   // attr 'T' specifies which template instantiation of op to use, default float
    .Attr("quant_scheme: int")
    .Attr("op_mode: int")
    .Attr("bitwidth: int")
    .Attr("encoding_min: float")
    .Attr("encoding_max: float")
    .Attr("round_mode: {'NEAREST', 'STOCHASTIC'} = 'NEAREST'")
    .Attr("is_symmetric: bool")

    .Doc(R"doc(Static version of the QcQuantize custom op.)doc")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });



// OpKernel definition.
// 'Device is templated on the type of device.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class QcQuantizeStaticOp : public OpKernel
{
public:
    explicit QcQuantizeStaticOp(OpKernelConstruction* context) : OpKernel(context)
    {
        // Read and store the attributes
        int quantSchemeInt;
        OP_REQUIRES_OK(context, context->GetAttr("quant_scheme", &quantSchemeInt));
        _quantScheme = static_cast<DlQuantization::QuantizationMode>(quantSchemeInt);

        OP_REQUIRES_OK(context, context->GetAttr("is_symmetric", &_isSymmetric));

        OP_REQUIRES_OK(context, context->GetAttr("bitwidth", &_bitwidth));
        OP_REQUIRES(context, (_bitwidth >= 4 || _bitwidth <= 31),
                    errors::InvalidArgument("Supported bitwidths are 4..31"));

        OP_REQUIRES_OK(context, context->GetAttr("encoding_min", &_encodingMin));
        OP_REQUIRES_OK(context, context->GetAttr("encoding_max", &_encodingMax));

        std::string roundModeString;
        OP_REQUIRES_OK(context, context->GetAttr("round_mode", &roundModeString));
        OP_REQUIRES(context, (roundModeString == "NEAREST" || roundModeString == "STOCHASTIC"),
                    errors::InvalidArgument("Round mode string must be NEAREST/STOCHASTIC"
                                            "; is '" + roundModeString + "'"));

        if (roundModeString == "NEAREST")
            _roundingMode = DlQuantization::ROUND_NEAREST;
        else
            _roundingMode = DlQuantization::ROUND_STOCHASTIC;

        int opModeInt;
        OP_REQUIRES_OK(context, context->GetAttr("op_mode", &opModeInt));
        _opMode = static_cast<DlQuantization::TensorQuantizerOpMode>(opModeInt);

        _tensorQuantizationSim = DlQuantization::getTensorQuantizationSim<float>();
    }

    void Compute(OpKernelContext* context) override
    {
        // Consume input tensor
        const Tensor& inTensor = context->input(0);
        auto inTensorFlat      = inTensor.flat<T>().data();

        // allocate output tensors
        Tensor* outTensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, inTensor.shape(), &outTensor));
        auto outTensorFlat = outTensor->flat<T>().data();

        modeSpecificAction(context->eigen_device<Device>(), inTensorFlat, inTensor.NumElements(), outTensorFlat);
    }

    void modeSpecificAction(const Device& d, const T* inTensor, size_t count, T* outTensor)
    {
        modeSpecificAction(d, inTensor, count, outTensor, nullptr);
    }

    void modeSpecificAction(const Device& d, const T* inTensor, size_t count, T* outTensor, DlQuantization::IAllocator* allocator)
    {
        bool useCuda = false;
        if (std::is_same<Device, GPUDevice>::value)
        {
            useCuda = true;
        }

        switch (_opMode)
        {
        case DlQuantization::TensorQuantizerOpMode::oneShotQuantizeDequantize:
        {
            auto tensorQuantizer = DlQuantization::TensorQuantizer(_quantScheme, _roundingMode);
            tensorQuantizer.updateStats(inTensor, count, useCuda, allocator);
            DlQuantization::TfEncoding initial_encoding = tensorQuantizer.computeEncoding(_bitwidth, _isSymmetric);
            tensorQuantizer.quantizeDequantize(inTensor, count, outTensor, initial_encoding.min,
                                               initial_encoding.max, _bitwidth, useCuda);
            break;
        }
        case DlQuantization::TensorQuantizerOpMode::quantizeDequantize:
        {
            _tensorQuantizationSim->quantizeDequantizeTensor(inTensor, count, outTensor, _encodingMin, _encodingMax,
                                                             _bitwidth, _roundingMode, useCuda);
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

private:
    int _bitwidth;
    T _encodingMin;
    T _encodingMax;
    DlQuantization::RoundingMode _roundingMode;
    DlQuantization::TensorQuantizerOpMode _opMode;
    DlQuantization::QuantizationMode _quantScheme;
    bool _isSymmetric;
    std::unique_ptr<DlQuantization::ITensorQuantizationSim<float>> _tensorQuantizationSim;
};


REGISTER_KERNEL_BUILDER(Name("QcQuantizeStatic").Device(DEVICE_CPU).TypeConstraint<float>("T"), QcQuantizeStaticOp<CPUDevice, float>);

#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("QcQuantizeStatic").Device(DEVICE_GPU).TypeConstraint<float>("T"), QcQuantizeStaticOp<GPUDevice, float>);
#endif   // GOOGLE_CUDA
