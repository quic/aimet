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

#include "QcQuantizeRecurrentParamOp.hpp"
#include "AimetOpUtils.h"
#include <type_traits>

#define EIGEN_USE_THREADS
using namespace tensorflow;

REGISTER_OP("QcQuantizeRecurrentParam")
    .Input("in_tensor: T")     // list of input tensors (weights/activations)
    .Input("op_mode: int32")   //{'ANALYSIS', 'ACTIVE', 'PASSTHROUGH'}")
    .Input("tensor_quantizer_reference: int64")
    .Input("encoding_min: double")
    .Input("encoding_max: double")
    .Input("bit_width: int8")
    .Input("use_symmetric_encoding: bool")
    .Input("time_steps: int32")
    .Output("out_tensor: T")   // list of output tensors (weights/activations)
    .Attr("T: {float} = DT_FLOAT")   // attr 'T' specifies which template instantiation of op to use, default float
    .Doc(R"doc(QcQuantizeRecurrentParam custom op.)doc")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

template <typename D, typename T>
void modeSpecificAction(const D& d, const T* inTensor, size_t count, T* outTensor,
                        const uint64* tensorQuantizerRef, const int32* opMode, const double* min, const double* max,
                        const int8* bw, const bool* useSymEncoding, const int32* timeStepInOp,
                        int &computeEncodingCounter, DlQuantization::TfEncoding &cachedEncoding, bool &isEncodingValid,
                        DlQuantization::IAllocator* allocator)
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
    auto tensorQuantizer = reinterpret_cast<DlQuantization::TensorQuantizerOpFacade*>(tensorQuantizerRefHost);
    // these are needed by QAT 2.0
    auto encodingMin = copyLiteralToHost<double>(d, min);
    auto encodingMax = copyLiteralToHost<double>(d, max);
    auto opModeEnum = static_cast<const DlQuantization::TensorQuantizerOpMode>(opModeHost);
    auto bitwidth = copyLiteralToHost<int8>(d, bw);
    auto useSymmetricEncoding = copyLiteralToHost<bool>(d, useSymEncoding);
    auto timesteps = copyLiteralToHost<int32>(d, timeStepInOp);

    switch (opModeEnum)
    {
    case DlQuantization::TensorQuantizerOpMode::oneShotQuantizeDequantize:
    {

        if(computeEncodingCounter % (int32)timesteps == 0)
        {
            // a recurrent param quantizer has special handling below to reduce the number of
            // updateStats and encoding computations performed.
            // Instead of every time step we compute encoidngs only at h0.
            tensorQuantizer->updateStats(inTensor, count, useCuda, allocator);
            cachedEncoding = tensorQuantizer->computeEncoding(bitwidth, useSymmetricEncoding);
            tensorQuantizer->quantizeDequantize(inTensor, count, outTensor, cachedEncoding.min, cachedEncoding.max,
                                                bitwidth, useCuda);
            isEncodingValid = true;
            // finished one set of time_steps reset the counter to zero
            computeEncodingCounter = 0;
        }
        else
        {
            // check if the encodings have been cached before usage
            assert(isEncodingValid);
            // use cached encoding for param quantizer in decimated oneShot mode
            tensorQuantizer->quantizeDequantize(inTensor, count, outTensor, cachedEncoding.min, cachedEncoding.max,
                                                bitwidth, useCuda);
        }

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

    // mark the number of times we have executed
    computeEncodingCounter++;
}

// OpKernel definition.
// 'Device is templated on the type of device.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class QcQuantizeRecurrentParamOp : public OpKernel
{
private:
    int _computeEncodingCounter;
    DlQuantization::TfEncoding _cachedEncoding;
    bool _isEncodingValid;
public:
    explicit QcQuantizeRecurrentParamOp(OpKernelConstruction* context) : OpKernel(context)
    {
        _computeEncodingCounter = 0;
        _cachedEncoding.min = 0.0;
        _cachedEncoding.max = 0.0;
        _cachedEncoding.offset = 0.0;
        _cachedEncoding.delta = 0.0;
        _cachedEncoding.bw = 0.0;
        _isEncodingValid = false;
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

        // num time_steps
        const Tensor* timeStepsTensor;
        OP_REQUIRES_OK(context, context->input("time_steps", &timeStepsTensor));
        const int32* time_steps = (int32*)  timeStepsTensor->flat<int32>().data();

        // allocate output tensors
        Tensor* outTensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, inTensor.shape(), &outTensor));
        auto outTensorFlat = outTensor->flat<T>().data();

        DlQuantization::IAllocator* allocator = nullptr;
#if GOOGLE_CUDA
        auto tf_allocator = context->get_allocator(context->output_alloc_attr(0));
        auto _allocator = TensorFlowCudaAllocator(tf_allocator);
        allocator = &_allocator;
#endif

        modeSpecificAction(context->eigen_device<Device>(), inTensorFlat, inTensor.NumElements(), outTensorFlat,
                           quantizerAddr, opMode,  encodingMin, encodingMax, bitwidth, useSymmetricEncoding,
                           time_steps, _computeEncodingCounter, _cachedEncoding, _isEncodingValid, allocator);
    }
};


#define REGISTER_CPU(T) \
    REGISTER_KERNEL_BUILDER(Name("QcQuantizeRecurrentParam").Device(DEVICE_CPU).TypeConstraint<T>("T"), QcQuantizeRecurrentParamOp<CPUDevice, T>);

REGISTER_CPU(float);

// Register the GPU kernels.

#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T) \
    REGISTER_KERNEL_BUILDER(Name("QcQuantizeRecurrentParam").Device(DEVICE_GPU).TypeConstraint<T>("T"), QcQuantizeRecurrentParamOp<GPUDevice, T>);
REGISTER_GPU(float);

#endif   // GOOGLE_CUDA
