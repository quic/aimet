//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2021-2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include "AimetOpUtils.h"
#include "AimetFp16OpUtils.h"
#include "QcQuantizePerChannelOp.hpp"

#define EIGEN_USE_THREADS
using namespace tensorflow;
using namespace std;
using namespace gtl;

enum AxisHandling
{
    LAST_AXIS = 0,
    LAST_TWO_AXES
};

REGISTER_OP("QcQuantizePerChannel")
    .Input("in_tensor: T")     // list of input tensors (weights/activations)
    .Input("op_mode: int32")   //{'ANALYSIS', 'ACTIVE', 'PASSTHROUGH'}")
    .Input("tensor_quantizer_reference: int64")
    .Input("encoding_min: double")
    .Input("encoding_max: double")
    .Input("bit_width: int8")
    .Input("use_symmetric_encoding: bool")
    .Input("is_int_data_type: bool")
    .Input("axis_handling: int32")
    .Input("is_training: bool")
    .Output("out_tensor: T")   // list of output tensors (weights/activations)

    .Attr("T: {float} = DT_FLOAT")   // attr 'T' specifies which template instantiation of op to use, default float
    .Doc(R"doc(QcQuantize Per Channel custom op.)doc")
    .SetShapeFn(
        [](::tensorflow::shape_inference::InferenceContext* c)
        {
            c->set_output(0, c->input(0));
            return Status::OK();
        });

TTypes<float>::ConstMatrix getTwoDimTensor(const Tensor& tensor, const AxisHandling axisHandling)
{
    if (axisHandling == AxisHandling::LAST_TWO_AXES)
    {
        // Combine first two dimensions and last two dimensions to isolate number of channels in the final
        // dimension.
        return tensor.flat_inner_outer_dims<float, 2>(1);
    }
    else
    {
        return tensor.flat_inner_dims<float, 2>();
    }
}

TTypes<float>::Matrix getTwoDimTensor(Tensor* tensor, const AxisHandling axisHandling)
{
    if (axisHandling == AxisHandling::LAST_TWO_AXES)
    {
        // Combine first two dimensions and last two dimensions to isolate number of channels in the final
        // dimension.
        return tensor->flat_inner_outer_dims<float, 2>(1);
    }
    else
    {
        return tensor->flat_inner_dims<float, 2>();
    }
}


DlQuantization::TfEncoding updateStatsAndComputeEncodingsTfFunctions(const Tensor inTensor, const int8 bw,
                                                                     const bool useSymEncoding)
{
    Tensor someScalar(DT_FLOAT, TensorShape({}));;
    auto scalar = someScalar.scalar<float>();

    scalar = inTensor.flat<float>().minimum();
    float min = scalar();
    scalar = inTensor.flat<float>().maximum();
    float max = scalar();

    float MIN_RANGE = 0.01;

    // Make sure zero value is within the range
    min  = std::min(0.0f, min);
    max  = std::max(0.0f, max);
    max = std::max(max, min + MIN_RANGE);

    double numSteps = std::pow(2, bw) - 1;

    DlQuantization::TfEncoding encoding;
    encoding.bw = (uint8) bw;

    if (useSymEncoding)
    {
        max = std::max(std::abs(max), std::abs(min));
        unsigned int numPositiveSteps = std::floor(numSteps / 2);
        encoding.delta = (double) max / numPositiveSteps;
        encoding.offset = -std::ceil(numSteps / 2);
        encoding.min = encoding.offset * encoding.delta;
        encoding.max = encoding.delta * numPositiveSteps;
    }
    else
    {
        encoding.delta = (max - min) / numSteps;
        encoding.offset = round(min / encoding.delta);
        encoding.min = encoding.offset * encoding.delta;
        encoding.max = encoding.min + (encoding.delta * numSteps);
    }

    return encoding;
}


template <typename D, typename T>
DlQuantization::TfEncoding updateStatsAndComputeEncodings(const D& d, const T* inTensor, size_t count,
                                                          const uint64* tensorQuantizerRef, const int8 bitwidth,
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
    auto tensorQuantizer = reinterpret_cast<DlQuantization::TensorQuantizerOpFacade*>(tensorQuantizerRefHost);
    auto useSymmetricEncoding = copyLiteralToHost<bool>(d, useSymEncoding);

    tensorQuantizer->updateStats(inTensor, count, useCuda, allocator);

    DlQuantization::TfEncoding initial_encoding = tensorQuantizer->computeEncoding(bitwidth, useSymmetricEncoding);
    return initial_encoding;

}

/*
 * Get TF encoding format by calculating delta offset from min and max.
 */
template <typename D>
DlQuantization::TfEncoding getTfEncoding(const D& d, const double* min, const double* max, const int8* bw)
{
    bool useCuda = false;
    if (std::is_same<D, GPUDevice>::value)
    {
        useCuda = true;
    }
    auto encodingMin = copyLiteralToHost<double>(d, min);
    auto encodingMax = copyLiteralToHost<double>(d, max);
    auto bitwidth    = copyLiteralToHost<int8>(d, bw);
    std::unique_ptr<DlQuantization::ITensorQuantizationSim<float>> _tensorQuantizationSim;
    _tensorQuantizationSim = DlQuantization::getTensorQuantizationSim<float>();

    DlQuantization::TfEncoding encoding;
    _tensorQuantizationSim->fillEncodingInfo(encoding, bitwidth, encodingMin, encodingMax);
    return encoding;
}

void generatePerChannelScaleOffset(Tensor* encodingMinTensor, Tensor* encodingMaxTensor, int8 bw,
                                   Tensor* encodingScaleTensor, Tensor* encodingOffsetTensor)
{
    int numChannels = encodingMinTensor->shape().dim_size(0);
    std::unique_ptr<DlQuantization::ITensorQuantizationSim<float>> _tensorQuantizationSim;
    _tensorQuantizationSim = DlQuantization::getTensorQuantizationSim<float>();

    double* encodingMin = encodingMinTensor->flat<double>().data();
    double* encodingMax = encodingMaxTensor->flat<double>().data();
    double* encodingScale = encodingScaleTensor->flat<double>().data();
    double* encodingOffset = encodingOffsetTensor->flat<double>().data();

    for(int channel = 0; channel < numChannels; channel++)
    {
        _tensorQuantizationSim->generateScaleOffset(*encodingMin, *encodingMax, bw, *encodingScale, *encodingOffset);
        encodingMin++; encodingMax++;
        encodingScale++; encodingOffset++;
    }
}

void generateInvScale(Tensor* encodingScaleTensor, Tensor* encodingInvScaleTensor)
{
    int numChannels = encodingScaleTensor->shape().dim_size(0);
    double *in = encodingScaleTensor->flat<double>().data();
    double *out = encodingInvScaleTensor->flat<double>().data();

    for(int i = 0; i < numChannels; i++)
    {
        *out = 1.0f / (double) *in;
        out++; in++;
    }
}

void UpdateEncodingTensorsForChannel(const DlQuantization::TfEncoding &encoding, int channelIdx, Tensor* encodingMinTensor,
                                     Tensor* encodingMaxTensor, Tensor* encodingScaleTensor, Tensor* encodingOffsetTensor)
{
    // Given the encoding, fill it in the encoding tensors at channelIdx
    // Assumes channelIdx is within bounds
    double* encodingMin = encodingMinTensor->flat<double>().data();
    double* encodingMax = encodingMaxTensor->flat<double>().data();
    double* encodingScale = encodingScaleTensor->flat<double>().data();
    double* encodingOffset = encodingOffsetTensor->flat<double>().data();

    encodingMin[channelIdx] = encoding.min;
    encodingMax[channelIdx] = encoding.max;
    encodingScale[channelIdx] = encoding.delta;
    encodingOffset[channelIdx] = encoding.offset;
}

template <typename D>
void copyConstTensorToNonConstTensor(const D& d, const Tensor* constTensor, Tensor* nonConstTensor)
{
    int numElements = constTensor->shape().dim_size(0);
    const double *in = constTensor->flat<double>().data();
    double *out = nonConstTensor->flat<double>().data();
    copyArrayToHost(d, in, out, numElements);
}

// OpKernel definition.
// 'Device is templated on the type of device.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class QcQuantizePerChannelOp : public OpKernel
{
public:
    explicit QcQuantizePerChannelOp(OpKernelConstruction* context) : OpKernel(context)
    {
    }

    void Compute(OpKernelContext* context) override
    {
        // Consume input tensor
        const Tensor& inTensor = context->input(0);
        // Get shape of input tensor by iterating over each dimension
        int numDimensionsTensor = inTensor.shape().dims();
        std::vector<int> shapeVector;
        for (int axis = 0; axis < numDimensionsTensor; axis++)
        {
            shapeVector.push_back(inTensor.shape().dim_size(axis));
        }

        // Read the op_mode
        const Tensor* opModeTensor;
        OP_REQUIRES_OK(context, context->input("op_mode", &opModeTensor));
        const int32* opMode = opModeTensor->flat<int32>().data();

        // Read the tensor quantizer ref
        const Tensor* quantizerRefTensor;
        OP_REQUIRES_OK(context, context->input("tensor_quantizer_reference", &quantizerRefTensor));
        uint64* quantizerAddr = (uint64*) quantizerRefTensor->flat<int64>().data();

        // Read the encoding_min
        const Tensor* encodingMinTensorConst;
        OP_REQUIRES_OK(context, context->input("encoding_min", &encodingMinTensorConst));

        // Read the encoding_max
        const Tensor* encodingMaxTensorConst;
        OP_REQUIRES_OK(context, context->input("encoding_max", &encodingMaxTensorConst));

        // read bitwidth
        const Tensor* bitwidthTensor;
        OP_REQUIRES_OK(context, context->input("bit_width", &bitwidthTensor));
        const int8 bitwidth = copyLiteralToHost<int8>(context->eigen_device<Device>(), bitwidthTensor->flat<int8>().data());

        // Read axis for per channel quantization
        const Tensor* axisHandlingTensor;
        OP_REQUIRES_OK(context, context->input("axis_handling", &axisHandlingTensor));
        const int32* axisHandlingInt = axisHandlingTensor->flat<int32>().data();

        // Move axis to correct device and get value
        auto axisHandling     = copyLiteralToHost<int32>(context->eigen_device<Device>(), axisHandlingInt);
        auto axisHandlingEnum = static_cast<const AxisHandling>(axisHandling);

        // Get number of channels
        int numChannels = 0;
        if (axisHandlingEnum == AxisHandling::LAST_TWO_AXES)
        {
            numChannels = shapeVector[numDimensionsTensor - 2] * shapeVector[numDimensionsTensor - 1];
        }
        else
        {
            // For normal case, last axis as number of channels.
            // This includes conv transpose since py function will transpose kernel prior to this op.
            numChannels = shapeVector[numDimensionsTensor - 1];
        }
        // Number of channels should be equal to the number of encodings provided.
        assert(numChannels == encodingMaxTensorConst->shape().dim_size(0));
        assert(numChannels == encodingMinTensorConst->shape().dim_size(0));

        // use symmetric encoding
        const Tensor* useSymmetricEncodingTensor;
        OP_REQUIRES_OK(context, context->input("use_symmetric_encoding", &useSymmetricEncodingTensor));
        auto useSymmetricEncoding = useSymmetricEncodingTensor->flat<bool>().data();

        // is_int_data_type
        const Tensor* isIntDataTypeTensor;
        OP_REQUIRES_OK(context, context->input("is_int_data_type", &isIntDataTypeTensor));
        auto isIntDataType = isIntDataTypeTensor->flat<bool>().data();

        // is_training flag
        const Tensor* isTrainingTensor;
        OP_REQUIRES_OK(context, context->input("is_training", &isTrainingTensor));
        auto isTraining = isTrainingTensor->flat<bool>().data();

        // allocate output tensors
        Tensor* outTensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, inTensor.shape(), &outTensor));

        if(!copyLiteralToHost<bool>(context->eigen_device<Device>(), isIntDataType))
        {
            assert(bitwidth == 16);
            modeSpecificActionFp16<Device, T>(context, inTensor, quantizerAddr, opMode, outTensor);
        }
        else
        {
            // Performs per layer quantization
            // For parameters in convolution layers or linear layers
            // TODO: transposed conv2d

            auto opModeHost = copyLiteralToHost<int32>(context->eigen_device<Device>(), opMode);
            auto opModeEnum = static_cast<const DlQuantization::TensorQuantizerOpMode>(opModeHost);

            if (opModeEnum == DlQuantization::TensorQuantizerOpMode::passThrough)
            {
                auto inTensorFlat  = inTensor.flat<T>().data();
                auto outTensorFlat = outTensor->flat<T>().data();
                copyInputTensorsToOutputTensors(context->eigen_device<Device>(), inTensorFlat, inTensor.NumElements(),
                                                outTensorFlat);
            }
            else
            {
                if (numDimensionsTensor == 4 or numDimensionsTensor == 2)
                {
                    // For linear layers
                    int numElements = shapeVector[0];
                    // For conv layers
                    if (numDimensionsTensor == 4)
                    {
                        if (axisHandlingEnum == AxisHandling::LAST_TWO_AXES)
                        {
                            // In tf nn depthwise case, 3rd and 4th dimensions will be combined as number of channels.
                            numElements = numElements * shapeVector[1];
                        }
                        else
                        {
                            numElements = numElements * shapeVector[1] * shapeVector[2];
                        }
                    }
                    Tensor temp1;
                    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({2, numElements}), &temp1));

                    TTypes<float>::ConstMatrix inTensorTwoDim = getTwoDimTensor(inTensor, axisHandlingEnum);
                    TTypes<float>::Matrix outTensorTwoDim     = getTwoDimTensor(outTensor, axisHandlingEnum);

                    DlQuantization::IAllocator* allocator = nullptr;
#if GOOGLE_CUDA
                    auto tf_allocator = context->get_allocator(context->output_alloc_attr(0));
                    auto _allocator = TensorFlowCudaAllocator(tf_allocator);
                    allocator = &_allocator;
#endif

                    // allocate tensors for scale and offset. By default, TF would allocate tensors in the device
                    // where the op is currently executing in. To always allocate on the Host, additional argument
                    // of type AllocatorAttributes needs to be passed. The object should be set to be allocated on
                    // host and also made GPU compatible.
                    Tensor encodingMinTensor, encodingMaxTensor, encodingScaleTensor, encodingOffsetTensor, encodingInvScaleTensor;
                    AllocatorAttributes attr;
                    attr.set_on_host(true);
#if GOOGLE_CUDA
                    attr.set_gpu_compatible(true);
#endif

                    OP_REQUIRES_OK(context, context->allocate_temp(DT_DOUBLE, encodingMinTensorConst->shape(),
                                    &encodingMinTensor, attr));
                    OP_REQUIRES_OK(context, context->allocate_temp(DT_DOUBLE, encodingMinTensorConst->shape(),
                                    &encodingMaxTensor, attr));
                    OP_REQUIRES_OK(context, context->allocate_temp(DT_DOUBLE, encodingMinTensorConst->shape(),
                                    &encodingScaleTensor, attr));
                    OP_REQUIRES_OK(context, context->allocate_temp(DT_DOUBLE, encodingMinTensorConst->shape(),
                                    &encodingOffsetTensor, attr));
                    OP_REQUIRES_OK(context, context->allocate_temp(DT_DOUBLE, encodingMinTensorConst->shape(),
                                    &encodingInvScaleTensor, attr));

                    // min/max tensors need to be made non-const because they will be modified
                    copyConstTensorToNonConstTensor(context->eigen_device<Device>(), encodingMinTensorConst,
                                                    &encodingMinTensor);
                    copyConstTensorToNonConstTensor(context->eigen_device<Device>(), encodingMaxTensorConst,
                                                    &encodingMaxTensor);

                    if (opModeEnum == DlQuantization::TensorQuantizerOpMode::oneShotQuantizeDequantize)
                    {
                        for (int channel = 0; channel < numChannels; channel++)
                        {
                            // Chip input tensor along last dimensions
                            chipAndCopyPerChannelValues(context->eigen_device<Device>(), temp1, inTensorTwoDim, channel);
                            auto inpData = temp1.flat<float>().data();

                            DlQuantization::TfEncoding encodings = updateStatsAndComputeEncodings(context->eigen_device<Device>(),
                                                                                                  inpData, numElements,
                                                                                                  quantizerAddr++,
                                                                                                  bitwidth, useSymmetricEncoding,
                                                                                                  allocator);
                            UpdateEncodingTensorsForChannel(encodings, channel, &encodingMinTensor, &encodingMaxTensor,
                                                            &encodingScaleTensor, &encodingOffsetTensor);
                        }
                    }
                    else if (opModeEnum == DlQuantization::TensorQuantizerOpMode::quantizeDequantize)
                    {

                        generatePerChannelScaleOffset(&encodingMinTensor, &encodingMaxTensor, bitwidth,
                                                      &encodingScaleTensor, &encodingOffsetTensor);
                    }
                    else
                    {
                        printf("unsupported mode\n");
                        assert(0);
                    }
                    generateInvScale(&encodingScaleTensor, &encodingInvScaleTensor);
                    quantizeDequantizePerChannel(context->eigen_device<Device>(), inTensorTwoDim, outTensorTwoDim,
                                                 &encodingMinTensor, &encodingMaxTensor, &encodingScaleTensor,
                                                 &encodingOffsetTensor, &encodingInvScaleTensor);
                }
                else if (numDimensionsTensor == 1)
                {
                    DlQuantization::IAllocator* allocator = nullptr;
#if GOOGLE_CUDA
                    auto tf_allocator = context->get_allocator(context->output_alloc_attr(0));
                    auto _allocator = TensorFlowCudaAllocator(tf_allocator);
                    allocator = &_allocator;
#endif
                    // Per channel quantization for Bias
                    int numElements    = 1;
                    auto inTensorFlat  = inTensor.flat<T>().data();
                    auto outTensorFlat = outTensor->flat<T>().data();
                    for (int channel = 0; channel < numChannels; channel++)
                    {
                        const double* encodingMax = encodingMaxTensorConst->flat<double>().data();
                        const double* encodingMin = encodingMinTensorConst->flat<double>().data();
                        modeSpecificActionInt(context->eigen_device<Device>(), inTensorFlat++, numElements, outTensorFlat++,
                                              quantizerAddr++, opMode, encodingMin++, encodingMax++,
                                              bitwidthTensor->flat<int8>().data(), useSymmetricEncoding, allocator);
                    }
                }
            }
        }
    }
};


#define REGISTER_CPU(T)                                                                             \
    REGISTER_KERNEL_BUILDER(Name("QcQuantizePerChannel").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
                            QcQuantizePerChannelOp<CPUDevice, T>);

REGISTER_CPU(float);

// Register the GPU kernels.

#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                                                             \
    REGISTER_KERNEL_BUILDER(Name("QcQuantizePerChannel").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
                            QcQuantizePerChannelOp<GPUDevice, T>);
REGISTER_GPU(float);

#endif   // GOOGLE_CUDA
