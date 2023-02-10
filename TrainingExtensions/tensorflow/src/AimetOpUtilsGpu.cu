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

#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "AimetOpUtils.h"

using namespace tensorflow;

#define EIGEN_USE_GPU
typedef Eigen::GpuDevice GPUDevice;


// GPU specialization of actual computations.
template <typename T>
void copyInputTensorsToOutputTensors(const GPUDevice& d, const T* inTensor, size_t count, T* outTensor)
{
    // copy input_tensor to output_tensor
    cudaMemcpy(outTensor, inTensor, count * sizeof(float), cudaMemcpyDeviceToDevice);
}

template <typename T>
T copyLiteralToHost(const GPUDevice& d, const T* deviceValue)
{
    T hostValue;
    cudaMemcpy(&hostValue, deviceValue, sizeof(T), cudaMemcpyDeviceToHost);

    return hostValue;
}

template <typename T>
void copyArrayToHost(const GPUDevice& d, const T* srcPtr, T* destPtr, int count)
{
    // copies array from device to host
    // assumes srcPtr is pointing to array in the device memory, destPtr points to array in host memory
    cudaMemcpy(destPtr, srcPtr, sizeof(T) * count, cudaMemcpyDeviceToHost);
}

void chipAndCopyPerChannelValues(const GPUDevice& d, Tensor tensorToCopyInto,
                                 TTypes<float>::ConstMatrix tensorToCopyFrom, int channel)
{
    // Tensor.chip<dimension>(offset) means it slice tensor and get sub-tensor at the given offset in the dimension dim
    // For example, if tensor has 16x3 shape, the result tensor of
    // chip<0>(0) will take sub-tensor 0th tensor from row dimension having 1x3 shape tensor
    // chip<1>(2) will take sub-tensor 2nd tensor from column dimension having 16x1 shape tensor
    tensorToCopyInto.tensor<float, 2>().chip<0>(0).device(d) = tensorToCopyFrom.chip<1>(channel);
}

void sliceAndStoreTensor(const GPUDevice& d, Tensor* slicedTensor, Tensor tensorToSlice, int channel)
{
    auto slicedTensorTwoDim = slicedTensor->flat_inner_dims<float, 2>();
    slicedTensorTwoDim.chip<1>(channel).device(d) = tensorToSlice.tensor<float, 2>().chip<0>(0);
}

void quantizeDequantize(const GPUDevice& d, TTypes<float>::ConstMatrix inputs,
                        DlQuantization::TfEncoding encodings, TTypes<float>::Matrix outputs, int channel)
{
    float invScale, scale, offset, min, max;
    invScale = 1.0f / ((float) encodings.delta);
    scale    = (float) encodings.delta;
    offset   = (float) encodings.offset;
    min      = (float) encodings.min;
    max      = (float) encodings.max;

    const auto clampedTensor         = inputs.chip<1>(channel).cwiseMax(min).cwiseMin(max);
    const auto tensor = (clampedTensor * invScale).round() + offset;
    outputs.chip<1>(channel).device(d) = (tensor - offset) * scale;

};

void quantizeDequantizePerChannel(const GPUDevice& d, TTypes<float>::ConstMatrix inputs, TTypes<float>::Matrix outputs,
                           Tensor* encodingMinTensor, Tensor* encodingMaxTensor, Tensor* encodingScaleTensor,
                           Tensor* encodingOffset, Tensor* encodingInvScaleTensor)
{
    // Input matrix dimensions [numRows X numChannels]. Extracting numRows below
    int numRows =  inputs.dimension(0);

    // the encodings tensors have the dimensions of [numChannels X 1]. But, since the input is of shape
    // [numRows X numChannels], we would need to broadcast the encodings tensors to bring them to required shape.
    // Ref: https://eigen.tuxfamily.org/dox/unsupported/eigen_tensors.html#title89
    Eigen::array<int, 2> bcast({numRows, 1});
    auto encodingMinBcast = encodingMinTensor->flat<double>().template cast<float>().broadcast(bcast);
    auto encodingMaxBcast = encodingMaxTensor->flat<double>().template cast<float>().broadcast(bcast);
    auto encodingScaleBcast = encodingScaleTensor->flat<double>().template cast<float>().broadcast(bcast);
    auto encodingInvScaleBcast = encodingInvScaleTensor->flat<double>().template cast<float>().broadcast(bcast);

    // Do note that the below operations omit doing offset add/subtract since it is unnecessary here
    auto clampedTensor = inputs.cwiseMax(encodingMinBcast).cwiseMin(encodingMaxBcast);
    auto tensor1 = (clampedTensor * encodingInvScaleBcast).round() * encodingScaleBcast;
    outputs.device(d) = tensor1;
}

template void copyInputTensorsToOutputTensors(const GPUDevice& d, const float* inTensor, size_t count, float* outTensor);
template int8 copyLiteralToHost<int8>(const GPUDevice&, const int8* deviceValue);
template int32 copyLiteralToHost<int32>(const GPUDevice&, const int32* deviceValue);
template uint64 copyLiteralToHost<uint64>(const GPUDevice&, const uint64* deviceValue);
template double copyLiteralToHost<double>(const GPUDevice&, const double* deviceValue);
template bool copyLiteralToHost<bool>(const GPUDevice&, const bool* deviceValue);
template void copyArrayToHost<double>(const GPUDevice& d, const double* srcPtr, double* destPtr, int count);
template void copyArrayToHost<float>(const GPUDevice& d, const float* srcPtr, float* destPtr, int count);

#endif   // GOOGLE_CUDA