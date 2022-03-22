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

#include "AimetOpUtils.h"

#include <type_traits>

#define EIGEN_USE_THREADS
using namespace tensorflow;

// CPU specialization of actual computations.
template <typename T>
void copyInputTensorsToOutputTensors(const CPUDevice& d, const T* inTensor, size_t count, T* outTensor)
{
    // copy input_tensor to output_tensor
    std::copy(inTensor, inTensor + count, outTensor);
}

template <typename T>
T copyLiteralToHost(const CPUDevice& d, const T* deviceValue)
{
    return *deviceValue;
}

void sliceTensorAlongLastDim(const CPUDevice& d, Tensor slicedTensor, const Tensor& tensorToSlice, int channel)
{
    // K x K x I x O -> N x O
    auto tensorToSliceTwoDim = tensorToSlice.flat_inner_dims<float, 2>();
    slicedTensor.tensor<float, 2>().chip<0>(0) = tensorToSliceTwoDim.chip<1>(channel);


}

void sliceAndStoreTensor(const CPUDevice& d, Tensor* slicedTensor, Tensor tensorToSlice, int channel)
{
    auto slicedTensorTwoDim = slicedTensor->flat_inner_dims<float, 2>();
    slicedTensorTwoDim.chip<1>(channel) = tensorToSlice.tensor<float, 2>().chip<0>(0);

}

void quantizeDequantize(const CPUDevice& d, TTypes<float>::ConstMatrix inputs,
                        DlQuantization::TfEncoding encodings, TTypes<float>::Matrix outputs, int channel)
{
    float invScale, scale, offset, min, max;
    // Add epsilon 10-5 to avoid divide by zero error
    invScale = 1.0f / ((float) encodings.delta + 0.00001);
    scale    = (float) encodings.delta;
    offset   = (float) encodings.offset;
    min      = (float) encodings.min;
    max      = (float) encodings.max;

    const auto clampedTensor         = inputs.chip<1>(channel).cwiseMax(min).cwiseMin(max);
    const auto tensor = (clampedTensor * invScale).round() + offset;
    outputs.chip<1>(channel) = (tensor - offset) * scale;

};

template void copyInputTensorsToOutputTensors(const CPUDevice& d, const float* inTensor, size_t count, float* outTensor);
template int8 copyLiteralToHost<int8>(const CPUDevice&, const int8* deviceValue);
template int32 copyLiteralToHost<int32>(const CPUDevice&, const int32* deviceValue);
template uint64 copyLiteralToHost<uint64>(const CPUDevice&, const uint64* deviceValue);
template double copyLiteralToHost<double>(const CPUDevice&, const double* deviceValue);
template bool copyLiteralToHost<bool>(const CPUDevice&, const bool* deviceValue);
