//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2019-2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

#include <math.h>
#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>

#include "trim_functions.hpp"
#include "quantization_utils.hpp"
#include "TensorQuantizationSim.h"

namespace DlQuantization
{

uint8_t getBw(int bw) { return std::max(bw,8); }

template <typename DTYPE>
TensorQuantizationSim<DTYPE>::TensorQuantizationSim()
{
}

template <typename DTYPE>
void TensorQuantizationSim<DTYPE>::generateScaleOffset(double &encodingMin, double &encodingMax, uint8_t bw,
                                                       double &encodingScale, double &encodingOffset)
{
    gateMinMax(encodingMin, encodingMax);

    // Detect if we are in strict-symmetric mode
    double numSteps = pow(2, bw) - 1;
    if (encodingMin == -encodingMax)
    {
        numSteps -= 1;  // in case of 8-bits, strict symmetric means we use 254 int values, instead of 255
    }

    // compute offset and delta on the fly
    encodingScale = computeDelta(encodingMin, encodingMax, numSteps);
    encodingOffset = computeOffset(encodingMin, encodingScale);

    // recalculate the encoding.min and encoding.max based on the new delta and offset
    encodingMin = encodingOffset * encodingScale;
    encodingMax = encodingScale * numSteps + encodingMin;
}

template <typename DTYPE>
void TensorQuantizationSim<DTYPE>::fillEncodingInfo(TfEncoding& encoding, uint8_t bw, double encodingMin,
                                                    double encodingMax)
{

    encoding.bw = bw;
    encoding.min = encodingMin;
    encoding.max = encodingMax;
    generateScaleOffset(encoding.min, encoding.max, bw, encoding.delta, encoding.offset);

}

template <typename DTYPE>
void TensorQuantizationSim<DTYPE>::quantizeDequantizeTensor(const DTYPE* inputTensorData, size_t inputTensorCount,
                                                            DTYPE* outputTensorData, double encodingMin,
                                                            double encodingMax, uint8_t bw, RoundingMode roundingMode,
                                                            bool use_cuda)
{
    TfEncoding encoding;
    fillEncodingInfo(encoding, bw, encodingMin, encodingMax);
    quantizeDequantize(inputTensorData, inputTensorCount, encoding, outputTensorData, getComputationMode(use_cuda),
                        roundingMode);
}

template <typename DTYPE>
void TensorQuantizationSim<DTYPE>::quantizeTensor(const DTYPE* inputTensorData, size_t inputTensorCount,
                                                  DTYPE* outputTensorData, double encodingMin, double encodingMax,
                                                  uint8_t bw, RoundingMode roundingMode, bool use_cuda,
                                                  bool shiftToSigned)
{
    TfEncoding encoding;
    fillEncodingInfo(encoding, bw, encodingMin, encodingMax);
    quantizeToFxp(inputTensorData, inputTensorCount, encoding, outputTensorData, getComputationMode(use_cuda),
                  roundingMode, shiftToSigned);
}

template <typename DTYPE>
void TensorQuantizationSim<DTYPE>::quantizeTensorPacked(const DTYPE* inputTensorData, size_t inputTensorCount,
                                                        std::vector<uint8_t>& outputTensorData, double encodingMin, double encodingMax,
                                                        uint8_t bw, RoundingMode roundMode, bool useCuda,
                                                        bool shiftToSigned)
{
    TfEncoding encoding{};
    fillEncodingInfo(encoding, bw, encodingMin, encodingMax);
    outputTensorData.resize(ceil(getBw(bw) * inputTensorCount / 8.0));
    quantizeToFxpPacked(inputTensorData, inputTensorCount, encoding, outputTensorData.data(), outputTensorData.size(),
                        getComputationMode(useCuda), roundMode, shiftToSigned);
}

template <typename DTYPE>
void TensorQuantizationSim<DTYPE>::quantizeDequantizePerChannelTensor(
    std::vector<std::vector<DTYPE>>& splits, std::vector<uint32_t> splitShape, uint32_t axis, DTYPE* outputTensorData,
    const std::vector<TfEncoding>& encodings, uint8_t bw, RoundingMode roundMode, bool useCuda)
{
    std::vector<TfEncoding> completeEncodings;

    // assume encoding max and min, then fill delta and offset info after gating
    completeEncodings.resize(encodings.size());
    for (auto idx = 0; idx < encodings.size(); idx++)
    {
        fillEncodingInfo(completeEncodings[idx], bw, encodings[idx].min, encodings[idx].max);
    }

    // Loop through splits and quantize each independently
    for (uint32_t i = 0; i < splits.size(); ++i) {
        auto& split = splits[i];
        quantizeDequantize(split.data(), split.size(), completeEncodings[i], split.data(), getComputationMode(useCuda),
                            roundMode);
    }

    // Concatenate the quantized data back into its original shape.
    std::vector <uint32_t> outputShape;
    concat(splits, splitShape, axis, outputTensorData, outputShape);
}

template <typename DTYPE>
void TensorQuantizationSim<DTYPE>::quantizePerChannelTensorPacked(std::vector<std::vector<DTYPE>>& splits,
                                                                  std::vector<uint32_t> splitShape, uint32_t axis,
                                                                  std::vector<uint8_t>& outputTensorData,
                                                                  const std::vector<TfEncoding>& encodings, uint8_t bw,
                                                                  RoundingMode roundMode, bool useCuda,
                                                                  bool shiftToSigned)
{
    std::vector<TfEncoding> completeEncodings;

    // assume encoding max and min
    completeEncodings.resize(encodings.size());

    std::vector<std::vector<uint8_t>> qSplits(splits.size());
    uint32_t qSplitSize = ceil((getBw(bw) * splits[0].size())/8.0);

    for (auto idx = 0; idx < encodings.size(); idx++)
    {
        fillEncodingInfo(completeEncodings[idx], bw, encodings[idx].min, encodings[idx].max);
    }

    // Loop through splits and quantize each independently
    for (uint32_t i = 0; i < splits.size(); ++i) {
        auto& split = splits[i];
        auto& qSplit = qSplits[i];
        qSplit.resize(qSplitSize);
        quantizeToFxpPacked(split.data(), split.size(), completeEncodings[i], qSplit.data(),
                            qSplitSize, getComputationMode(useCuda), roundMode, shiftToSigned);
    }

    uint32_t outputCount = std::accumulate(std::begin(splitShape), std::end(splitShape), splits.size(),
                                           std::multiplies<uint32_t>());

    // Concatenate the quantized data back into its original shape.
    outputTensorData.resize(ceil((getBw(bw) * outputCount) / 8.0));
    std::vector <uint32_t> outputShape;
    concat(qSplits, splitShape, axis, outputTensorData.data(), outputShape);
}

template <typename DTYPE>
void TensorQuantizationSim<DTYPE>::dequantizeTensor(const uint8_t* inputTensorData, size_t inputTensorCount,
                                                    DTYPE* output, double encodingMin, double encodingMax, uint8_t bw,
                                                    bool shiftToSigned)
{
    TfEncoding encoding{};
    fillEncodingInfo(encoding, bw, encodingMin, encodingMax);
    dequantizeFromPackedFxp(inputTensorData, inputTensorCount, encoding, output, getComputationMode(false),
                            shiftToSigned);
}
template <typename DTYPE>
void TensorQuantizationSim<DTYPE>::dequantizePerChannelTensor(const uint8_t* inputTensorData,
                                                              const std::vector<uint32_t>& inputShape, uint32_t axis,
                                                              DTYPE* outputTensorData, uint8_t bw,
                                                              const std::vector<TfEncoding>& encodings,
                                                              bool shiftToSigned)
{
    std::vector<TfEncoding> completeEncodings;

    // assume encoding max and min
    completeEncodings.resize(encodings.size());
    for (auto idx = 0; idx < encodings.size(); idx++)
    {
        fillEncodingInfo(completeEncodings[idx], bw, encodings[idx].min, encodings[idx].max);
    }

    std::vector<uint32_t> splitShape;
    std::vector<std::vector<uint8_t>> splits;

    if(inputShape.size() != 4) {
        throw std::invalid_argument("Per-channel quantization only operates on 4 dimensional data!");
    }

    if(axis > 3) {
        throw std::invalid_argument("Per-channel axis must be < 4");
    }

    if(encodings.size() != inputShape[axis]) {
        throw std::invalid_argument("Must provide all encodings for per-channel dequantization");
    }

    // Split the data by axis
    slice(inputTensorData, inputShape, axis, splits, splitShape);
    if(splits.size() != inputShape[axis]) {
        throw std::runtime_error("Invalid slice count generated. Count must be equal to axis split on!");
    }

    uint32_t splitCount = std::accumulate(std::begin(splitShape), std::end(splitShape), 1, std::multiplies<uint32_t>());
    uint32_t outputCount = std::accumulate(std::begin(inputShape), std::end(inputShape), 1, std::multiplies<uint32_t>());
    if(outputCount != splitCount*splits.size()) {
        throw std::runtime_error("Accumulated split count doesn't match original input count");
    }

    std::vector<std::vector<DTYPE>> splits_dequant(splits.size(), std::vector<DTYPE>(splitCount));

    for(uint32_t i = 0; i < splits.size(); ++i) {
        auto& e = encodings[i];
        auto& split = splits[i];
        if(split.size() != splitCount) {
            throw std::runtime_error("Tensor split size mismatch!");
        }
        dequantizeTensor(split.data(), split.size(), splits_dequant[i].data(),
                         e.min,e.max, bw, shiftToSigned);
    }

    std::vector<uint32_t> dummy;
    concat(splits_dequant, splitShape, axis, outputTensorData, dummy);
    (void)dummy;
}

template class TensorQuantizationSim<float>;
template class TensorQuantizationSim<double>;

}