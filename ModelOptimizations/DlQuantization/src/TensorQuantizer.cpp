//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2020 - 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

#include "DlQuantization/TensorQuantizer.h"
#include "DlQuantization/QuantizerFactory.hpp"
#include "quantization_utils.hpp"
#include <cassert>
#include <numeric>
#include <stdexcept>

namespace DlQuantization
{

TensorQuantizer::TensorQuantizer(QuantizationMode quantScheme, RoundingMode roundingMode) :
    _quantScheme(quantScheme),
    roundingMode(roundingMode),
    isEncodingValid(false),
    _useStrictSymmetric(false),
    _useUnsignedSymmetric(false),
    _validStats(false)
{
    _encodingAnalyzer      = getEncodingAnalyzerInstance<float>(quantScheme);
    _tensorQuantizationSim = getTensorQuantizationSim<float>();
}

QuantizationMode TensorQuantizer::getQuantScheme()
{
    return _quantScheme;
}

void TensorQuantizer::setQuantScheme(QuantizationMode quantScheme)
{
    // update quantScheme held by Tensor Quantizer
    _quantScheme = quantScheme;

    // create new encoding analyzer instance and reset associated flags
    // _validStats is tightly coupled with the encoding analyzer instance, needs reset
    resetEncodingStats();
}

bool TensorQuantizer::getStrictSymmetric()
{
    return _useStrictSymmetric;
}

void TensorQuantizer::setStrictSymmetric(bool useStrictSymmetric)
{
    // update strict symmetric flag held by Tensor Quantizer
    _useStrictSymmetric = useStrictSymmetric;
    // create new encoding analyzer instance and reset associated flags
    // _validStats is tightly coupled with the encoding analyzer instance, needs reset
    resetEncodingStats();
}

bool TensorQuantizer::getUnsignedSymmetric()
{
    return _useUnsignedSymmetric;
}

void TensorQuantizer::setUnsignedSymmetric(bool useUnsignedSymmetric)
{
    // update unsignedSymmetric flag held by Tensor Quantizer
    _useUnsignedSymmetric = useUnsignedSymmetric;
    // create new encoding analyzer instance and reset associated flags
    // _validStats is tightly coupled with the encoding analyzer instance, needs reset
    resetEncodingStats();
}

void TensorQuantizer::resetEncodingStats()
{
    _validStats     = false;
    isEncodingValid = false;

    // This is syntactic sugar provided by unique_ptr to call reset() - delete the underlying object
    _encodingAnalyzer = nullptr;
    _encodingAnalyzer = getEncodingAnalyzerInstance<float>(_quantScheme);
}

void TensorQuantizer::updateStats(const float* tensor, std::size_t tensorSize, bool useCuda)
{
    updateStats(tensor, tensorSize, useCuda, nullptr);
}

void TensorQuantizer::updateStats(const float* tensor, std::size_t tensorSize, bool useCuda, IAllocator* alloc)
{
    // Set encoding as valid
    _validStats = true;

    ComputationMode cpuGpuMode = useCuda ? ComputationMode::COMP_MODE_GPU : ComputationMode::COMP_MODE_CPU;
    _encodingAnalyzer->updateStats(tensor, tensorSize, cpuGpuMode, alloc);
}


TfEncoding TensorQuantizer::computeEncoding(unsigned int bitwidth, bool useSymmetricEncoding)
{
    TfEncoding encoding;

    if (_validStats)
    {
        encoding        = _encodingAnalyzer->computeEncoding(bitwidth, useSymmetricEncoding, _useStrictSymmetric,
                                                             _useUnsignedSymmetric);
        isEncodingValid = true;
    }

    return encoding;
}

void TensorQuantizer::computeEncodingFromData(uint8_t bw, const float* data, size_t count, TfEncoding& encoding,
                                              ComputationMode cpuGpuMode, bool useSymmetricEncodings,
                                              bool useUnsignedSymmetric, bool useStrictSymmetric)
{
    // Forget all settings except the choice of encoding analyzer
    if (encoding.delta == 0 && bw != 0)
    {
        resetEncodingStats();
        // Use data to compute min, max statistics (forget any accumulated/updated stats)
        // To avoid duplication use update stats since internal functions rely on this->_stats
        _encodingAnalyzer->updateStats(data, count, cpuGpuMode);

        encoding =
            _encodingAnalyzer->computeEncoding(bw, useSymmetricEncodings, useStrictSymmetric, useUnsignedSymmetric);
    }
    else
    {
        throw std::runtime_error("This function is only valid when encodings must be computed"
                                 " from data");
    }
}

void TensorQuantizer::quantizeDequantize(const float* input, std::size_t tensorSize, float* output, double encodingMin,
                                         double encodingMax, unsigned int bitwidth, bool useCuda)
{
    assert(isEncodingValid);
    _tensorQuantizationSim->quantizeDequantizeTensor(input, tensorSize, output, encodingMin, encodingMax, bitwidth,
                                                     roundingMode, useCuda);
}

void TensorQuantizer::quantizeTensorPacked(const float* input, std::size_t tensorSize, std::vector<uint8_t>& output,
                                           double encodingMin, double encodingMax, uint8_t bw, RoundingMode roundMode,
                                           bool useCuda, bool useStrictSymmetric)
{
    assert(isEncodingValid);
    _tensorQuantizationSim->quantizeTensorPacked(input, tensorSize, output, encodingMin, encodingMax, bw, roundMode,
                                                 useCuda, useStrictSymmetric);
}

void TensorQuantizer::quantizeDequantizePerChannelTensor(const float* input, const std::vector<uint32_t>& inputShape,
                                                         uint32_t axis, float* output,
                                                         std::vector<TfEncoding>& encodings, uint8_t bw,
                                                         RoundingMode roundMode, bool useCuda, bool useStrictSymmetric)
{
    std::vector<uint32_t> splitShape;
    std::vector<std::vector<float>> splits;

    this->setStrictSymmetric(useStrictSymmetric);   // currently we only support strict symmetric
    generatePerChannelEncodings(input, inputShape, axis, encodings, bw, splits, splitShape, useCuda);

    _tensorQuantizationSim->quantizeDequantizePerChannelTensor(splits, splitShape, axis, output, encodings, bw,
                                                               roundMode, useCuda);
}

void TensorQuantizer::quantizePerChannelTensorPacked(const float* input, const std::vector<uint32_t>& inputShape,
                                                     uint32_t axis, std::vector<uint8_t>& output,
                                                     std::vector<TfEncoding>& encodings, uint8_t bw,
                                                     RoundingMode roundMode, bool useCuda, bool useStrictSymmetric)
{
    std::vector<uint32_t> splitShape;
    std::vector<std::vector<float>> splits;

    // currently we only support strict symmetric for packed tensors
    this->setStrictSymmetric(useStrictSymmetric);
    generatePerChannelEncodings(input, inputShape, axis, encodings, bw, splits, splitShape, useCuda);
    _tensorQuantizationSim->quantizePerChannelTensorPacked(splits, splitShape, axis, output, encodings, bw, roundMode,
                                                           useCuda, useStrictSymmetric);
}

void TensorQuantizer::dequantize(const uint8_t* input, std::size_t tensorSize, double encodingMin, double encodingMax,
                                 uint8_t bw, float* output, bool useStrictSymmetric)
{
    _tensorQuantizationSim->dequantizeTensor(input, tensorSize, output, encodingMin, encodingMax, bw,
                                             useStrictSymmetric);
}

void TensorQuantizer::dequantizePerChannelTensor(const uint8_t* input, const std::vector<uint32_t>& inputShape,
                                                 uint32_t axis, const std::vector<TfEncoding>& encodings, uint8_t bw,
                                                 float* output, bool useStrictSymmetric)
{
    _tensorQuantizationSim->dequantizePerChannelTensor(input, inputShape, axis, output, bw, encodings,
                                                       useStrictSymmetric);
}

std::vector<std::tuple<double, double>> TensorQuantizer::getStatsHistogram()
{
    auto histogram = _encodingAnalyzer->getStatsHistogram();
    return histogram;
}
void TensorQuantizer::generatePerChannelEncodings(const float* input, const std::vector<uint32_t>& inputShape,
                                                  uint32_t axis, std::vector<TfEncoding>& encodings, uint32_t bw,
                                                  std::vector<std::vector<float>>& splits,
                                                  std::vector<uint32_t>& splitShape, bool useCuda)
{
    if (bw < 8)
    {
        throw std::runtime_error("Only bitwidths >= 8 supported for per-channel quantization");
    }

    if (inputShape.size() != 4)
    {
        throw std::runtime_error("Per-channel quantization only operates on 4 dimensional data!");
    }

    if (axis > 3)
    {
        throw std::runtime_error("Per-channel axis must be < 4");
    }

    if (encodings.size() != 0 && encodings.size() != inputShape[axis])
    {
        throw std::runtime_error("Must provide 0 or all encodings for per-channel quantization");
    }

    encodings.resize(inputShape[axis]);

    // Split the data by axis and perform quantization analysis
    slice(input, inputShape, axis, splits, splitShape);
    if (splits.size() != inputShape[axis])
    {
        throw std::runtime_error("Invalid slice count generated. Count must be equal to axis split on!");
    }

    uint32_t splitCount = std::accumulate(std::begin(splitShape), std::end(splitShape), 1, std::multiplies<uint32_t>());
    uint32_t outputCount =
        std::accumulate(std::begin(inputShape), std::end(inputShape), 1, std::multiplies<uint32_t>());
    if (outputCount != splitCount * splits.size())
    {
        throw std::runtime_error("Accumulated split count doesn't match original input count");
    }

    for (uint32_t i = 0; i < splits.size(); ++i)
    {
        auto& e     = encodings[i];
        auto& split = splits[i];

        if (split.size() != splitCount)
        {
            throw std::runtime_error("Tensor split size mismatch!");
        }

        if (e.bw != (double) bw)
        {
            e.bw    = (double) bw;
            e.delta = 0, e.offset = 0;
        }
        // compute encodings
        ComputationMode mode       = useCuda ? COMP_MODE_GPU : COMP_MODE_CPU;
        auto useSymmetricEncodings = _useStrictSymmetric || _useUnsignedSymmetric;

        if (!isEncodingValid)
        {
            this->computeEncodingFromData(bw, split.data(), split.size(), e, mode, useSymmetricEncodings,
                                          _useUnsignedSymmetric, _useStrictSymmetric);
        }
    }
}

void TensorQuantizer::computePartialEncoding(uint8_t bw, TfEncoding& encoding, bool useSymmetricEncodings,
                                             bool useUnsignedSymmetric, bool useStrictSymmetric)
{
    if (encoding.min == 0 && encoding.max == 0)
    {
        computeMinMaxRangeFromDeltaOffset(bw, encoding, useSymmetricEncodings, useUnsignedSymmetric, useStrictSymmetric);
    }
    else if (encoding.delta == 0)
    {
        computeDeltaAndOffsetFromMinMax(bw, encoding, useSymmetricEncodings, useUnsignedSymmetric, useStrictSymmetric);
    }
    else
    {
        throw std::runtime_error("Cannot determine how to compute partial encoding");
    }
}

}   // namespace DlQuantization
