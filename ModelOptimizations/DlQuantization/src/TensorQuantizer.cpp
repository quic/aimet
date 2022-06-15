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

#include <DlQuantization/TensorQuantizer.h>
#include <cassert>
#include <pybind11/numpy.h>

namespace DlQuantization
{

TensorQuantizer::TensorQuantizer(QuantizationMode quantScheme, RoundingMode roundingMode) :
        _quantScheme(quantScheme),
        roundingMode(roundingMode),
        isEncodingValid(false),
        _useStrictSymmetric(false),
        _useUnsignedSymmetric(true),
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
    // Set encoding as valid
    _validStats = true;

    ComputationMode cpuGpuMode = useCuda ? ComputationMode::COMP_MODE_GPU : ComputationMode::COMP_MODE_CPU;
    _encodingAnalyzer->updateStats(tensor, tensorSize, cpuGpuMode);
}


TfEncoding TensorQuantizer::computeEncoding(unsigned int bitwidth, bool useSymmetricEncoding)
{
    TfEncoding encoding;

    if (_validStats)
    {
        encoding = _encodingAnalyzer->computeEncoding(bitwidth, useSymmetricEncoding,
                _useStrictSymmetric, _useUnsignedSymmetric);
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
        // Use data to compute min, max statistics (forget any accumulated/updated stats)ze
        // To avoid duplication use update stats since internal functions rely on this->_stats
        _encodingAnalyzer->updateStats(data, count, cpuGpuMode);

       encoding = _encodingAnalyzer->computeEncoding(bw, useSymmetricEncodings, useStrictSymmetric, useUnsignedSymmetric);
    }
    else {
        throw std::runtime_error("This function is only valid when encodings must be computed"
                                 " from data");
    }
}

void TensorQuantizer::quantizeDequantize(const float* input, std::size_t tensorSize, float* output,
                                         double encodingMin, double encodingMax, unsigned int bitwidth, bool useCuda)
{
    assert(isEncodingValid);
    _tensorQuantizationSim->quantizeDequantizeTensor(input, tensorSize, output, encodingMin,
            encodingMax, bitwidth, roundingMode, useCuda);
}

std::vector<std::tuple<double, double>> TensorQuantizer::getStatsHistogram()
{
    auto histogram = _encodingAnalyzer->getStatsHistogram();
    return histogram;
}

}
