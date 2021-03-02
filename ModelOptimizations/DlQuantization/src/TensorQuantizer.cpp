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

#include <DlQuantization/TensorQuantizer.h>
#include <cassert>
#include <pybind11/numpy.h>

namespace DlQuantization
{

TensorQuantizer::TensorQuantizer(QuantizationMode quantScheme, RoundingMode roundingMode) :
        _quantScheme(quantScheme),
        roundingMode(roundingMode),
        isEncodingValid(false),
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


void TensorQuantizer::updateStats(py::array_t<float> tensor, bool useCuda)
{
    auto npArr        = tensor.mutable_unchecked<4>();
    size_t tensorSize = npArr.shape(0) * npArr.shape(1) * npArr.shape(2) * npArr.shape(3);

    // Get a pointer to the tensor data
    auto tensorPtr = (float*) npArr.mutable_data(0, 0, 0, 0);

    // Delegate
    updateStats(tensorPtr, tensorSize, useCuda);

    _validStats = true;
}

TfEncoding TensorQuantizer::computeEncoding(unsigned int bitwidth, bool useSymmetricEncoding, bool useStrictSymmetric,
                                            bool useUnsignedSymmetric)
{
    TfEncoding encoding;

    if (_validStats)
    {
        encoding        = _encodingAnalyzer->computeEncoding(bitwidth, useSymmetricEncoding, useStrictSymmetric,
                                                      useUnsignedSymmetric);
        isEncodingValid = true;
    }

    return encoding;
}

void TensorQuantizer::quantizeDequantize(const float* input, std::size_t tensorSize, float* output,
                                         double encodingMin, double encodingMax, unsigned int bitwidth, bool useCuda)
{
    assert(isEncodingValid);
    _tensorQuantizationSim->quantizeDequantizeTensor(input, tensorSize, output, encodingMin,
            encodingMax, bitwidth, roundingMode, useCuda);
}

void TensorQuantizer::quantizeDequantize(py::array_t<float> inputTensor, py::array_t<float> outputTensor,
                                         double encodingMin, double encodingMax, unsigned int  bitwidth, bool useCuda)
{
    auto inputArr     = inputTensor.mutable_unchecked<4>();
    auto outputArr    = outputTensor.mutable_unchecked<4>();
    size_t tensorSize = inputArr.shape(0) * inputArr.shape(1) * inputArr.shape(2) * inputArr.shape(3);

    auto inputTensorPtr  = static_cast<float*>(inputArr.mutable_data(0, 0, 0, 0));
    auto outputTensorPtr = static_cast<float*>(outputArr.mutable_data(0, 0, 0, 0));

    // Delegate
    quantizeDequantize(inputTensorPtr, tensorSize, outputTensorPtr, encodingMin, encodingMax, bitwidth, useCuda);
}


}
