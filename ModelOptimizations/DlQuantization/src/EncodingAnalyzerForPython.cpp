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

#include <DlQuantization/EncodingAnalyzerForPython.h>
#include <DlQuantization/IQuantizationEncodingAnalyzer.hpp>
#include <DlQuantization/IQuantizer.hpp>
#include <DlQuantization/ITensorQuantizationSim.h>
#include <DlQuantization/Quantization.hpp>
#include <DlQuantization/QuantizerFactory.hpp>
#include <iostream>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
namespace py = pybind11;

namespace DlQuantization
{
EncodingAnalyzerForPython::EncodingAnalyzerForPython(DlQuantization::QuantizationMode quantizationScheme) :
    _quantizationScheme(quantizationScheme)
{
    _encodingAnalyzer = DlQuantization::getEncodingAnalyzerInstance<float>(quantizationScheme);
}

void EncodingAnalyzerForPython::updateStats(py::array_t<float> input, bool use_cuda)
{
    auto npArr = input.mutable_unchecked<>();

    // Set encoding as valid
    _isEncodingValid = true;

    size_t inputTensorSize = npArr.size();

    // Get a pointer to the tensor data
    auto inputDataPtr = (float*) npArr.mutable_data();

    DlQuantization::ComputationMode cpu_gpu_mode =
        use_cuda ? DlQuantization::ComputationMode::COMP_MODE_GPU : DlQuantization::ComputationMode::COMP_MODE_CPU;
    _encodingAnalyzer->updateStats(inputDataPtr, inputTensorSize, cpu_gpu_mode);
}


std::tuple<DlQuantization::TfEncoding, bool> EncodingAnalyzerForPython::computeEncoding(unsigned int bitwidth,
                                                                                        bool isSymmetric,
                                                                                        bool useStrictSymmetric,
                                                                                        bool useUnsignedSymmetric)
{
    DlQuantization::TfEncoding out_encoding;

    if (_isEncodingValid)
    {
        out_encoding = _encodingAnalyzer->computeEncoding(bitwidth, isSymmetric, useStrictSymmetric,
                                                          useUnsignedSymmetric);
    }

    return std::make_tuple(out_encoding, _isEncodingValid);
}


}