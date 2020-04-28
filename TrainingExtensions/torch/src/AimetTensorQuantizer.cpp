//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

#include <DlQuantization/IQuantizationEncodingAnalyzer.hpp>
#include <DlQuantization/IQuantizer.hpp>
#include <DlQuantization/ITensorQuantizationSim.h>
#include <DlQuantization/Quantization.hpp>
#include <DlQuantization/QuantizerFactory.hpp>

#include <iostream>
#include <string>
#include <torch/extension.h>
#include <vector>

class AimetTensorQuantizer
{
public:
    AimetTensorQuantizer(DlQuantization::QuantizationMode quantizationScheme) :
        _isEncodingValid(false),
        _quantizationScheme(quantizationScheme)
    {
        _encodingAnalyzer      = DlQuantization::getEncodingAnalyzerInstance<float>(quantizationScheme);
        _tensorQuantizationSim = DlQuantization::getTensorQuantizationSim<float>();
    }

    void resetEncodingStats()
    {
        _isEncodingValid = false;

        // This is syntactic sugar provided by unique_ptr to call reset() - delete the underlying object
        _encodingAnalyzer = nullptr;
        _encodingAnalyzer = DlQuantization::getEncodingAnalyzerInstance<float>(_quantizationScheme);
    }

    void updateStats(at::Tensor input, bool use_cuda)
    {
        // Set encoding as valid
        _isEncodingValid = true;

        at::IntArrayRef sizes  = input.sizes();
        size_t inputTensorSize = 1;
        for (auto size: sizes)
            inputTensorSize *= size;

        // Get a pointer to the tensor data
        float* inputDataPtr = input.data<float>();

        DlQuantization::ComputationMode cpu_gpu_mode =
            use_cuda ? DlQuantization::ComputationMode::COMP_MODE_GPU : DlQuantization::ComputationMode::COMP_MODE_CPU;
        _encodingAnalyzer->updateStats(inputDataPtr, inputTensorSize, cpu_gpu_mode);
    }


    at::Tensor quantizeDequantize(at::Tensor input, DlQuantization::TfEncoding& encoding,
                                  DlQuantization::RoundingMode roundingMode, bool use_cuda)
    {
        // Allocate an output tensor as the same shape as the input
        at::Tensor output = input;

        at::IntArrayRef sizes  = input.sizes();
        size_t inputTensorSize = 1;
        for (auto size: sizes)
            inputTensorSize *= size;

        _tensorQuantizationSim->quantizeDequantizeTensor(input.data<float>(), inputTensorSize, output.data<float>(),
                                                         encoding.min, encoding.max, encoding.bw, roundingMode, use_cuda);

        return output;
    }

    std::tuple<DlQuantization::TfEncoding, bool> getEncoding(unsigned int bitwidth, bool useSymmetricEncodings)
    {
        DlQuantization::TfEncoding out_encoding;

        if (_isEncodingValid)
        {
            out_encoding = _encodingAnalyzer->computeEncoding(bitwidth, useSymmetricEncodings);
        }

        return std::make_tuple(out_encoding, _isEncodingValid);
    }

private:
    bool _isEncodingValid;
    DlQuantization::QuantizationMode _quantizationScheme;
    std::unique_ptr<DlQuantization::IQuantizationEncodingAnalyzer<float>> _encodingAnalyzer;
    std::unique_ptr<DlQuantization::ITensorQuantizationSim<float>> _tensorQuantizationSim;
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::class_<AimetTensorQuantizer>(m, "AimetTensorQuantizer")
        .def(pybind11::init<DlQuantization::QuantizationMode>())
        .def("updateStats", &AimetTensorQuantizer::updateStats)
        .def("quantizeDequantize", &AimetTensorQuantizer::quantizeDequantize)
        .def("getEncoding", &AimetTensorQuantizer::getEncoding)
        .def("resetEncodingStats", &AimetTensorQuantizer::resetEncodingStats);
}