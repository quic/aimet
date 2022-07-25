//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2016-2022, Qualcomm Innovation Center, Inc. All rights reserved.
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


#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "DlQuantization/IQuantizer.hpp"
#include "DlQuantization/Quantization.hpp"
#include "DlQuantization/QuantizerFactory.hpp"
#include "EntropyEncodingAnalyzer.h"
#include "MainQuantizationClass.hpp"
#include "MseEncodingAnalyzer.h"
#include "PercentileEncodingAnalyzer.h"
#include "TensorQuantizationSim.h"
#include "TfEncodingAnalyzer.h"
#include "TfEnhancedEncodingAnalyzer.h"

namespace DlQuantization
{
template <typename DTYPE>
IQuantizer<DTYPE>* GetQuantizerInstance(const std::vector<std::string>& layer_names, ComputationMode mode_cpu_gpu,
                                        const std::vector<int>& bw_activations, QuantizationMode quantization_mode)
{
    IQuantizer<DTYPE>* instance =
        new MainQuantizationClass<DTYPE>(layer_names, mode_cpu_gpu, bw_activations, quantization_mode);

    return instance;
}

std::unique_ptr<GraphQuantizer> getGraphQuantizerInstance (const std::vector<std::string>& tensorNames,
                                                          ComputationMode modeCpuGpu,
                                                          QuantizationMode quantMode)
{
    return std::unique_ptr<GraphQuantizer>(new GraphQuantizer(tensorNames, modeCpuGpu, quantMode));
}

template <typename DTYPE>
std::unique_ptr<IQuantizationEncodingAnalyzer<DTYPE>> getEncodingAnalyzerInstance(QuantizationMode quantization_mode)
{
    if (quantization_mode == QUANTIZATION_TF_ENHANCED)
    {
        return std::unique_ptr<IQuantizationEncodingAnalyzer<DTYPE>>(new TfEnhancedEncodingAnalyzer<DTYPE>);
    }
    else if (quantization_mode == QUANTIZATION_PERCENTILE)
    {
        return std::unique_ptr<IQuantizationEncodingAnalyzer<DTYPE>>(new PercentileEncodingAnalyzer<DTYPE>);
    }
    else if (quantization_mode == QUANTIZATION_MSE)
    {
        return std::unique_ptr<IQuantizationEncodingAnalyzer<DTYPE>>(new MseEncodingAnalyzer<DTYPE>);
    }
    else if (quantization_mode == QUANTIZATION_ENTROPY)
    {
        return std::unique_ptr<IQuantizationEncodingAnalyzer<DTYPE>>(new EntropyEncodingAnalyzer<DTYPE>);
    }
    else
    {
        return std::unique_ptr<IQuantizationEncodingAnalyzer<DTYPE>>(new TfEncodingAnalyzer<DTYPE>);
    }
}


template <typename DTYPE>
std::unique_ptr<ITensorQuantizationSim<DTYPE>> getTensorQuantizationSim()
{
    return std::unique_ptr<ITensorQuantizationSim<DTYPE>>(new TensorQuantizationSim<DTYPE>());
}


// Explicit instantiations
template IQuantizer<double>* GetQuantizerInstance(const std::vector<std::string>& layer_names,
                                                  ComputationMode mode_cpu_gpu, const std::vector<int>& bw_activations,
                                                  QuantizationMode quantization_mode);

template IQuantizer<float>* GetQuantizerInstance(const std::vector<std::string>& layer_names,
                                                 ComputationMode mode_cpu_gpu, const std::vector<int>& bw_activations,
                                                 QuantizationMode quantization_mode);

template std::unique_ptr<IQuantizationEncodingAnalyzer<float>>
getEncodingAnalyzerInstance(QuantizationMode quantization_mode);
template std::unique_ptr<IQuantizationEncodingAnalyzer<double>>
getEncodingAnalyzerInstance(QuantizationMode quantization_mode);

template std::unique_ptr<ITensorQuantizationSim<float>> getTensorQuantizationSim();
template std::unique_ptr<ITensorQuantizationSim<double>> getTensorQuantizationSim();

}   // End of namespace DlQuantization
