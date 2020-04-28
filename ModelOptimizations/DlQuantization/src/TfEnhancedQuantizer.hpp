//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2016-2017, Qualcomm Innovation Center, Inc. All rights reserved.
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


#ifndef TF_ENHANCED_QUANTIZER_HPP
#define TF_ENHANCED_QUANTIZER_HPP

#include <map>
#include <string>
#include <vector>

#include "DlQuantization/Quantization.hpp"
#include "IQuantizationAlgorithm.hpp"
#include "TfEnhancedEncodingAnalyzer.h"
#include "math_functions.hpp"

namespace DlQuantization
{
/**
 * @brief Enhanced TensorFlow quantization.
 *
 * This is a TensorFlow quantization scheme where delta can be any positive real
 * number and the offset is an integer in range [-255,0]. Therefor zero can be
 * represented precisely.
 * To choose an encoding, we gather a probability density function for all
 * activation and weight tensors. Then we perform a search over possible
 * encodings to find the one which yields the highest SQNR. For activations, we
 * average the probability density function over multiple batches.
 */
template <typename DTYPE>
class TfEnhancedQuantizer final : public IQuantizationAlgorithm<DTYPE>
{
public:
    TfEnhancedQuantizer(const std::vector<std::string>& layer_names, ComputationMode mode_cpu_gpu);

    /**
     * @brief Capture the probability density of a number distribution.
     */
    virtual void UpdateStatsModeSpecific(const std::string& layer, LayerInOut mode_in_out,
                                         const std::vector<const DTYPE*>& activations,
                                         const std::vector<size_t>& count) override;

    /**
     * @brief Using the probability density function, search for the encoding with
     * the lowest quantization error.
     */
    virtual void StatsToFxpFormat(const std::string& layer, LayerInOut mode_in_out, int bw,
                                  std::vector<TfEncoding>& encoding) override;

    /**
     * @brief Generate a probability density function and convert it into an
     * encoding with PdfToTfFxp.
     */
    virtual void NumberDistributionToFxpFormat(int bw, const DTYPE* data, size_t count, TfEncoding& encoding) override;

    virtual void ComputeDeltaAndOffsetModeSpecific(int bw, double& min, double& max, double& delta,
                                                   double& offset) override;

private:
    struct LayerEncodingAnalyzers
    {
        std::vector<TfEnhancedEncodingAnalyzer<DTYPE>> in;
        std::vector<TfEnhancedEncodingAnalyzer<DTYPE>> out;
    };

    // The probability density functions of all activation tensors in the network.
    std::map<std::string, LayerEncodingAnalyzers> m_StatsNet;
    ComputationMode m_ModeCpuGpu;

    // Fudge factor which trades-off quantization and saturation error.
    // The cost function will be "quantization cost" + GAMMA * "saturation cost".
    static constexpr DTYPE GAMMA = 3.0;
    // Minimum range of quantization
    static constexpr double MIN_RANGE = 0.01;
};

}   // End of namespace DlQuantization

#endif   // TF_ENHANCED_QUANTIZER_HPP
