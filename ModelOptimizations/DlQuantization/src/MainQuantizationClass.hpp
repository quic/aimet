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


#ifndef MAIN_QUANTIZATION_CLASS_HPP
#define MAIN_QUANTIZATION_CLASS_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "DlQuantization/IQuantizer.hpp"
#include "DlQuantization/Quantization.hpp"
#include "IQuantizationAlgorithm.hpp"

namespace DlQuantization
{
enum class FxpFormatSource : char
{
    UNDEFINED,
    EXTERNAL,
    STATS
};

template <typename DTYPE>
class MainQuantizationClass final : public IQuantizer<DTYPE>
{
public:
    MainQuantizationClass(const std::vector<std::string>& layer_names, ComputationMode mode_cpu_gpu,
                          const std::vector<int>& bw_activations, QuantizationMode m_QuantizationMode);

    // As input we get the input OR output activations of a layer. Calculate
    // the statistical data using the subclass's stats format, and do the
    // averaging. Note some layer can have input from multiple CNN layers, which
    // is why the activations are in a vector.
    virtual void UpdateStats(const std::string& layer, LayerInOut mode_in_out,
                             const std::vector<const DTYPE*>& activations, const std::vector<size_t>& count) override;

    virtual void QuantizeDequantizeActs(const std::string& layer, LayerInOut mode_in_out, int bw,
                                        std::vector<DTYPE*>& acts, const std::vector<size_t>& count,
                                        std::vector<DTYPE*>& acts_quantized,
                                        std::vector<TfEncoding>& encoding) override;

    virtual void QuantizeDequantizeParams(int bw, DTYPE* params, size_t count, RoundingMode mode_rounding,
                                          DTYPE* params_quantized, TfEncoding& encoding) override;

    // Set encoding of activations using TF format
    virtual void SetEncoding(const std::map<std::string, TfEncodingLayer>& stats) override;

    virtual void SetEncoding(const std::string& layer, TfEncodingLayer& encoding) override;

    // Get encoding of activations. If in 'generate stats mode', convert the
    // IQantizationAlgorithm's stats into the TF format.
    virtual void GetEncoding(const std::map<std::string, int>& bws,
                             std::map<std::string, TfEncodingLayer>& stats) override;

    virtual void GetEncoding(const std::string& layer, unsigned int bw, TfEncodingLayer& encoding) override;

    virtual void GetAccumulatorFormat(const TfEncoding& input_acts, const TfEncoding& weights,
                                      TfEncoding& accumulator) override;

    virtual void ComputeDeltaAndOffset(int bw, double& min, double& max, double& delta, double& offset) override;

    virtual ~MainQuantizationClass() = default;

private:
    ComputationMode m_ModeCpuGpu;
    // the TF encodings. those are only set by SetEncoding. those are only used
    // in EXTERNAL
    std::map<std::string, TfEncodingLayer> m_TfEncodingNet;
    // We have two modes. Either the TF encoding is set from externally. Or we
    // keep stats for the activations and infer the TF encoding from the stats.
    FxpFormatSource m_FxpFormatSource;
    std::vector<std::string> m_LayerNames;
    QuantizationMode m_QuantizationMode;
    std::shared_ptr<IQuantizationAlgorithm<DTYPE> > m_QuantAlgo;

    // Get fixed point encoding. Two possible sources: 1) Encoding was set by
    // user with SetEncoding(). 2) Encoding can be computed from statistical data
    // using IQuantizationAlgorithm object.
    void GetEncodingFromStatsOrExternal(const std::string& layer, LayerInOut mode_in_out,
                                        std::vector<TfEncoding>& encoding, int bw);
};

}   // End of namespace DlQuantization

#endif   // MAIN_QUANTIZATION_CLASS_HPP
