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


#include <cstdint>
#include <map>
#include <math.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "DlQuantization/Quantization.hpp"
#include "MainQuantizationClass.hpp"
#include "TfEnhancedQuantizer.hpp"
#include "TfQuantizer.hpp"
#include "trim_functions.hpp"

namespace DlQuantization
{
using namespace std;

template <typename DTYPE>
MainQuantizationClass<DTYPE>::MainQuantizationClass(const vector<string>& layer_names, ComputationMode mode_cpu_gpu,
                                                    const vector<int>& bw_activations,
                                                    QuantizationMode quantization_mode)
{
    m_LayerNames       = layer_names;
    m_ModeCpuGpu       = mode_cpu_gpu;
    m_FxpFormatSource  = FxpFormatSource::UNDEFINED;
    m_QuantizationMode = quantization_mode;
    switch (m_QuantizationMode)
    {
    case QUANTIZATION_TF:
        m_QuantAlgo = shared_ptr<TfQuantizer<DTYPE> >(new TfQuantizer<DTYPE>(layer_names, mode_cpu_gpu));
        break;
    case QUANTIZATION_TF_ENHANCED:
        m_QuantAlgo =
            shared_ptr<TfEnhancedQuantizer<DTYPE> >(new TfEnhancedQuantizer<DTYPE>(layer_names, mode_cpu_gpu));
        break;
    default:
        throw std::runtime_error("Unknown quantization mode");
        break;
    }
    // TODO-PGYSEL: remove this unnecessary argument in the quantizer factory.
    (void) bw_activations;
}

template <typename DTYPE>
void MainQuantizationClass<DTYPE>::UpdateStats(const string& layer, LayerInOut mode_in_out,
                                               const vector<const DTYPE*>& input, const vector<size_t>& count)
{
    switch (m_FxpFormatSource)
    {
    case FxpFormatSource::UNDEFINED:
    case FxpFormatSource::STATS:
        m_FxpFormatSource = FxpFormatSource::STATS;
        m_QuantAlgo->UpdateStatsModeSpecific(layer, mode_in_out, input, count);
        break;
    case FxpFormatSource::EXTERNAL:
        throw runtime_error("State mismatch: Can't use SetEncoding AND "
                            "UpdateStats.");
        break;
    default:
        throw runtime_error("Unknown fixed point format source.");
        break;
    }
}

template <typename DTYPE>
void MainQuantizationClass<DTYPE>::QuantizeDequantizeActs(const string& layer, LayerInOut mode_in_out, int bw,
                                                          vector<DTYPE*>& acts, const vector<size_t>& count,
                                                          vector<DTYPE*>& acts_quantized, vector<TfEncoding>& encoding)
{
    if (acts.size() != count.size())
    {
        throw runtime_error("Input vector size has to match count vector size.");
    }
    GetEncodingFromStatsOrExternal(layer, mode_in_out, encoding, bw);
    // Call quantization routine
    for (unsigned int blob_id = 0; blob_id < acts.size(); ++blob_id)
    {
        quantizeDequantize(acts[blob_id], count[blob_id], encoding[blob_id], acts_quantized[blob_id], m_ModeCpuGpu,
                           ROUND_NEAREST);
    }
}

template <typename DTYPE>
void MainQuantizationClass<DTYPE>::QuantizeDequantizeParams(int bw, DTYPE* params, size_t count,
                                                            RoundingMode mode_rounding, DTYPE* params_quantized,
                                                            TfEncoding& encoding)
{
    m_QuantAlgo->NumberDistributionToFxpFormat(bw, params, count, encoding);
    quantizeDequantize(params, count, encoding, params_quantized, m_ModeCpuGpu, mode_rounding);
}

template <typename DTYPE>
void MainQuantizationClass<DTYPE>::SetEncoding(const map<string, TfEncodingLayer>& stats)
{
    switch (m_FxpFormatSource)
    {
    case FxpFormatSource::UNDEFINED:
    case FxpFormatSource::EXTERNAL:
        m_FxpFormatSource = FxpFormatSource::EXTERNAL;
        m_TfEncodingNet   = stats;
        break;
    case FxpFormatSource::STATS:
        throw runtime_error("State mismatch: Can't use SetEncoding AND "
                            "UpdateStats.");
        break;
    default:
        throw runtime_error("Unknown fixed point format source.");
        break;
    }
}

template <typename DTYPE>
void MainQuantizationClass<DTYPE>::SetEncoding(const std::string& layer, TfEncodingLayer& encoding)
{
    switch (m_FxpFormatSource)
    {
    case FxpFormatSource::UNDEFINED:
    case FxpFormatSource::EXTERNAL:
        m_FxpFormatSource      = FxpFormatSource::EXTERNAL;
        m_TfEncodingNet[layer] = encoding;
        break;
    case FxpFormatSource::STATS:
        throw runtime_error("State mismatch: Can't use SetEncoding AND "
                            "UpdateStats.");
        break;
    default:
        throw runtime_error("Unknown fixed point format source.");
        break;
    }
}

template <typename DTYPE>
void MainQuantizationClass<DTYPE>::GetEncoding(const map<string, int>& bws, map<string, TfEncodingLayer>& stats)
{
    switch (m_FxpFormatSource)
    {
    case FxpFormatSource::EXTERNAL:
        stats = m_TfEncodingNet;
        break;
    case FxpFormatSource::STATS:
        stats.clear();
        for (unsigned int layer_id = 0; layer_id < m_LayerNames.size(); ++layer_id)
        {
            const string layer_name = m_LayerNames[layer_id];
            if (!bws.count(layer_name))
            {
                throw runtime_error("Unknown layer name: " + layer_name);
            }
            int bw = bws.find(layer_name)->second;
            TfEncodingLayer layer_encoding;
            m_QuantAlgo->StatsToFxpFormat(layer_name, LAYER_INPUT, bw, layer_encoding.in);
            m_QuantAlgo->StatsToFxpFormat(layer_name, LAYER_OUTPUT, bw, layer_encoding.out);
            stats[layer_name] = layer_encoding;
        }
        break;
    case FxpFormatSource::UNDEFINED:
        throw runtime_error("State mismatch: Use SetEncoding OR UpdateStats "
                            "first.");
    default:
        throw runtime_error("Unknown fixed point format source.");
        break;
    }
}

template <typename DTYPE>
void MainQuantizationClass<DTYPE>::GetEncoding(const string& layer, unsigned int bw, TfEncodingLayer& encoding)
{
    switch (m_FxpFormatSource)
    {
    case FxpFormatSource::EXTERNAL:
        if (!m_TfEncodingNet.count(layer))
        {
            throw runtime_error("Unknown layer name: " + layer);
        }
        encoding = m_TfEncodingNet[layer];
        break;
    case FxpFormatSource::STATS:
    {
        m_QuantAlgo->StatsToFxpFormat(layer, LAYER_INPUT, bw, encoding.in);
        m_QuantAlgo->StatsToFxpFormat(layer, LAYER_OUTPUT, bw, encoding.out);
        break;
    }
    case FxpFormatSource::UNDEFINED:
        throw runtime_error("State mismatch: Use SetEncoding OR UpdateStats "
                            "first.");
        break;
    default:
        throw runtime_error("Unknown fixed point format source.");
        break;
    }
}

template <typename DTYPE>
void MainQuantizationClass<DTYPE>::GetAccumulatorFormat(const TfEncoding& input_acts, const TfEncoding& weights,
                                                        TfEncoding& accumulator)
{
    accumulator.delta  = input_acts.delta * weights.delta;
    accumulator.offset = 0;
    accumulator.min    = accumulator.delta * std::numeric_limits<double>::lowest();
    accumulator.max    = accumulator.delta * std::numeric_limits<double>::max();
    accumulator.bw     = 32;
}

template <typename DTYPE>
void MainQuantizationClass<DTYPE>::ComputeDeltaAndOffset(int bw, double& min, double& max, double& delta,
                                                         double& offset)
{
    m_QuantAlgo->ComputeDeltaAndOffsetModeSpecific(bw, min, max, delta, offset);
}

template <typename DTYPE>
void MainQuantizationClass<DTYPE>::GetEncodingFromStatsOrExternal(const string& layer, LayerInOut mode_in_out,
                                                                  vector<TfEncoding>& encoding, int bw)
{
    switch (m_FxpFormatSource)
    {
    case FxpFormatSource::EXTERNAL:
    {
        if (0 == m_TfEncodingNet.count(layer))
        {
            throw runtime_error("Unknown layer name: " + layer);
        }
        TfEncodingLayer tmp = m_TfEncodingNet[layer];
        encoding            = (mode_in_out == LAYER_INPUT) ? tmp.in : tmp.out;
        break;
    }
    case FxpFormatSource::STATS:
        m_QuantAlgo->StatsToFxpFormat(layer, mode_in_out, bw, encoding);
        break;
    case FxpFormatSource::UNDEFINED:
        throw runtime_error("State mismatch: need to call UpdateStats or "
                            "SetEncoding first.");
    default:
        throw runtime_error("Unknown fixed point format source");
        break;
    }
}

// Explicit instantiations
template class MainQuantizationClass<double>;

template class MainQuantizationClass<float>;

}   // End of namespace DlQuantization
