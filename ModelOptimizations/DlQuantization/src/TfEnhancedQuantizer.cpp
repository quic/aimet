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


#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "DlQuantization/Quantization.hpp"
#include "TfEnhancedQuantizer.hpp"
#include "TfQuantizer.hpp"

#ifdef GPU_QUANTIZATION_ENABLED

#include "cuda_util.hpp"

#endif

#include "math_functions.hpp"

namespace DlQuantization
{
using namespace std;

template <typename DTYPE>
TfEnhancedQuantizer<DTYPE>::TfEnhancedQuantizer(const vector<string>& layer_names, ComputationMode mode_cpu_gpu)
{
    m_ModeCpuGpu = mode_cpu_gpu;
    // Initialize m_StatsTfNet to contain all layers
    LayerEncodingAnalyzers empty_layer_stats;
    for (string s: layer_names)
    {
        m_StatsNet[s] = empty_layer_stats;
    }
}

template <typename DTYPE>
void TfEnhancedQuantizer<DTYPE>::UpdateStatsModeSpecific(const std::string& layer, LayerInOut mode_in_out,
                                                         const std::vector<const DTYPE*>& activations,
                                                         const std::vector<size_t>& count)
{
    if (activations.size() != count.size())
    {
        throw runtime_error("Input vector size doesn't match count vector size.");
    }
    // Fetch average stats we have so far
    if (!m_StatsNet.count(layer))
    {
        throw runtime_error("Unknown layer name: " + layer);
    }

    LayerEncodingAnalyzers& layerAnalyzers = m_StatsNet[layer];
    vector<TfEnhancedEncodingAnalyzer<DTYPE>>& pdf_average =
        (mode_in_out == LAYER_INPUT) ? layerAnalyzers.in : layerAnalyzers.out;
    // resize average stats vector if it's not initialized yet
    if (pdf_average.size() != activations.size())
    {
        pdf_average.resize(activations.size());
    }
    // iterate over all blobs of this layer
    for (unsigned int i = 0; i < activations.size(); ++i)
    {
        // compute stats for one blob
        pdf_average[i].updateStats(activations[i], count[i], m_ModeCpuGpu);
    }
}

template <typename DTYPE>
void TfEnhancedQuantizer<DTYPE>::StatsToFxpFormat(const std::string& layer, LayerInOut mode_in_out, int bw,
                                                  std::vector<TfEncoding>& encoding)
{
    if (!m_StatsNet.count(layer))
    {
        throw runtime_error("Unknown layer name: " + layer);
    }

    LayerEncodingAnalyzers& layerAnalyzers = m_StatsNet[layer];
    vector<TfEnhancedEncodingAnalyzer<DTYPE>>& analyzers =
        (mode_in_out == LAYER_INPUT) ? layerAnalyzers.in : layerAnalyzers.out;
    encoding.clear();
    for (const TfEnhancedEncodingAnalyzer<DTYPE>& analyzer: analyzers)
    {
        TfEncoding current_encoding = analyzer.computeEncoding(bw, false, false, false);
        encoding.push_back(current_encoding);
    }
}

template <typename DTYPE>
void TfEnhancedQuantizer<DTYPE>::NumberDistributionToFxpFormat(int bw, const DTYPE* data, size_t count,
                                                               TfEncoding& encoding)
{
    TfEnhancedEncodingAnalyzer<DTYPE> analyzer;

    analyzer.updateStats(data, count, m_ModeCpuGpu);
    encoding = analyzer.computeEncoding(bw, false, false, false);
}

template <typename DTYPE>
void TfEnhancedQuantizer<DTYPE>::ComputeDeltaAndOffsetModeSpecific(int bw, double& min, double& max, double& delta,
                                                                   double& offset)
{
    // Fall back to TfQuantizer.
    StatsTf stats {min, max};
    TfEncoding encoding;
    TfQuantizer<DTYPE>::MinAndMaxToFxpFormat(stats, bw, encoding);
    min    = encoding.min;
    max    = encoding.max;
    delta  = encoding.delta;
    offset = encoding.offset;
}

// Explicit instantiations
template class TfEnhancedQuantizer<double>;

template class TfEnhancedQuantizer<float>;

}   // End of namespace DlQuantization
