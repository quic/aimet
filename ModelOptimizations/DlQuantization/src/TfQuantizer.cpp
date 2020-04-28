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


#include <math.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "DlQuantization/Quantization.hpp"
#include "TfQuantizer.hpp"
#include "math_functions.hpp"

namespace DlQuantization
{
using namespace std;

template <typename DTYPE>
TfQuantizer<DTYPE>::TfQuantizer(const vector<string>& layer_names, ComputationMode mode_cpu_gpu)
{
    m_LayerNames = layer_names;
    m_ModeCpuGpu = mode_cpu_gpu;
    // Initialize m_StatsTfNet to contain all layers
    StatsLayerTf empty_layer_stat;
    for (vector<string>::iterator layer_name = m_LayerNames.begin(); layer_name != m_LayerNames.end(); ++layer_name)
    {
        m_StatsTfNet[*layer_name] = empty_layer_stat;
    }
}

template <typename DTYPE>
void TfQuantizer<DTYPE>::UpdateStatsModeSpecific(const std::string& layer, LayerInOut mode_in_out,
                                                 const std::vector<const DTYPE*>& activations,
                                                 const std::vector<size_t>& count)
{
    if (activations.size() != count.size())
    {
        throw runtime_error("Input vector size doesn't match count vector size.");
    }
    // Fetch average stats we have so far
    if (!m_StatsTfNet.count(layer))
    {
        throw runtime_error("Unknown layer name: " + layer);
    }
    StatsLayerTf& stats_tmp        = m_StatsTfNet[layer];
    vector<StatsTf>& stats_average = (mode_in_out == LAYER_INPUT) ? stats_tmp.in : stats_tmp.out;
    // resize average stats vector if it's not initialized yet
    if (stats_average.size() != activations.size())
    {
        stats_average.resize(activations.size());
        for (StatsTf& s: stats_average)
        {
            s.min = std::numeric_limits<double>::max();
            s.max = -std::numeric_limits<double>::max();
        }
    }
    // iterate over all blobs of this layer
    for (unsigned int i = 0; i < activations.size(); ++i)
    {
        // compute stats for one blob
        double min_val = (double) GetMin(activations[i], count[i], m_ModeCpuGpu);
        double max_val = (double) GetMax(activations[i], count[i], m_ModeCpuGpu);
        // average stats
        stats_average[i].min = min(stats_average[i].min, min_val);
        stats_average[i].max = max(stats_average[i].max, max_val);
    }
}

template <typename DTYPE>
void TfQuantizer<DTYPE>::StatsToFxpFormat(const std::string& layer, LayerInOut mode_in_out, int bw,
                                          std::vector<TfEncoding>& encoding)
{
    if (!m_StatsTfNet.count(layer))
    {
        throw runtime_error("Unknown layer name: " + layer);
    }
    StatsLayerTf tmp             = m_StatsTfNet[layer];
    vector<StatsTf> stats_vector = (mode_in_out == LAYER_INPUT) ? tmp.in : tmp.out;
    encoding.clear();
    TfEncoding encoding_tmp;
    for (vector<StatsTf>::iterator stats = stats_vector.begin(); stats != stats_vector.end(); ++stats)
    {
        MinAndMaxToFxpFormat(*stats, bw, encoding_tmp);
        encoding.push_back(encoding_tmp);
    }
}

template <typename DTYPE>
void TfQuantizer<DTYPE>::NumberDistributionToFxpFormat(int bw, const DTYPE* data, size_t count, TfEncoding& encoding)
{
    StatsTf stats;
    stats.max = (double) GetMax(data, count, m_ModeCpuGpu);
    stats.min = (double) GetMin(data, count, m_ModeCpuGpu);
    MinAndMaxToFxpFormat(stats, bw, encoding);
}

template <typename DTYPE>
void TfQuantizer<DTYPE>::MinAndMaxToFxpFormat(const StatsTf& stats, int bw, TfEncoding& encoding)
{
    double num_steps = pow(2, bw) - 1;
    // Make sure zero value is within the range
    double new_min = std::min(0.0, stats.min);
    double new_max = std::max(0.0, stats.max);

    // When the min and max are too close together, nudge the maximum to meet the
    // minimum range requirement
    // This also handles the case where min==max==0 to avoid division by zero
    new_max = std::max(new_max, new_min + MIN_RANGE);

    encoding.delta = (new_max - new_min) / num_steps;
    if (new_min < 0 && new_max > 0)
    {
        // Need to make sure 0-value is exactly quantizable
        // Quantization of q into b is given by:
        //     b = q / delta - offset, where
        //                             delta = (max - min)/#steps
        //                             offset = min / delta
        // For q = 0: b = -min / delta
        // Find the closest round b, and set q=0 for it
        double b_zero   = round(-new_min / encoding.delta);
        b_zero          = std::min(num_steps, std::max(0.0, b_zero));   // just to be safe
        encoding.offset = -b_zero;
    }
    else
    {
        // One of min or max is guaranteed to be zero, so 0 is exactly quantizable already
        encoding.offset = round(new_min / encoding.delta);
    }

    // Calculate 'min' and 'max' based on 'delta' and 'offset'.
    // Note this min and max can vary from the one in 'stats'. This min and max
    // can really be represented with the integer offset.
    encoding.min = encoding.delta * encoding.offset;
    // We want to calculate: max = delta * num_steps + min.
    // To avoid numerical accuracy issues on Linaro, we simplify the math.
    encoding.max = new_max - new_min + encoding.min;
    encoding.bw  = bw;
}

template <typename DTYPE>
void TfQuantizer<DTYPE>::ComputeDeltaAndOffsetModeSpecific(int bw, double& min, double& max, double& delta,
                                                           double& offset)
{
    StatsTf stats {min, max};
    TfEncoding encoding;
    MinAndMaxToFxpFormat(stats, bw, encoding);
    min    = encoding.min;
    max    = encoding.max;
    delta  = encoding.delta;
    offset = encoding.offset;
}

// Explicit instantiations
template class TfQuantizer<double>;

template class TfQuantizer<float>;

}   // End of namespace DlQuantization
