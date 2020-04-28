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


#ifndef TF_QUANTIZER_HPP
#define TF_QUANTIZER_HPP

#include <limits>
#include <map>
#include <string>
#include <vector>

#include "DlQuantization/Quantization.hpp"
#include "IQuantizationAlgorithm.hpp"

namespace DlQuantization
{
// The TF-style statistics for one given data blob.
struct StatsTf
{
    double min;
    double max;
};

// The TF-style statistics for all input and output blobs of a network layer.
struct StatsLayerTf
{
    std::vector<StatsTf> in;
    std::vector<StatsTf> out;
};

template <typename DTYPE>
class TfQuantizer final : public IQuantizationAlgorithm<DTYPE>
{
public:
    TfQuantizer(const std::vector<std::string>& layer_names, ComputationMode mode_cpu_gpu);

    // generate stats
    // average stats into existing stats
    virtual void UpdateStatsModeSpecific(const std::string& layer, LayerInOut mode_in_out,
                                         const std::vector<const DTYPE*>& activations,
                                         const std::vector<size_t>& count) override;

    // fetch mode specific stats: max and min of given layer activation vector
    // convert stats to encoding with StatsToFxpFormat
    virtual void StatsToFxpFormat(const std::string& layer, LayerInOut mode_in_out, int bw,
                                  std::vector<TfEncoding>& encoding) override;

    // generate stats
    // convert stats to encoding with StatsToFxpFormat
    virtual void NumberDistributionToFxpFormat(int bw, const DTYPE* data, size_t count, TfEncoding& encoding) override;

    virtual void ComputeDeltaAndOffsetModeSpecific(int bw, double& min, double& max, double& delta,
                                                   double& offset) override;

    // Take stats (in this case max and min) and generate TfEncoding
    // The encoding has a rounded offset, and the value zero can be represented
    // precisely with this encoding.
    static void MinAndMaxToFxpFormat(const StatsTf& stats, int bw, TfEncoding& encoding);

    // Minimum range of quantization
    static constexpr double MIN_RANGE = 0.01;

private:
    std::map<std::string, StatsLayerTf> m_StatsTfNet;
    std::vector<std::string> m_LayerNames;
    ComputationMode m_ModeCpuGpu;
};

}   // End of namespace DlQuantization

#endif   // TF_QUANTIZER_HPP
