//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2019 - 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

#ifndef DL_QUANTIZATION_TF_ENCODING_ANALYZER_H
#define DL_QUANTIZATION_TF_ENCODING_ANALYZER_H

// This file contains code to analyze and calculate quantization encodings
// This code is specific for the TF quantization scheme

#include "math_functions.hpp"
#include <DlQuantization/IQuantizationEncodingAnalyzer.hpp>

namespace DlQuantization
{
template <typename DTYPE>
class TfEncodingAnalyzer : public IQuantizationEncodingAnalyzer<DTYPE>
{
public:
    void updateStats(const DTYPE* tensor, const size_t tensorSize, ComputationMode tensorCpuGpuMode) override;

    void updateStats(const DTYPE* tensor, const size_t tensorSize,
                     ComputationMode tensorCpuGpuMode, IAllocator* allocator) override;

    /**
     * @brief Given a number distribution in CPU memory, compute the TensorFlow
     * encoding with the highest possible SQNR.
     *
     * To do so, we perform a grid search over different deltas and offsets.
     * This grid search optimizes the encoding to reduce the cost of quantization.
     * In this cost function, saturation errors are weighted higher than
     * quantization errors.
     */
    TfEncoding computeEncoding(uint8_t bw, bool useSymmetricEncodings,
                               bool useStrictSymmetric, bool useUnsignedSymmetric) const override;

    /**
     * @brief Returns a histogram that represents a PDF of tensor values seen by this encoding analyzer so far
     *
     * @return Histogram of statistics. The histogram returned is a vector of buckets. Each bucket is a tuple of
     * two values - the float value representing the left edge of the bucket and a PDF of the values in this bucket
     * relative to all the values seen across all buckets
     */
    std::vector<std::tuple<double, double>> getStatsHistogram() const override;

    // Minimum range of quantization
    static constexpr double MIN_RANGE = 0.01;


private:
    bool _statsUpdated = false;
    struct
    {
        double min = std::numeric_limits<double>::max();
        double max = -std::numeric_limits<double>::max();

    } _accumulatedStats;
};

}   // namespace DlQuantization

#endif   // DL_QUANTIZATION_TF_ENCODING_ANALYZER_H
