//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>


#include "DlQuantization/Quantization.hpp"
#include "math_functions.hpp"
#include "quantization_utils.hpp"

#include "PercentileEncodingAnalyzer.h"

namespace DlQuantization
{
template <typename DTYPE>
std::vector<std::tuple<double, double>> PercentileEncodingAnalyzer<DTYPE>::getStatsHistogram() const
{
    // Return the collected histogram data.
    return getCollectedHistogram(this->_stats);
}

template <typename DTYPE>
void PercentileEncodingAnalyzer<DTYPE>::updateStats(const DTYPE* tensor, const size_t tensorSize,
                                                    ComputationMode tensorCpuGpuMode)
{
    updateStats(tensor, tensorSize, tensorCpuGpuMode, nullptr);
}

template <typename DTYPE>
void PercentileEncodingAnalyzer<DTYPE>::updateStats(const DTYPE* tensor, const size_t tensorSize,
                                                    ComputationMode tensorCpuGpuMode,
                                                    IAllocator* allocator)
{
    this->_statsUpdated = true;

    // update pdf
    UpdatePdf(tensor, tensorSize, tensorCpuGpuMode, true, this->_stats, allocator);
}

template <typename DTYPE>
TfEncoding PercentileEncodingAnalyzer<DTYPE>::computeEncoding(uint8_t bw, bool useSymmetricEncodings,
                                                              bool useStrictSymmetric, bool useUnsignedSymmetric) const
{
    TfEncoding encoding = {0, 0, 0, 0, 0};
    DTYPE numSteps      = pow(2, bw) - 1;

    // For strict symmetric mode, we make even number of buckets
    if (useSymmetricEncodings && useStrictSymmetric)
    {
        numSteps -= 1;
    }

    if (this->_stats.xLeft.size() == 0)
    {
        if (this->_statsUpdated)
        {
            // Histogram has not been initialized yet, we have seen all zero data
            // We generate a valid encoding that covers float 0
            encoding.min    = -1;
            encoding.max    = 1;
            encoding.delta  = (encoding.max - encoding.min) / int(numSteps);
            encoding.offset = floor(encoding.min / encoding.delta);
            encoding.min    = encoding.offset * encoding.delta;
            encoding.max    = encoding.min + int(numSteps) * encoding.delta;
            encoding.bw     = bw;

            return encoding;
        }
        else
        {
            // Histogram has not been initialized yet because we have not seen any data
            // We return a zero encoding - which is a failure indicator
            return encoding;
        }
    }

    // Find the adjusted min and max
    DTYPE aMin, aMax;
    std::tie(aMin, aMax) = _computePercentileRange();

    // After Min and Max adjustment, the requirement that 0 be an exactly
    // representable value must be met.
    // There is a possibility that 0 may not be present in the Percentile
    // calibrated Min and Max range. Hence, extend the interval
    // [aMin, aMax] to ensure that it contains 0.
    aMin = std::min(aMin, DTYPE(0.f));
    aMax = std::max(aMax, DTYPE(0.f));

    assert(aMin <= aMax && "min must not be bigger than max");

    return getComputedEncodings(bw, aMin, aMax, useSymmetricEncodings, useStrictSymmetric, useUnsignedSymmetric);
}

template <typename DTYPE>
std::tuple<DTYPE, DTYPE> PercentileEncodingAnalyzer<DTYPE>::_computePercentileRange() const
{
    // Number of histogram bins.
    const int numBins = PDF_SIZE;

    // Find the range of our collected stats
    DTYPE minVal, maxVal;
    std::tie(minVal, maxVal) = _findRangeOfAggregateStats();

    // Incase of percenitle value of 100 no need of calibration.
    if (this->_percentile == 100.0f)
    {
        return std::tuple<DTYPE, DTYPE>(minVal, maxVal);
    }

    const float histBinWidth = this->_stats.xLeft[1] - this->_stats.xLeft[0];
    DTYPE histMin            = this->_stats.xLeft[0];
    DTYPE histMax            = this->_stats.xLeft[PDF_SIZE - 1] + histBinWidth;

    DTYPE percentileMin = histMin;
    DTYPE percentileMax = histMax;

    // Copy the pdf collected and compute cdf.
    std::vector<double> cdf(this->_stats.pdf);
    for (auto i = 1; i < cdf.size(); i++)
    {
        cdf[i] += cdf[i - 1];
    }

    // Compute percentile calibration Min.
    float leftPercentile = 1 - this->_percentile / 100;
    for (auto i = 0; i < numBins; i++)
    {
        if (cdf[i] >= leftPercentile)
        {
            percentileMin = this->_stats.xLeft[i];
            break;
        }
    }

    // Compute percentile calibration Max.
    float rightPercentile = this->_percentile / 100;
    for (auto i = numBins - 1; i >= 0; i--)
    {
        // Ensure that percentileMax is not greater than the max value of the tensor.
        if (cdf[i] < rightPercentile && this->_stats.xLeft[i] < maxVal)
        {
            percentileMax = this->_stats.xLeft[i] + histBinWidth;
            break;
        }
    }

    // Enforce difference between percentileMin and percentileMax to be atleast
    // one bin width. This will ensure that percentileMin and percentileMax are
    // not equal in the scenarios where most of the tensor values are concentrated
    // in a single histogram bin. This will also ensure that percentileMin and
    // percentileMax are edges of the single bin where most of the tensor elements
    // are concentrated.
    if (percentileMin == percentileMax)
        percentileMax += histBinWidth;

    return std::tuple<DTYPE, DTYPE>(percentileMin, percentileMax);
}

template <typename DTYPE>
std::tuple<DTYPE, DTYPE> PercentileEncodingAnalyzer<DTYPE>::_findRangeOfAggregateStats() const
{
    return findOriginalRange<DTYPE>(this->_stats);
}

template <typename DTYPE>
void PercentileEncodingAnalyzer<DTYPE>::setPercentileValue(float percentile)
{
    this->_percentile = percentile;
}

// Explicit instantiations
template class PercentileEncodingAnalyzer<double>;

template class PercentileEncodingAnalyzer<float>;

}   // namespace DlQuantization
