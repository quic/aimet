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
#include <numeric>
#include <vector>


#include "DlQuantization/Quantization.hpp"
#include "math_functions.hpp"
#include "quantization_utils.hpp"

#include "EntropyEncodingAnalyzer.h"

namespace DlQuantization
{
template <typename DTYPE>
std::vector<std::tuple<double, double>> EntropyEncodingAnalyzer<DTYPE>::getStatsHistogram() const
{
    // Return the collected histogram data.
    PDF stats;
    stats.xLeft.resize(PDF_SIZE);
    stats.pdf.resize(PDF_SIZE);
    double histMin      = this->_tensorProfilingParams.min;
    double histMax      = this->_tensorProfilingParams.max;
    double histBinWidth = (histMax - histMin) / PDF_SIZE;
    double histSum      = std::accumulate(this->_tensorProfilingParams.histogram.begin(),
                                     this->_tensorProfilingParams.histogram.end(), 0.f);
    // Initialize the probability for each bucket and left sides of the buckets.
    for (int i = 0; i < PDF_SIZE; ++i)
    {
        stats.pdf.push_back(this->_tensorProfilingParams.histogram[i] / histSum);
    }
    for (auto i = histMin; i <= histMax; i += histBinWidth)
    {
        stats.xLeft.push_back(i);
    }

    return getCollectedHistogram(stats);
}

template <typename DTYPE>
void EntropyEncodingAnalyzer<DTYPE>::updateStats(const DTYPE* tensor, const size_t tensorSize,
                                                 ComputationMode tensorCpuGpuMode)
{
    this->_statsUpdated = true;

    // update Histogram
    updateTensorHistogram(tensor, tensorSize, tensorCpuGpuMode, this->_tensorProfilingParams);
}

template <typename DTYPE>
void EntropyEncodingAnalyzer<DTYPE>::updateStats(const DTYPE* tensor, const size_t tensorSize,
                                                 ComputationMode tensorCpuGpuMode, IAllocator* allocator)
{
    updateStats(tensor, tensorSize, tensorCpuGpuMode);
}

template <typename DTYPE>
TfEncoding EntropyEncodingAnalyzer<DTYPE>::computeEncoding(uint8_t bw, bool useSymmetricEncodings,
                                                           bool useStrictSymmetric, bool useUnsignedSymmetric) const
{
    TfEncoding encoding = {0, 0, 0, 0, 0};
    DTYPE numSteps      = pow(2, bw) - 1;

    // For strict symmetric mode, we make even number of buckets
    if (useSymmetricEncodings && useStrictSymmetric)
    {
        numSteps -= 1;
    }

    if (this->_tensorProfilingParams.histogram.size() == 0)
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

    std::tie(aMin, aMax) = _optimizeKL(bw, useSymmetricEncodings, useStrictSymmetric, useUnsignedSymmetric);

    // After Min and Max adjustment, the requirement that 0 be an exactly
    // representable value must be met. Hence, extend the interval
    // [aMin, aMax] to ensure that it contains 0.
    aMin = std::min(aMin, DTYPE(0.f));
    aMax = std::max(aMax, DTYPE(0.f));

    assert(aMin <= aMax && "min must not be bigger than max");

    return getComputedEncodings(bw, aMin, aMax, useSymmetricEncodings, useStrictSymmetric, useUnsignedSymmetric);
}

template <typename DTYPE>
std::tuple<DTYPE, DTYPE> EntropyEncodingAnalyzer<DTYPE>::_findRangeOfAggregateStats() const
{
    return {this->_tensorProfilingParams.min, this->_tensorProfilingParams.max};
}

static void _conditionHistogram(double* hist, const size_t length)
{
    const double epsZero = 0.0001;
    // If histogram is empty then return.
    if (length == 0)
    {
        return;
    }

    // Get information about the zero values within the histogram.
    std::vector<int> isZero(length);
    size_t numZeros = 0;
    for (size_t idx = 0, e = length; idx < e; idx++)
    {
        isZero[idx] = static_cast<int>(hist[idx] == 0.f);
        numZeros += isZero[idx];
    }

    // If histogram is all zeros then return.
    if (numZeros == length)
    {
        return;
    }

    // Compute epsilon to subtract from non-zero histogram values.
    size_t numNonZeros = length - numZeros;
    double epsNonZero  = epsZero * static_cast<double>(numZeros) / static_cast<double>(numNonZeros);

    // If value to subtract from non-zero values is higher than 1.0 then return.
    if (epsNonZero >= 1.0)
    {
        return;
    }

    // Perform histogram conditioning:
    // - zero histogram values are increased with epsZero.
    // - non-zero histogram values are decreased with epsNonZero.
    for (size_t idx = 0, e = length; idx < e; idx++)
    {
        hist[idx] += epsZero * isZero[idx];
        hist[idx] -= epsNonZero * (1 - isZero[idx]);
    }
}

static double _computeKL(double* P, double* Q, size_t length)
{
    // Compute sum of P and Q to use for normalization.
    double sumP = std::accumulate(P, P + length, 0.f);
    double sumQ = std::accumulate(Q, Q + length, 0.f);

    // optimizeKL function should stop KL divergence computation when P or Q
    // distributions are all zeros. Hence P or Q distributions cannot be all
    // zeros in computeKL function.
    assert(sumP != 0 && "P distribution is all zeros!");
    assert(sumQ != 0 && "Q distribution is all zeros!");

    // Compute relative entropy.
    double divergence = 0;
    for (size_t idx = 0, e = length; idx < e; idx++)
    {
        P[idx] /= sumP;
        Q[idx] /= sumQ;
        if ((P[idx] > 0) && (Q[idx] > 0))
        {
            divergence += P[idx] * std::log(P[idx] / Q[idx]);
        }
    }
    return divergence;
}

template <typename DTYPE>
std::tuple<DTYPE, DTYPE> EntropyEncodingAnalyzer<DTYPE>::_optimizeKL(uint8_t bw, bool useSymmetricEncodings,
                                                                     bool useStrictSymmetric,
                                                                     bool useUnsignedSymmetric) const
{
    double histMin = this->_tensorProfilingParams.min;
    double histMax = this->_tensorProfilingParams.max;

    std::vector<double> hist;
    // Incase of symmetric quantization adjust the histMin, histMax and rescale the hisogram.
    if (useSymmetricEncodings && ((histMin < 0.0) || (!useUnsignedSymmetric)))
    {
        DTYPE absoluteMax = std::max(std::abs(histMax), std::abs(histMin));
        DTYPE absoluteMin = -absoluteMax;
        hist    = rescaleHistogram(this->_tensorProfilingParams.histogram, histMin, histMax, absoluteMin, absoluteMax);
        histMin = absoluteMin;
        histMax = absoluteMax;
    }
    else
    {
        hist = this->_tensorProfilingParams.histogram;
    }

    // Number of histogram bins.
    const size_t numBins = hist.size();

    // Number of quantized bins.
    const size_t numQuantizedBins = 255;

    // If the input histogram is empty or the number of histogram bins is smaller
    // than numQuantizedBins then return the histogramthis->_tensorProfilingParams.histogram range.
    if ((numBins == 0) || (numBins < numQuantizedBins) || (bw != 8))
    {
        return {histMin, histMax};
    }

    // Histogram bin width.
    assert(histMin < histMax && "Invalid histogram min/max range!");
    const double histBinWidth = (histMax - histMin) / (double) numBins;

    // Optimal divergence value (minimum).
    double divergenceOpt = std::numeric_limits<double>::infinity();

    // Optimal threshold values for minimum divergence.
    double thresholdMinOpt = histMin;
    double thresholdMaxOpt = histMax;

    // Initialize start/stop bin indices (inclusive) with the first and last bin.
    size_t histWinIdxStart = 0;
    size_t histWinIdxStop  = numBins - 1;

    while ((histWinIdxStop - histWinIdxStart + 1) >= numQuantizedBins)
    {
        // Current histogram window size.
        const size_t histWinSize = histWinIdxStop - histWinIdxStart + 1;

        // Current histogram window raw pointer.
        const double* histWinPtr = hist.data() + histWinIdxStart;

        // Compute the reference distribution P as the input histogram saturated in
        // the current window given by histWinIdxStart and histWinIdxStop.
        std::vector<double> P(histWinSize);

        // Saturate the histogram left.
        double leftSum = 0;
        for (size_t histIdx = 0; histIdx <= histWinIdxStart; histIdx++)
        {
            leftSum += hist[histIdx];
        }
        P.front() += leftSum;

        // Extract the non-saturated part of the histogram.
        for (size_t histIdx = histWinIdxStart + 1; histIdx < histWinIdxStop; histIdx++)
        {
            P[histIdx - histWinIdxStart] = hist[histIdx];
        }

        // Saturate the histogram right.
        double rightSum = 0;
        for (size_t histIdx = histWinIdxStop; histIdx < numBins; histIdx++)
        {
            rightSum += hist[histIdx];
        }
        P.back() += rightSum;

        const double numMergedBins = static_cast<double>(histWinSize) / static_cast<double>(numQuantizedBins);

        // Compute Q.
        std::vector<double> Q(histWinSize, 0);
        for (size_t qIdx = 0; qIdx < numQuantizedBins; qIdx++)
        {
            // Histogram window bin start index (inclusive) for this quantized bin.
            const size_t idxStart = ceil(qIdx * numMergedBins);

            // Histogram window bin stop index (exclusive) for this quantized bin.
            // If last quantized bin then go to the end of the window.
            const size_t idxStop = (qIdx < (numQuantizedBins - 1)) ? ceil((qIdx + 1) * numMergedBins) : histWinSize;

            // Sum all the values for this quantized bin.
            // Count all the positive values for this quantized bin to use for
            // normalization.
            double sum  = 0;
            double norm = 0;
            for (size_t idx = idxStart; idx < idxStop; idx++)
            {
                sum += histWinPtr[idx];
                norm += (histWinPtr[idx] != 0);
            }

            // Compute Q by expanding and normalizing the quantized bins.
            if (norm != 0)
            {
                for (size_t idx = idxStart; idx < idxStop; idx++)
                {
                    if (histWinPtr[idx])
                    {
                        Q[idx] = sum / static_cast<double>(norm);
                    }
                }
            }
        }

        // For one sided distributions, there is a possibility that all the values
        // in the shrinked histogram are zeros. In such cases, Q distribution is
        // all zeros and the KL divergence value is infinity. Hence, shrinking of
        // the histogram further and analyzing for minimum KL divergence value is
        // redundant.
        // Break the histogram shrinking and KL divergence computation when all
        // elements in P or Q are zeros.
        double sumP = std::accumulate(P.begin(), P.end(), 0.f);
        double sumQ = std::accumulate(Q.begin(), Q.end(), 0.f);
        if (sumP == 0 || sumQ == 0)
        {
            break;
        }

        // Compute the KL divergence metric and check for optimal values.
        // Condition the histograms P and Q.
        _conditionHistogram(P.data(), P.size());
        _conditionHistogram(Q.data(), Q.size());

        // Compute the divergence of P with respect to Q.
        double divergence = _computeKL(P.data(), Q.data(), P.size());

        // Check if current divergence is the new optimal.
        if (divergence < divergenceOpt)
        {
            // Update optimal divergence with current divergence.
            divergenceOpt = divergence;

            // Update optimal thresholds with current thresholds.
            thresholdMinOpt = histMin + histWinIdxStart * histBinWidth;
            thresholdMaxOpt = histMin + (histWinIdxStop + 1) * histBinWidth;
        }

        // Update histogram window for next iteration.
        if (useSymmetricEncodings || useStrictSymmetric)
        {
            // For symmetric schema we shrink the histogram window symmetrically.
            histWinIdxStart++;
            histWinIdxStop--;
        }
        else
        {
            // For asymmetric schema we shrink the histogram window either left-only,
            // right-only or symmetrically depending on which case has minimum
            // histogram data loss.
            double symmLoss  = hist[histWinIdxStart] + hist[histWinIdxStop];
            double leftLoss  = hist[histWinIdxStart] + hist[histWinIdxStart + 1];
            double rightLoss = hist[histWinIdxStop] + hist[histWinIdxStop - 1];

            std::vector<double> loss = {symmLoss, leftLoss, rightLoss};
            auto lossMinIdx          = std::distance(loss.begin(), std::min_element(loss.begin(), loss.end()));

            // The requirement that 0 be an exactly representable value must be met
            // during min/max adjustments with calibration.
            // If adjusted min > 0, shrink histogram only on the right
            // If adjusted max < 0, shrink histogram only on the left
            if ((lossMinIdx == 0 && (histMin + (histWinIdxStart + 1) * histBinWidth) > 0) ||
                (lossMinIdx == 1 && (histMin + (histWinIdxStart + 2) * histBinWidth) > 0))
            {
                lossMinIdx = 2;
            }
            else if ((lossMinIdx == 0 && (histMin + histWinIdxStop * histBinWidth) < 0) ||
                     (lossMinIdx == 2 && (histMin + (histWinIdxStop - 1) * histBinWidth) < 0))
            {
                lossMinIdx = 1;
            }

            if (lossMinIdx == 0)
            {
                // Saturate symmetrically.
                histWinIdxStart++;
                histWinIdxStop--;
            }
            else if (lossMinIdx == 1)
            {
                // Saturate left.
                histWinIdxStart += 2;
            }
            else
            {
                // Saturate right.
                histWinIdxStop -= 2;
            }
        }
    }

    return {thresholdMinOpt, thresholdMaxOpt};
}

// Explicit instantiations
template class EntropyEncodingAnalyzer<double>;

template class EntropyEncodingAnalyzer<float>;

}   // namespace DlQuantization
