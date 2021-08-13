//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2019-2021, Qualcomm Innovation Center, Inc. All rights reserved.
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
#include <cstddef>
#include <vector>
#include <cmath>

#include "DlQuantization/Quantization.hpp"
#include "math_functions.hpp"

#include "TfEnhancedEncodingAnalyzer.h"

namespace DlQuantization
{

template <typename DTYPE>
std::vector<std::tuple<double, double>> TfEnhancedEncodingAnalyzer<DTYPE>::getStatsHistogram() const
{
    // Allocate a vector to hold tuples of left edges and pdf for each bucket
    std::vector<std::tuple<double, double>> histogram;

    // Assert that the stats structure is well formed
    assert(this->_stats.xLeft.size() == this->_stats.hist.size());

    unsigned index = 0;
    for (auto entry: this->_stats.xLeft)
    {
        histogram.push_back(std::make_tuple(entry, (int)this->_stats.hist[index]));
        index++;
    }
    return histogram;
}

template <typename DTYPE>
void TfEnhancedEncodingAnalyzer<DTYPE>::updateStats(const DTYPE* tensor, const size_t tensorSize,
                                                    ComputationMode tensorCpuGpuMode)
{
    UpdatePdf(tensor, tensorSize, tensorCpuGpuMode, true, this->_stats);
}


template <typename DTYPE>
TfEncoding TfEnhancedEncodingAnalyzer<DTYPE>::computeEncoding(uint8_t bw, bool useSymmetricEncodings,
                                                              bool useStrictSymmetric, bool useUnsignedSymmetric) const
{
    TfEncoding encoding = {0, 0, 0, 0, 0};

    if (this->_stats.xLeft.size() == 0)
    {
        // Histogram has not been initialized yet
        // We return a zero encoding
        return encoding;
    }

    // Find the range of our collected stats
    DTYPE minVal, maxVal;
    std::tie(minVal, maxVal) = _findRangeOfAggregateStats();

    // Find test candidates
    DTYPE numSteps = pow(2, bw) - 1;
    std::vector<std::tuple<DTYPE, int>> testCandidates;

    if (useSymmetricEncodings)
    {
        // For strict symmetric mode, we make even number of buckets
        if (useStrictSymmetric)
            numSteps -= 1;

        _pickTestCandidatesSymmetric(minVal, maxVal, numSteps, testCandidates, useUnsignedSymmetric);
    }
    else
    {
        _pickTestCandidatesAsymmetric(minVal, maxVal, numSteps, testCandidates);
    }

    // Find the best candidate
    DTYPE bestDelta;
    int bestOffset;
    std::tie(bestDelta, bestOffset) = _findBestCandidate(bw, testCandidates);

    // Using the best delta and offset, calculate the encoding.
    encoding.delta  = bestDelta;
    encoding.offset = bestOffset;
    encoding.bw     = bw;
    encoding.min    = bestDelta * bestOffset;
    encoding.max    = bestDelta * (float) numSteps + encoding.min;

    return encoding;
}

template <typename DTYPE>
std::tuple<DTYPE, int>
TfEnhancedEncodingAnalyzer<DTYPE>::_findBestCandidate(uint8_t bw,
                                                      const std::vector<std::tuple<DTYPE, int>>& testCandidates) const
{
    DTYPE bestDelta  = -1;
    int bestOffset   = -1;
    // Go through all <delta, offset> pairs and calculate the quantization and
    // saturation cost.
    // This is a 2d grid search.

    DTYPE bestCost = std::numeric_limits<float>::max();

    for (auto candidate: testCandidates)
    {
        DTYPE testDelta;
        int testOffset;

        std::tie(testDelta, testOffset) = candidate;

        DTYPE cost = _quantAndSatCost(_stats, bw, testDelta, testOffset);

        // Remember the best encoding.
        if (cost < bestCost)
        {
            bestCost    = cost;
            bestDelta   = testDelta;
            bestOffset  = testOffset;
        }
    }

    return std::tuple<DTYPE, int>(bestDelta, bestOffset);
}

template <typename DTYPE>
bool TfEnhancedEncodingAnalyzer<DTYPE>::_clampToObservedMinMax(DTYPE observedMin, DTYPE observedMax, DTYPE numSteps,
                                                               DTYPE& testDelta, int& testOffset) const
{
    // Calculate observed delta and offset
    DTYPE testMin = testDelta * testOffset;
    DTYPE testMax = testMin + testDelta * numSteps;

    if ((testMin < observedMin) && (testMax > observedMax))
    {
        return false;
    }

    testMin = std::max(observedMin, testMin);
    testMax = std::min(observedMax, testMax);

    // Recalculate the test delta and offset
    testDelta = (testMax - testMin) / numSteps;
    testOffset = round(testMin / testDelta);

    return true;
}

template <typename DTYPE>
void TfEnhancedEncodingAnalyzer<DTYPE>::_pickTestCandidatesAsymmetric(
    DTYPE observedMin, DTYPE observedMax, DTYPE numSteps, std::vector<std::tuple<DTYPE, int>>& testCandidates) const
{
    // Map observedMin and observedMax to grid points
    DTYPE observedDelta = (observedMax - observedMin) / numSteps;
    int observedOffset = round(observedMin / observedDelta);
    observedMin = observedDelta * observedOffset;
    observedMax = observedMin + observedDelta * numSteps;

    // Compute the largest TF delta which would make sense, based on the range
    // [observedMin ... observedMax] we just calculated.
    DTYPE deltaMax = observedDelta;

    // Compute the deltas we will test.
    // We test 17 deltas, equally spaced between 1*deltaMax/16 and
    // 17*deltaMax/16. Note we consider one delta which is larger than deltaMax.
    // The reason we do this is as follows: Due to floating point rounding errors,
    // deltaMax might not be able to fully cover the whole range.
    for (DTYPE f = 1.0 / 16; f <= 1 + 1.0 / 16; f += 1.0 / 16)
    {
        DTYPE testDelta = f * deltaMax;

        // Compute the offsets we will test.
        // We consider 20 different offsets, equally spaced from -255 to 0.
        for (int i = 0; i <= 20; ++i)
        {
            int testOffset = -numSteps + numSteps / 20.0 * i;

            // Clamp test candidates to the observedMin and observedMax range.
            if (!_clampToObservedMinMax(observedMin, observedMax, numSteps, testDelta, testOffset))
                continue;
            testCandidates.push_back(std::tuple<DTYPE, int>(testDelta, testOffset));
        }
    }

    // Add one candidate corresponding to the observed max and min
    testCandidates.push_back(std::tuple<DTYPE, int>(observedDelta, observedOffset));
}

template <typename DTYPE>
void TfEnhancedEncodingAnalyzer<DTYPE>::_pickTestCandidatesSymmetric(DTYPE minVal, DTYPE maxVal, DTYPE numSteps,
                                                                     std::vector<std::tuple<DTYPE, int>>& testCandidates,
                                                                     bool useUnsignedSymmetric) const
{
    // Compute the largest TF delta which would make sense, based on the range
    // [minVal ... maxVal] we just calculated.

    DTYPE deltaMax  = 0.0;
    int testOffset  = 0;

    if ((minVal == 0.0) && (useUnsignedSymmetric))
    {
        // Special case for symmetric encodings. If all values are positive or 0, we can treat the
        // symmetric encodings as unsigned
        deltaMax    = maxVal / numSteps;
        testOffset  = 0;        // Indicates all positive values
    }
    else
    {
        DTYPE absoluteMax  = std::max(std::abs(maxVal), std::abs(minVal));
        deltaMax           = (2 * absoluteMax) / numSteps;

        // Compute the offset - since we are finding symmetric candidates, offset can be computed given the delta
        testOffset = floor(-numSteps / 2);
    }

    // Compute the deltas we will test.
    // We test 101 deltas, equally spaced between 1*deltaMax/100 and
    // 101*deltaMax/100. Note we consider one delta which is larger than deltaMax.
    // The reason we do this is as follows: Due to floating point rounding errors,
    // deltaMax might not be able to fully cover the whole range.
    for (DTYPE f = 1.0 / 100; f <= 1 + 1.0 / 100; f += 1.0 / 100)
    {
        DTYPE testDelta = f * deltaMax;
        testCandidates.push_back(std::tuple<DTYPE, int>(testDelta, testOffset));
    }
}

template <typename DTYPE>
std::tuple<DTYPE, DTYPE> TfEnhancedEncodingAnalyzer<DTYPE>::_findRangeOfAggregateStats() const
{
    DTYPE minVal = _stats.xLeft[0];
    DTYPE maxVal =
        _stats.xLeft[PDF_SIZE - 1];   // First we need to find which range we want to cover with our TF encoding.

    // To do so we search for the smallest and largest value from the this->_stats
    // Search for the lowest bucket which has probability > 0.
    for (int i = 0; i < PDF_SIZE; ++i)
    {
        if (_stats.pdf[i] > 0)
        {
            minVal = _stats.xLeft[i];
            break;
        }
    }

    // Search for the highest bucket which has probability > 0.
    for (int i = PDF_SIZE - 1; i > 0; --i)
    {
        if (_stats.pdf[i] > 0)
        {
            maxVal = _stats.xLeft[i];
            break;
        }
    }

    // Make sure we include zero in range.
    minVal  = std::min(minVal, (DTYPE) 0);
    maxVal  = std::max(maxVal, (DTYPE) 0);

    // Make sure we have a real range.
    maxVal = std::max(maxVal, minVal + (DTYPE) MIN_RANGE);

    return std::tuple<DTYPE, DTYPE>(minVal, maxVal);
}

template <typename DTYPE>
DTYPE TfEnhancedEncodingAnalyzer<DTYPE>::_quantAndSatCost(const PDF& pdf, int bw, DTYPE delta, int offset) const
{
    // Given the TensorFlow fixed point format (delta and offset), we calculate
    // the smallest and biggest floating point values we can represent.
    DTYPE minVal    = delta * offset;
    DTYPE stepSize  = pow(2, bw) - 1;
    DTYPE maxVal    = delta * stepSize + minVal;
    // Calculate the indices of the smallest and largest representable value.
    DTYPE pdfStart  = pdf.xLeft[0];
    DTYPE pdfStep   = pdf.xLeft[1] - pdf.xLeft[0];
    int minInd      = (int) std::floor((minVal - pdfStart) / pdfStep);
    minInd          = std::min(std::max(0, minInd), PDF_SIZE - 1);
    int maxInd      = (int) std::floor((maxVal - pdfStart) / pdfStep);
    maxInd          = std::min(std::max(0, maxInd), PDF_SIZE - 1);

    // Calculate the saturation cost of the bottom part of the PDF.
    DTYPE satCostBottom = 0;
    // Calculate the smallest value we can represent (middle of respective
    // bucket).
    DTYPE minValMiddleOfBucket = pdfStart + (minInd * pdfStep) + pdfStep / 2;
    // Go through all buckets which go into saturation.
    for (int i = 0; i < minInd; ++i)
    {
        // Calculate the midpoint of this bin.
        DTYPE midVal = pdfStart + i * pdfStep + pdfStep / 2;
        // The saturation cost is the MSE.
        satCostBottom += pdf.pdf[i] * pow(midVal - minValMiddleOfBucket, 2);
    }

    // Calculate the saturation cost of the top part of the PDF.
    DTYPE satCostTop = 0;
    // Calculate the largest value we can represent (middle of respective
    // bucket).
    DTYPE maxValMiddleOfBucket = pdfStart + (maxInd * pdfStep) + pdfStep / 2;
    // Go through all buckets which go into saturation.
    for (int i = maxInd; i < PDF_SIZE; ++i)
    {
        // Calculate the midpoint of this bin.
        DTYPE midVal = pdfStart + i * pdfStep + pdfStep / 2;
        // The saturation cost is the MSE.
        satCostTop += pdf.pdf[i] * pow(midVal - maxValMiddleOfBucket, 2);
    }

    // Calculate the quantization cost in the middle part of the PDF.
    DTYPE quantCost = 0;
    // Go through all buckets which lie in the range we can represent.
    for (int i = minInd; i < maxInd; ++i)
    {
        // The floating point value in the middle of this bucket.
        DTYPE floatVal = pdfStart + i * pdfStep + pdfStep / 2;
        // The quantized equivalent.
        int quantized = (int) round(floatVal / delta - offset);
        // The de-quantized value: this is 'floatVal' plus the quantization error.
        DTYPE dequantized = delta * (quantized + offset);
        // The quantization cost is the MSE.
        quantCost += pdf.pdf[i] * pow(floatVal - dequantized, 2);
    }

    // Calculate the total cost as the sum of quantization and saturation cost.
    DTYPE sqnr = GAMMA * (satCostBottom + satCostTop) + quantCost;
    return sqnr;
}


// Explicit instantiations
template class TfEnhancedEncodingAnalyzer<double>;

template class TfEnhancedEncodingAnalyzer<float>;

}