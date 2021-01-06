//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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
#include <iostream>

#include "DlQuantization/Quantization.hpp"
#include "math_functions.hpp"

#include "TfEnhancedEncodingAnalyzer.h"

namespace DlQuantization
{
template <typename DTYPE>
void TfEnhancedEncodingAnalyzer<DTYPE>::updateStats(const DTYPE* tensor, const size_t tensorSize,
                                                    ComputationMode tensorCpuGpuMode)
{
    UpdatePdf(tensor, tensorSize, tensorCpuGpuMode, true, this->_stats);
}


template <typename DTYPE>
TfEncoding TfEnhancedEncodingAnalyzer<DTYPE>::computeEncoding(uint8_t bw, bool useSymmetricEncodings) const
{
    // Find the range of our collected stats
    DTYPE min_val, max_val;
    std::tie(min_val, max_val) = _findRangeOfAggregateStats();

    // Find test candidates
    DTYPE num_steps = pow(2, bw) - 1;
    std::vector<std::tuple<DTYPE, int>> test_candidates;

    if (useSymmetricEncodings)
    {
        _pickTestCandidatesSymmetric(min_val, max_val, num_steps, test_candidates);
    }
    else
    {
        _pickTestCandidatesAsymmetric(min_val, max_val, num_steps, test_candidates);
    }

    // Find the best candidate
    DTYPE best_delta;
    int best_offset;
    std::tie(best_delta, best_offset) = _findBestCandidate(bw, test_candidates);

    // Using the best delta and offset, calculate the encoding.
    TfEncoding encoding;
    encoding.delta  = best_delta;
    encoding.offset = best_offset;
    encoding.bw     = bw;
    encoding.min    = best_delta * best_offset;
    encoding.max    = best_delta * (float) num_steps + encoding.min;

    return encoding;
}

template <typename DTYPE>
std::tuple<DTYPE, int>
TfEnhancedEncodingAnalyzer<DTYPE>::_findBestCandidate(uint8_t bw,
                                                      const std::vector<std::tuple<DTYPE, int>>& test_candidates) const
{
    DTYPE best_delta = -1;
    int best_offset  = -1;
    // Go through all <delta, offset> pairs and calculate the quantization and
    // saturation cost.
    // This is a 2d grid search.

    DTYPE best_cost = std::numeric_limits<float>::max();

    for (auto candidate: test_candidates)
    {
        DTYPE test_delta;
        int test_offset;

        std::tie(test_delta, test_offset) = candidate;

        DTYPE cost = _quantAndSatCost(_stats, bw, test_delta, test_offset);

        // Remember the best encoding.
        if (cost < best_cost)
        {
            best_cost   = cost;
            best_delta  = test_delta;
            best_offset = test_offset;
        }
    }

    return std::tuple<DTYPE, int>(best_delta, best_offset);
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
    DTYPE observedMin, DTYPE observedMax, DTYPE numSteps, std::vector<std::tuple<DTYPE, int>>& test_candidates) const
{
    // Map observedMin and observedMax to grid points
    DTYPE observedDelta = (observedMax - observedMin) / numSteps;
    int observedOffset = round(observedMin / observedDelta);
    observedMin = observedDelta * observedOffset;
    observedMax = observedMin + observedDelta * numSteps;

    // Compute the largest TF delta which would make sense, based on the range
    // [observedMin ... observedMax] we just calculated.
    DTYPE delta_max = observedDelta;

    // Compute the deltas we will test.
    // We test 17 deltas, equally spaced between 1*delta_max/16 and
    // 17*delta_max/16. Note we consider one delta which is larger than delta_max.
    // The reason we do this is as follows: Due to floating point rounding errors,
    // delta_max might not be able to fully cover the whole range.
    for (DTYPE f = 1.0 / 16; f <= 1 + 1.0 / 16; f += 1.0 / 16)
    {
        DTYPE testDelta = f * delta_max;

        // Compute the offsets we will test.
        // We consider 20 different offsets, equally spaced from -255 to 0.
        for (int i = 0; i <= 20; ++i)
        {
            int testOffset = -numSteps + numSteps / 20.0 * i;

            // Clamp test candidates to the observedMin and observedMax range.
            if (!_clampToObservedMinMax(observedMin, observedMax, numSteps, testDelta, testOffset))
                continue;
            test_candidates.push_back(std::tuple<DTYPE, int>(testDelta, testOffset));
        }
    }

    // Add one candidate corresponding to the observed max and min
    test_candidates.push_back(std::tuple<DTYPE, int>(observedDelta, observedOffset));
}

template <typename DTYPE>
void TfEnhancedEncodingAnalyzer<DTYPE>::_pickTestCandidatesSymmetric(
    DTYPE min_val, DTYPE max_val, DTYPE num_steps, std::vector<std::tuple<DTYPE, int>>& test_candidates) const
{
    // Compute the largest TF delta which would make sense, based on the range
    // [min_val ... max_val] we just calculated.

    DTYPE delta_max = 0.0;
    int test_offset = 0;

    if (min_val == 0.0)
    {
        // Special case for symmetric encodings. If all values are positive or 0, we can treat the
        // symmetric encodings as unsigned
        delta_max = max_val / num_steps;
        test_offset = 0;        // Indicates all positive values
    }
    else
    {
        DTYPE absolute_max = std::max(std::abs(max_val), std::abs(min_val));
        delta_max    = (2 * absolute_max) / num_steps;

        // Compute the offset - since we are finding symmetric candidates, offset can be computed given the delta
        test_offset = -(num_steps / 2);
    }

    // Compute the deltas we will test.
    // We test 101 deltas, equally spaced between 1*delta_max/100 and
    // 101*delta_max/100. Note we consider one delta which is larger than delta_max.
    // The reason we do this is as follows: Due to floating point rounding errors,
    // delta_max might not be able to fully cover the whole range.
    for (DTYPE f = 1.0 / 100; f <= 1 + 1.0 / 100; f += 1.0 / 100)
    {
        DTYPE test_delta = f * delta_max;
        test_candidates.push_back(std::tuple<DTYPE, int>(test_delta, test_offset));
    }
}

template <typename DTYPE>
std::tuple<DTYPE, DTYPE> TfEnhancedEncodingAnalyzer<DTYPE>::_findRangeOfAggregateStats() const
{
    DTYPE min_val = _stats.x_left[0];
    DTYPE max_val =
        _stats.x_left[PDF_SIZE - 1];   // First we need to find which range we want to cover with our TF encoding.

    // To do so we search for the smallest and largest value from the this->_stats
    // Search for the lowest bucket which has probability > 0.
    for (int i = 0; i < PDF_SIZE; ++i)
    {
        if (_stats.pdf[i] > 0)
        {
            min_val = _stats.x_left[i];
            break;
        }
    }

    // Search for the highest bucket which has probability > 0.
    for (int i = PDF_SIZE - 1; i > 0; --i)
    {
        if (_stats.pdf[i] > 0)
        {
            max_val = _stats.x_left[i];
            break;
        }
    }

    // Make sure we include zero in range.
    min_val = std::min(min_val, (DTYPE) 0);
    max_val = std::max(max_val, (DTYPE) 0);

    // Make sure we have a real range.
    max_val = std::max(max_val, min_val + (DTYPE) MIN_RANGE);

    return std::tuple<DTYPE, DTYPE>(min_val, max_val);
}

template <typename DTYPE>
DTYPE TfEnhancedEncodingAnalyzer<DTYPE>::_quantAndSatCost(const PDF& pdf, int bw, DTYPE delta, int offset) const
{
    // Given the TensorFlow fixed point format (delta and offset), we calculate
    // the smallest and biggest floating point values we can represent.
    DTYPE min_val   = delta * offset;
    DTYPE step_size = pow(2, bw) - 1;
    DTYPE max_val   = delta * step_size + min_val;
    // Calculate the indices of the smallest and largest representable value.
    DTYPE pdf_start = pdf.x_left[0];
    DTYPE pdf_step  = pdf.x_left[1] - pdf.x_left[0];
    int min_ind     = (int) std::floor((min_val - pdf_start) / pdf_step);
    min_ind         = std::min(std::max(0, min_ind), PDF_SIZE - 1);
    int maxInd      = (int) std::floor((max_val - pdf_start) / pdf_step);
    maxInd          = std::min(std::max(0, maxInd), PDF_SIZE - 1);

    // Calculate the saturation cost of the bottom part of the PDF.
    DTYPE sat_cost_bottom = 0;
    // Calculate the smallest value we can represent (middle of respective
    // bucket).
    DTYPE min_val_middle_of_bucket = pdf_start + (min_ind * pdf_step) + pdf_step / 2;
    // Go through all buckets which go into saturation.
    for (int i = 0; i < min_ind; ++i)
    {
        // Calculate the midpoint of this bin.
        DTYPE mid_val = pdf_start + i * pdf_step + pdf_step / 2;
        // The saturation cost is the MSE.
        sat_cost_bottom += pdf.pdf[i] * pow(mid_val - min_val_middle_of_bucket, 2);
    }

    // Calculate the saturation cost of the top part of the PDF.
    DTYPE sat_cost_top = 0;
    // Calculate the largest value we can represent (middle of respective
    // bucket).
    DTYPE max_val_middle_of_bucket = pdf_start + (maxInd * pdf_step) + pdf_step / 2;
    // Go through all buckets which go into saturation.
    for (int i = maxInd; i < PDF_SIZE; ++i)
    {
        // Calculate the midpoint of this bin.
        DTYPE mid_val = pdf_start + i * pdf_step + pdf_step / 2;
        // The saturation cost is the MSE.
        sat_cost_top += pdf.pdf[i] * pow(mid_val - max_val_middle_of_bucket, 2);
    }

    // Calculate the quantization cost in the middle part of the PDF.
    DTYPE quant_cost = 0;
    // Go through all buckets which lie in the range we can represent.
    for (int i = min_ind; i < maxInd; ++i)
    {
        // The floating point value in the middle of this bucket.
        DTYPE float_val = pdf_start + i * pdf_step + pdf_step / 2;
        // The quantized equivalent.
        int quantized = (int) round(float_val / delta - offset);
        // The de-quantized value: this is 'float_val' plus the quantization error.
        DTYPE dequantized = delta * (quantized + offset);
        // The quantization cost is the MSE.
        quant_cost += pdf.pdf[i] * pow(float_val - dequantized, 2);
    }

    // Calculate the total cost as the sum of quantization and saturation cost.
    DTYPE sqnr = GAMMA * (sat_cost_bottom + sat_cost_top) + quant_cost;
    return sqnr;
}


// Explicit instantiations
template class TfEnhancedEncodingAnalyzer<double>;

template class TfEnhancedEncodingAnalyzer<float>;

}