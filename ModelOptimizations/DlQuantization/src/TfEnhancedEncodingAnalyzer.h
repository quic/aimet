//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2019-2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

#ifndef DL_QUANTIZATION_TF_ENHANCED_ENCODING_ANALYZER_H
#define DL_QUANTIZATION_TF_ENHANCED_ENCODING_ANALYZER_H

// This file contains code to analyze and calculate quantization encodings
// This code is specific for the TF Enhanced quantization scheme

#include "math_functions.hpp"
#include <DlQuantization/IQuantizationEncodingAnalyzer.hpp>

namespace DlQuantization
{
template <typename DTYPE>
class TfEnhancedEncodingAnalyzer : public IQuantizationEncodingAnalyzer<DTYPE>
{
public:
    /**
     * Updates internal PDF stats given a tensor.
     * Intent is to keep a histogram of all the values that we have seen over multiple instances of a tensor
     * @param tensor Reference to a tensor
     * @param tensorSize Size of the tensor (number of elements)
     * @param tensorCpuGpuMode Indicates if the tensor is in CPU or GPU memory
     */
    void updateStats(const DTYPE* tensor, const size_t tensorSize, ComputationMode tensorCpuGpuMode) override;

    void updateStats(const DTYPE* tensor, const size_t tensorSize,
                     ComputationMode tensorCpuGpuMode, IAllocator* allocator) override;

    /***
     * Given a number distribution in CPU memory, compute the TensorFlow encoding with the highest possible SQNR
     *
     * To do so, we perform a grid search over different deltas and offsets. This grid search optimizes the encoding
     * to reduce the cost of quantization. In this cost function, saturation errors are weighted higher than
     * quantization errors.
     *
     * @param bw Bitwidth to use for computing encodings
     * @param useSymmetricEncodings If true, compute symmetric encodings
     * @return Computed encoding
     */
    TfEncoding computeEncoding(uint8_t bw, bool useSymmetricEncodings, bool useStrictSymmetric,
                               bool useUnsignedSymmetric) const override;

    /**
     *  Adjusts min and max according quant scheme and determines an acceptable range
     *  for a meaningful scale and offset
     */
    void getComputedEncodings(int bw, TfEncoding& encoding, bool useSymmetricEncodings,
                              bool useStrictSymmetric, bool useUnsignedSymmetric) const;


    /**
     * @brief Returns a histogram that represents a PDF of tensor values seen by this encoding analyzer so far
     *
     * @return Histogram of statistics. The histogram returned is a vector of buckets. Each bucket is a tuple of
     * two values - the float value representing the left edge of the bucket and a PDF of the values in this bucket
     * relative to all the values seen across all buckets
     */
    std::vector<std::tuple<double, double>> getStatsHistogram() const override;

private:
    PDF _stats;
    bool _statsUpdated = false;

    // Fudge factor which trades-off quantization and saturation error.
    // The cost function will be "quantization cost" + GAMMA * "saturation cost".
    static constexpr DTYPE GAMMA = 3.0;

    // Minimum range of quantization
    static constexpr double MIN_RANGE = 0.01;

    /**
     * @brief Given a probability density and a fixed point encoding, compute the
     * quantization and saturation error of this number distribution.
     *
     * Note: This function computes the cost of quantizing a number distribution.
     * The cost is defined as "quantization cost" + GAMMA * "saturation cost".
     * For GAMMA==1, this function computes the means square error introduced
     * by this specific fixed point encoding.
     */
    DTYPE _quantAndSatCost(const PDF& pdf, int bw, DTYPE delta, int offset) const;

    /**
     * Pick asymmetric test candidates to use for searching the lowest quantized cost. Each candidates is expressed
     * in terms of its delta and offset
     * @param observedMin Minimum value of the stats
     * @param observedMax Max value of the stats
     * @param numSteps Number of delta steps (based on the bitwidth)
     * @param test_deltas Vector of deltas (test candidate returned)
     * @param test_offsets Vector of offsets (test candidate returned)
     */
    void _pickTestCandidatesAsymmetric(DTYPE observedMin, DTYPE observedMax, DTYPE numSteps,
                                       std::vector<std::tuple<DTYPE, int>>& testCandidates) const;

    /**
     * Pick symmetric test candidates to use for searching the lowest quantized cost. Each candidates is expressed
     * in terms of its delta and offset
     * @param minVal Minimum value of the stats
     * @param maxVal Max value of the stats
     * @param numSteps Number of delta steps (based on the bitwidth)
     * @param test_deltas Vector of deltas (test candidate returned)
     * @param test_offsets Vector of offsets (test candidate returned)
     */
    void _pickTestCandidatesSymmetric(DTYPE minVal, DTYPE maxVal, DTYPE numSteps,
                                      std::vector<std::tuple<DTYPE, int>>& testCandidates, bool useUnsignedSymmetric) const;

    /**
     * Clamp given test delta and test offset based on observed min and max
     * @param observedMin Minimum value of the stats
     * @param observedMax Max value of the stats
     * @param numSteps Number of delta steps (based on the bitwidth)
     * @param testDelta test delta
     * @param testOffset test offset
     * @return False if the test candidate is outside both the observed min and max
     */
    bool _clampToObservedMinMax(DTYPE observedMin, DTYPE observedMax, DTYPE numSteps,
                                DTYPE& testDelta, int& testOffset) const;
    /**
     * Given a set of test candidates (delta x offsets), find the best candidate with the lowest cost
     * @param bw Bitwidth
     * @param test_candidates Vector of tuples (test-delta and test-offset)
     * @return Tuple of <best delta, best offset>
     */
    std::tuple<DTYPE, int> _findBestCandidate(uint8_t bw,
                                              const std::vector<std::tuple<DTYPE, int>>& testCandidates) const;

    /**
   * Find range (min, max) of the aggregated stats
   * @return Tuple of min and max values
   */
    std::tuple<DTYPE, DTYPE> _findRangeOfAggregateStats() const;
};

}   // namespace DlQuantization

#endif   // DL_QUANTIZATION_TF_ENHANCED_ENCODING_ANALYZER_H
