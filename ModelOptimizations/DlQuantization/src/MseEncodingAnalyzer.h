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

#ifndef DL_QUANTIZATION_MSE_ENCODING_ANALYZER_H
#define DL_QUANTIZATION_MSE_ENCODING_ANALYZER_H

#include "math_functions.hpp"
#include <DlQuantization/IQuantizationEncodingAnalyzer.hpp>

namespace DlQuantization
{
template <typename DTYPE>
class MseEncodingAnalyzer : public IQuantizationEncodingAnalyzer<DTYPE>
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
     * Compute the encodings using the collected histogram stats by minimizing the mean square
     * error due to quantization of the tensor.
     * @param bw Bitwidth to use for computing encodings
     * @param useSymmetricEncodings If true, compute symmetric encodings
     * @param useStrictSymmetric If true, compute symmetric encodings with even number of buckets
     * @param useUnsignedSymmetric If true, compute asymmetric encodings
     * @return Computed encoding
     */
    TfEncoding computeEncoding(uint8_t bw, bool useSymmetricEncodings, bool useStrictSymmetric,
                               bool useUnsignedSymmetric) const override;


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

    /**
     * Find range (min, max) of the aggregated stats
     * @return Tuple of min and max values
     */
    std::tuple<DTYPE, DTYPE> _findRangeOfAggregateStats() const;

    /**
     * @brief Adjust the min/max range of tensor values  by minimizing MSE due to quantization
     *
     * @param bw Bitwidth to use for computing encodings
     * @param useSymmetricEncodings If true, compute symmetric encodings
     * @param useStrictSymmetric If true, compute symmetric encodings with even number of buckets
     * @param useUnsignedSymmetric If true, compute asymmetric encodings
     * @return Tuple of <best min, best max> to be used for calculating the best delta and offset
     */
    std::tuple<DTYPE, DTYPE> _minimizeMSE(uint8_t bw, bool useSymmetricEncodings, bool useStrictSymmetric,
                                          bool useUnsignedSymmetric) const;

    /**
     * @brief Select min-max candidates from the bin edges
     *
     * @param candidates Bin edges computed from collected stats used for selecting min-max candidates
     * @param observedMin Minimum value of the histogram stats collected
     * @param observedMax  Maximum value of the histogram stats collected
     * @param minMaxCandidates Vector of pair (min and max candidates)
     */
    void _pickMinMaxCandidatesMSECalib(std::vector<DTYPE>& candidates, DTYPE observedMin, DTYPE observedMax,
                                       std::vector<std::pair<DTYPE, DTYPE>>& minMaxCandidates) const;

    /**
     * @brief Given a candidate min and max, find the mse by fake quantizing the bin centers.
     *
     * @param bw Bitwidth to use for computing encodings
     * @param binCentersPdf Vector of pair (bin centers and correspongding pdf value)
     * @param candidateMin Min value to be used for computing candidate scale and offset
     * @param candidateMax Max value to be used for computing candidate scale and offset
     * @return mse value computed for the given candidateMin and candidateMax
     */
    DTYPE _computeMSECost(uint8_t bw, std::vector<std::pair<DTYPE, DTYPE>>& binCentersPdf, DTYPE candidateMin,
                          DTYPE candidateMax, bool useSymmetricEncodings, bool useStrictSymmetric,
                          bool useUnsignedSymmetric) const;
};

}   // namespace DlQuantization

#endif   // DL_QUANTIZATION_MSE_ENCODING_ANALYZER_H 
