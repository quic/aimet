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

#include "MseEncodingAnalyzer.h"

namespace DlQuantization
{
template <typename DTYPE>
std::vector<std::tuple<double, double>> MseEncodingAnalyzer<DTYPE>::getStatsHistogram() const
{
    // Return the collected histogram data.
    return getCollectedHistogram(this->_stats);
}

template <typename DTYPE>
void MseEncodingAnalyzer<DTYPE>::updateStats(const DTYPE* tensor, const size_t tensorSize,
                                                    ComputationMode tensorCpuGpuMode)
{
    updateStats(tensor, tensorSize, tensorCpuGpuMode, nullptr);
}


template <typename DTYPE>
void MseEncodingAnalyzer<DTYPE>::updateStats(const DTYPE* tensor, const size_t tensorSize,
                                                    ComputationMode tensorCpuGpuMode,
                                                    IAllocator* allocator)
{
    this->_statsUpdated = true;

    // update pdf
    UpdatePdf(tensor, tensorSize, tensorCpuGpuMode, true, this->_stats, allocator);
}

template <typename DTYPE>
TfEncoding MseEncodingAnalyzer<DTYPE>::computeEncoding(uint8_t bw, bool useSymmetricEncodings, bool useStrictSymmetric,
                                                       bool useUnsignedSymmetric) const
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
    std::tie(aMin, aMax) = _minimizeMSE(bw, useSymmetricEncodings, useStrictSymmetric, useUnsignedSymmetric);

    // After Min and Max adjustment, the requirement that 0 be an exactly
    // representable value must be met. Hence, extend the interval
    // [aMin, aMax] to ensure that it contains 0.
    aMin = std::min(aMin, DTYPE(0.f));
    aMax = std::max(aMax, DTYPE(0.f));

    assert(aMin <= aMax && "min must not be bigger than max");

    return getComputedEncodings(bw, aMin, aMax, useSymmetricEncodings, useStrictSymmetric, useUnsignedSymmetric);
}

template <typename DTYPE>
std::tuple<DTYPE, DTYPE> MseEncodingAnalyzer<DTYPE>::_findRangeOfAggregateStats() const
{
    return findOriginalRange<DTYPE>(this->_stats);
}

template <typename DTYPE>
std::tuple<DTYPE, DTYPE> MseEncodingAnalyzer<DTYPE>::_minimizeMSE(uint8_t bw, bool useSymmetricEncodings,
                                                                  bool useStrictSymmetric,
                                                                  bool useUnsignedSymmetric) const
{
    // Histogram bin width.
    const float histBinWidth = this->_stats.xLeft[1] - this->_stats.xLeft[0];
    DTYPE histMin            = this->_stats.xLeft[0];
    DTYPE histMax            = this->_stats.xLeft[PDF_SIZE - 1] + histBinWidth;

    // Find the true range of our collected stats
    DTYPE minVal, maxVal;
    std::tie(minVal, maxVal) = _findRangeOfAggregateStats();
    maxVal                   = maxVal + histBinWidth;

    // Compute bin edges. They are used for selecting min-max candidates
    std::vector<DTYPE> binEdges;
    binEdges.push_back(minVal);
    for (auto i = histMin; (i <= histMax); i += histBinWidth)
    {
        if ((i >= minVal) && (i <= maxVal))
        {
            binEdges.push_back(i);
        }
    }

    // Select min-max candidates
    std::vector<std::pair<DTYPE, DTYPE>> minMaxCandidates;
    _pickMinMaxCandidatesMSECalib(binEdges, minVal, maxVal, minMaxCandidates);

    // Compute the bin centers and correspongding pdf value. They are used for computing MSE cost
    DTYPE pdfStart = this->_stats.xLeft[0];
    DTYPE pdfStep  = this->_stats.xLeft[1] - this->_stats.xLeft[0];

    const int numBinCenters = binEdges.size() - 1;
    std::vector<std::pair<DTYPE, DTYPE>> binCentersPdf(numBinCenters);
    binCentersPdf[0].first = minVal + histBinWidth / 2;
    // Calculate the index of the bucket which contains the corresponding pdf value.
    int pdfBinInd           = (int) std::floor((binCentersPdf[0].first - pdfStart) / pdfStep);
    pdfBinInd               = std::min(std::max(0, pdfBinInd), PDF_SIZE - 1);
    binCentersPdf[0].second = this->_stats.pdf[pdfBinInd];
    for (auto i = 1; i < numBinCenters; i++)
    {
        binCentersPdf[i].first  = binCentersPdf[i - 1].first + histBinWidth;
        pdfBinInd               = (int) std::floor((binCentersPdf[i].first - pdfStart) / pdfStep);
        pdfBinInd               = std::min(std::max(0, pdfBinInd), PDF_SIZE - 1);
        binCentersPdf[i].second = this->_stats.pdf[pdfBinInd];
    }

    // Compute MSE for each min-max candidate and find the optimal candidate for
    // which MSE cost is least
    float mseMin = std::numeric_limits<float>::max();
    std::tuple<DTYPE, DTYPE> bestCandidate(minVal, maxVal);
    for (auto& c: minMaxCandidates)
    {
        DTYPE mse = _computeMSECost(bw, binCentersPdf, c.first, c.second, useSymmetricEncodings, useStrictSymmetric,
                                    useUnsignedSymmetric);
        if (mse < mseMin)
        {
            mseMin        = mse;
            bestCandidate = c;
        }
    }
    return bestCandidate;
}

template <typename DTYPE>
void MseEncodingAnalyzer<DTYPE>::_pickMinMaxCandidatesMSECalib(
    std::vector<DTYPE>& candidates, DTYPE observedMin, DTYPE observedMax,
    std::vector<std::pair<DTYPE, DTYPE>>& minMaxCandidates) const
{
    std::vector<DTYPE> minCandidates;
    std::vector<DTYPE> maxCandidates;

    // Assign all candidates less than 0 to 'Min' and all candidates greater than
    // 0 to 'Max'. '0' is excluded because {0, 0} is not a valid min-max
    // candidate.
    for (auto& c: candidates)
    {
        if (c < 0)
        {
            minCandidates.push_back(c);
        }
        else if (c > 0)
        {
            maxCandidates.push_back(c);
        }
    }

    // Include '0' in min and max candidates list. {0, 0} will be removed later.
    minCandidates.push_back(0);
    maxCandidates.push_back(0);
    for (auto& i: minCandidates)
    {
        for (auto& j: maxCandidates)
        {
            minMaxCandidates.push_back({i, j});
        }
    }
    // Remove the last element '{0, 0}' which is not suitable as min-max
    // candidate.
    minMaxCandidates.pop_back();
}

template <typename DTYPE>
DTYPE MseEncodingAnalyzer<DTYPE>::_computeMSECost(uint8_t bw, std::vector<std::pair<DTYPE, DTYPE>>& binCentersPdf,
                                                  DTYPE candidateMin, DTYPE candidateMax, bool useSymmetricEncodings,
                                                  bool useStrictSymmetric, bool useUnsignedSymmetric) const
{
    // Compute the scale and offset based on the min and max provided
    TfEncoding encoding = getComputedEncodings(bw, candidateMin, candidateMax, useSymmetricEncodings,
                                               useStrictSymmetric, useUnsignedSymmetric);
    // Apply Fake quantization on the bin centers and compute MSE cost
    DTYPE weightedSquareErr = 0;
    for (int i = 0; i < binCentersPdf.size(); i++)
    {
        // The floating point value in the middle of this bucket.
        DTYPE floatVal        = binCentersPdf[i].first;
        DTYPE clampedFloatVal = std::max(candidateMin, std::min(floatVal, candidateMax));
        // The quantized equivalent.
        int quantized = (int) round(clampedFloatVal / encoding.delta - encoding.offset);
        // The de-quantized value: this is 'floatVal' plus the quantization error.
        DTYPE dequantized = encoding.delta * (quantized + encoding.offset);
        // The quantization cost is the MSE.
        weightedSquareErr += binCentersPdf[i].second * pow(floatVal - dequantized, 2);
    }
    return weightedSquareErr;
}

// Explicit instantiations
template class MseEncodingAnalyzer<double>;

template class MseEncodingAnalyzer<float>;

}   // namespace DlQuantization 
