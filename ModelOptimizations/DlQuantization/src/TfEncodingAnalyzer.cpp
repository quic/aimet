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

#include <cstddef>
#include <vector>

#include "DlQuantization/Quantization.hpp"
#include "math_functions.hpp"

#include "TfEncodingAnalyzer.h"

namespace DlQuantization
{
template <typename DTYPE>
void TfEncodingAnalyzer<DTYPE>::updateStats(const DTYPE* tensor, const size_t tensorSize,
                                            ComputationMode tensorCpuGpuMode)
{
    // Compute stats for the tensor being passed in
    auto currentMin  = (double) GetMin(tensor, tensorSize, tensorCpuGpuMode);
    auto currentMax  = (double) GetMax(tensor, tensorSize, tensorCpuGpuMode);

    // Update accumulated stats
    _accumulatedStats.min = std::min(_accumulatedStats.min, currentMin);
    _accumulatedStats.max = std::max(_accumulatedStats.max, currentMax);
}


template <typename DTYPE>
TfEncoding TfEncodingAnalyzer<DTYPE>::computeEncoding(uint8_t bw, bool useSymmetricEncodings,
                                                      bool useStrictSymmetric, bool useUnsignedSymmetric) const
{
    TfEncoding encoding;

    double numSteps = pow(2, bw) - 1;

    // Make sure zero value is within the range
    double newMin  = std::min(0.0, _accumulatedStats.min);
    double newMax  = std::max(0.0, _accumulatedStats.max);

    // When the min and max are too close together, nudge the maximum to meet the
    // minimum range requirement
    // This also handles the case where min==max==0 to avoid division by zero
    newMax       = std::max(newMax, newMin + MIN_RANGE);
    encoding.bw  = bw;

    // Special case for symmetric encodings. If all values are positive or 0, we can treat the
    // symmetric encodings as unsigned, which essentially translates to asymmetric
    if (useSymmetricEncodings && (newMin < 0.0))
    {

        // If we desire symmetric encodings then we need to expand either the min or max to be mirrors of each other
        // centered around 0
        newMax                          = std::max(std::abs(newMax), std::abs(newMin));
        unsigned int numPositiveSteps   = pow(2, bw - 1) - 1;
        encoding.delta = newMax / numPositiveSteps;
        encoding.offset = - (double)(numPositiveSteps + 1);
        encoding.min = encoding.offset * encoding.delta;
        encoding.max = encoding.delta * numPositiveSteps;
    }
    else
    {
        encoding.delta = (newMax - newMin) / numSteps;
        if (newMin < 0 && newMax > 0)
        {
            // Need to make sure 0-value is exactly quantizable
            // Quantization of q into b is given by:
            //     b = q / delta - offset, where
            //                             delta = (max - min)/#steps
            //                             offset = min / delta
            // For q = 0: b = -min / delta
            // Find the closest round b, and set q=0 for it
            double bZero    = round(-newMin / encoding.delta);
            bZero           = std::min(numSteps, std::max(0.0, bZero));   // just to be safe
            encoding.offset = -bZero;
        }
        else
        {
            // One of min or max is guaranteed to be zero, so 0 is exactly quantizable already
            encoding.offset = round(newMin / encoding.delta);
        }

        // Calculate 'min' and 'max' based on 'delta' and 'offset'.
        // Note this min and max can vary from the one in 'stats'. This min and max
        // can really be represented with the integer offset.
        encoding.min = encoding.delta * encoding.offset;
        // We want to calculate: max = delta * numSteps + min.
        // To avoid numerical accuracy issues on Linaro, we simplify the math.
        encoding.max = newMax - newMin + encoding.min;
    }


    return encoding;
}


// Explicit instantiations
template class TfEncodingAnalyzer<double>;

template class TfEncodingAnalyzer<float>;

}