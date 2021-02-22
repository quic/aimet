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
    auto current_min = (double) GetMin(tensor, tensorSize, tensorCpuGpuMode);
    auto current_max = (double) GetMax(tensor, tensorSize, tensorCpuGpuMode);

    // Update accumulated stats
    _accumulatedStats.min = std::min(_accumulatedStats.min, current_min);
    _accumulatedStats.max = std::max(_accumulatedStats.max, current_max);
}


template <typename DTYPE>
TfEncoding TfEncodingAnalyzer<DTYPE>::computeEncoding(uint8_t bw, bool useSymmetricEncodings,
                                                      bool useStrictSymmetric, bool useUnsignedSymmetric) const
{
    TfEncoding encoding;

    double num_steps = pow(2, bw) - 1;

    // Make sure zero value is within the range
    double new_min = std::min(0.0, _accumulatedStats.min);
    double new_max = std::max(0.0, _accumulatedStats.max);

    // When the min and max are too close together, nudge the maximum to meet the
    // minimum range requirement
    // This also handles the case where min==max==0 to avoid division by zero
    new_max = std::max(new_max, new_min + MIN_RANGE);
    encoding.bw  = bw;

    // Special case for symmetric encodings. If all values are positive or 0, we can treat the
    // symmetric encodings as unsigned, which essentially translates to asymmetric
    if (useSymmetricEncodings && (new_min < 0.0))
    {

        // If we desire symmetric encodings then we need to expand either the min or max to be mirrors of each other
        // centered around 0
        new_max = std::max(std::abs(new_max), std::abs(new_min));
        unsigned int num_positive_steps = pow(2, bw - 1) - 1;
        encoding.delta = new_max / num_positive_steps;
        encoding.offset = - (double)(num_positive_steps + 1);
        encoding.min = encoding.offset * encoding.delta;
        encoding.max = encoding.delta * num_positive_steps;
    }
    else
    {
        encoding.delta = (new_max - new_min) / num_steps;
        if (new_min < 0 && new_max > 0)
        {
            // Need to make sure 0-value is exactly quantizable
            // Quantization of q into b is given by:
            //     b = q / delta - offset, where
            //                             delta = (max - min)/#steps
            //                             offset = min / delta
            // For q = 0: b = -min / delta
            // Find the closest round b, and set q=0 for it
            double b_zero   = round(-new_min / encoding.delta);
            b_zero          = std::min(num_steps, std::max(0.0, b_zero));   // just to be safe
            encoding.offset = -b_zero;
        }
        else
        {
            // One of min or max is guaranteed to be zero, so 0 is exactly quantizable already
            encoding.offset = round(new_min / encoding.delta);
        }

        // Calculate 'min' and 'max' based on 'delta' and 'offset'.
        // Note this min and max can vary from the one in 'stats'. This min and max
        // can really be represented with the integer offset.
        encoding.min = encoding.delta * encoding.offset;
        // We want to calculate: max = delta * num_steps + min.
        // To avoid numerical accuracy issues on Linaro, we simplify the math.
        encoding.max = new_max - new_min + encoding.min;
    }


    return encoding;
}


// Explicit instantiations
template class TfEncodingAnalyzer<double>;

template class TfEncodingAnalyzer<float>;

}