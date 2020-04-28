//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2016-2017, Qualcomm Innovation Center, Inc. All rights reserved.
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


#ifndef I_QUANTIZATION_ALGORITHM
#define I_QUANTIZATION_ALGORITHM

#include "DlQuantization/Quantization.hpp"

namespace DlQuantization
{
/**
 * @brief This is the interface of a quantization algorithm.
 *
 * A quantization algorithm has be offer the capability of gathering statistical
 * data from tensors and turn these statistics into a fixed point encoding.
 */
template <typename DTYPE>
class IQuantizationAlgorithm
{
public:
    /**
     * @brief Update the internal statistical data for a given set of tensors.
     */
    virtual void UpdateStatsModeSpecific(const std::string& layer, LayerInOut mode_in_out,
                                         const std::vector<const DTYPE*>& activations,
                                         const std::vector<size_t>& count) = 0;

    /**
     * @brief Turn the internal statistics into a fixed point format.
     */
    virtual void StatsToFxpFormat(const std::string& layer, LayerInOut mode_in_out, int bw,
                                  std::vector<TfEncoding>& encoding) = 0;

    /**
     * @brief Calculate an encoding suitable for this number distribution.
     *
     * Don't remember anything (forget the statistical data and forget the
     * encoding).
     */
    virtual void NumberDistributionToFxpFormat(int bw, const DTYPE* data, size_t count, TfEncoding& encoding) = 0;

    /**
     * @brief Calculate an encoding, given the bit-width and min/max.
     */
    virtual void ComputeDeltaAndOffsetModeSpecific(int bw, double& min, double& max, double& delta, double& offset) = 0;

    virtual ~IQuantizationAlgorithm() = default;
};

}   // End of namespace DlQuantization

#endif   // I_QUANTIZATION_ALGORITHM
