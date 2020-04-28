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
#ifndef AIMET_SCALE_FACTOR_CALCULATOR_H
#define AIMET_SCALE_FACTOR_CALCULATOR_H

#include <opencv2/core/core.hpp>

namespace AimetEqualization
{
typedef struct RescalingParams
{
    // S_12
    cv::Mat scalingMatrix12;
    // S_23
    cv::Mat scalingMatrix23;

} RescalingParams;

class ScaleFactorCalculator
{
public:
    /**
     * @brief Internal method to compute scaling factors with the input/output layer weight tensors for
     * depth-wise separable layer case.
     * Layers connected as  : Layer1 (weightTensor1)  --> Layer2 (weightTensor2) --> Layer3 (weightTensor3)
     * @param weightTensor1 Tensor of weights (4D) of layer1
     * @param weightTensor2 Tensor of weights (4D) of layer2
     * @param weightTensor3 Tensor of weights (4D) of layer3
     * @return S12 & S23 scaling factors in opencv Mat format returned as RescalingParams type.
     * [Note]
     * weightTensor1 is assumed to be to be arranged in MxNxHxW format
     * weightTensor2 NxMxHxW format
     * weightTensor3 NxMxHxW format where,
     * M is outputChannels and N is inputChannels
     */
    static RescalingParams* ForDepthWiseSeparableLayer(const cv::Mat& weightTensor1, const cv::Mat& weightTensor2,
                                                       const cv::Mat& weightTensor3);

    /**
     * @brief Internal method to compute scaling factors with the input/output layer weight tensors for two layers
     * Layers connected as : Layer1 (weightTensor1) --> Layer2 (weightTensor2)
     * @param weightTensor1 of weights (4D) of Layer 1  in opencv Mat format
     * @param weightTensor2 of weights (4D) of Layer 2  in opencv Mat format
     * @return : Output scaling factors in opencv Mat format.
     * [Note]
     * weightTensor1 is assumed to be arranged in MxNxHxW format
     * weightTensor2 NxMxHxW format where,
     * M is outputChannels and N is inputChannels
     */
    static cv::Mat ForTwoConvLayers(const cv::Mat& weightTensor1, const cv::Mat& weightTensor2);
};


}   // namespace AimetEqualization

#endif   // AIMET_SCALE_FACTOR_CALCULATOR_H
