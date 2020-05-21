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

#include "ScaleFactorCalculator.h"
#include "TensorOperations.h"
#include <iostream>


namespace AimetEqualization
{
using namespace std;

cv::Mat ScaleFactorCalculator::ForTwoConvLayers(const cv::Mat& weightTensor1, const cv::Mat& weightTensor2)
{
    // invalid checks , zero dimension or
    // Num input channels of layer 2 not equal to num output channels of layer 1
    if (0 == weightTensor1.size[DIM_0] || 0 == weightTensor1.size[DIM_1] || 0 == weightTensor2.size[DIM_0] ||
        0 == weightTensor2.size[DIM_1] || weightTensor1.size[DIM_0] != weightTensor2.size[DIM_0])
    {
        std::cerr << "Invalid inputs" << std::endl;
        throw std::runtime_error("aborted computeScalingFactor");
    }

    // find max vectors with w1 and w2
    cv::Mat rangeVec1 = TensorOperations::computeRangeAlongFirstAxis(weightTensor1);
    cv::Mat rangeVec2 = TensorOperations::computeRangeAlongFirstAxis(weightTensor2);

    // compute S = range1/sqrt(range1.range2)

    cv::Mat sqrtMat;

    // perform element-wise multiplication on range vectors and find sqrt
    cv::sqrt((rangeVec1.mul(rangeVec2)), sqrtMat);

    // Denominator can hit zero when an element in either of the two range Mat hits zero
    // Avoid 'divide by zero' by using a value that does no scaling. i.e., 1.

    cv::Mat scalingFactorVec = cv::Mat::ones(1, rangeVec1.total(), FLOAT_32_TYPE);

    for (size_t s = 0; s < rangeVec1.total(); ++s)
    {
        if (sqrtMat.at<float>(s) != 0)
        {
            scalingFactorVec.at<float>(s) = (rangeVec1.at<float>(s)) * (1.0f / sqrtMat.at<float>(s));
        }
    }

    return scalingFactorVec;
}

AimetEqualization::RescalingParams* ScaleFactorCalculator::ForDepthWiseSeparableLayer(const cv::Mat& weightTensor1,
                                                                                      const cv::Mat& weightTensor2,
                                                                                      const cv::Mat& weightTensor3)
{
    AimetEqualization::RescalingParams* reScalingMats = new RescalingParams;

    // invalid checks
    if (0 == weightTensor1.size[DIM_0] || 0 == weightTensor1.size[DIM_1] || 0 == weightTensor2.size[DIM_0] ||
        0 == weightTensor2.size[DIM_1] || 0 == weightTensor3.size[DIM_0] || 0 == weightTensor3.size[DIM_1])
    {
        std::cerr << "Invalid inputs" << std::endl;
        throw std::runtime_error("aborted _computeScalingFactorDepthWiseSeparableLayer");
    }

    // compute S12 and S23 using :
    // S12 = range1/cubeRoot(range1 * range2 * range3)
    // S23 = cubeRoot(range1 * range2 * range3)/range3
    // assumes weightTensor1 passed as  MXNxHxW weightTensor2 and weightTensor3 as NxMxHxW
    // where, M is ouptut channels , N is input channels
    cv::Mat rangeVec1 = TensorOperations::computeRangeAlongFirstAxis(weightTensor1);
    cv::Mat rangeVec2 = TensorOperations::computeRangeAlongFirstAxis(weightTensor2);
    cv::Mat rangeVec3 = TensorOperations::computeRangeAlongFirstAxis(weightTensor3);

    cv::Mat cubeRootMat;
    // perform element-wise multiplication on range vectors and find sqrt
    cv::pow((rangeVec1.mul(rangeVec2).mul(rangeVec3)), 1.0f / 3, cubeRootMat);


    // Denominator can hit zero when an element in either sqrtMat or range2 Mat hits zero
    // Avoid 'divide by zero' by using a value that does no scaling. i.e., 1.

    reScalingMats->scalingMatrix12 = cv::Mat::ones(1, rangeVec1.total(), FLOAT_32_TYPE);
    reScalingMats->scalingMatrix23 = cv::Mat::ones(1, rangeVec2.total(), FLOAT_32_TYPE);

    for (size_t s = 0; s < rangeVec1.total(); ++s)
    {
        if ((rangeVec1.at<float>(s) != 0) && (rangeVec2.at<float>(s) != 0) && (rangeVec3.at<float>(s) != 0))
        {
            reScalingMats->scalingMatrix12.at<float>(s) = (rangeVec1.at<float>(s)) * (1.0f / cubeRootMat.at<float>(s));
        }
    }

    for (size_t s = 0; s < rangeVec2.total(); ++s)
    {
        if ((rangeVec1.at<float>(s) != 0) && (rangeVec2.at<float>(s) != 0) && (rangeVec3.at<float>(s) != 0))
        {
            reScalingMats->scalingMatrix23.at<float>(s) = (cubeRootMat.at<float>(s)) * (1.0f / rangeVec3.at<float>(s));
        }
    }

    return reScalingMats;
}

}