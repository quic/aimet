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

#include "DlEqualization/CrossLayerScaling.h"
#include "ScaleFactorCalculator.h"
#include "TensorOperations.h"
#include <iostream>


namespace AimetEqualization
{
using namespace std;

std::vector<float> CrossLayerScaling::scaleLayerParams(EqualizationParams& prevLayer, EqualizationParams& currLayer)
{
    // invoke weight computation with weight tensors in opencv mat format
    const int ndims = 4;
    int N           = prevLayer.weightShape[0];   // output channels

    cv::Mat weightTensor1 = cv::Mat(ndims, (int*) &prevLayer.weightShape[0], CV_32F, prevLayer.weight);
    cv::Mat biasTensor1;
    if (!prevLayer.isBiasNone)
    {
        biasTensor1 = cv::Mat(N, 1, CV_32F, (float*) &prevLayer.bias[0]);
    }

    cv::Mat weightTensor2        = cv::Mat(ndims, (int*) &currLayer.weightShape[0], CV_32F, currLayer.weight);
    cv::Mat flippedWeightTensor2 = TensorOperations::swapFirstTwoAxisIn4dMat(weightTensor2);

    cv::Mat scalingFactors = ScaleFactorCalculator::ForTwoConvLayers(weightTensor1, flippedWeightTensor2);

    for (size_t s = 0; s < scalingFactors.total(); ++s)
    {
        // Scaling Weight Matrix of prev layer
        cv::Mat w1PerChannel = TensorOperations::getDataPerChannelIn4dMat(weightTensor1, s, AXIS_0);
        w1PerChannel         = w1PerChannel * (1.0f / scalingFactors.at<float>(s));

        // Scaling the bias of prev layer
        if (!prevLayer.isBiasNone)
            biasTensor1.at<float>(s) = biasTensor1.at<float>(s) * (1.0f / scalingFactors.at<float>(s));

        // Scaling Weight Matrix of curr layer
        cv::Mat w2PerChannel = TensorOperations::getDataPerChannelIn4dMat(flippedWeightTensor2, s, AXIS_0);

        w2PerChannel = w2PerChannel * scalingFactors.at<float>(s);
    }

    cv::Mat(TensorOperations::swapFirstTwoAxisIn4dMat(flippedWeightTensor2)).copyTo(weightTensor2);

    // Convert scalingFactors to vector
    std::vector<float> scalingVector(scalingFactors.begin<float>(), scalingFactors.end<float>());

    return scalingVector;
}


AimetEqualization::CrossLayerScaling::RescalingParamsVectors
CrossLayerScaling::scaleDepthWiseSeparableLayer(AimetEqualization::EqualizationParams& prevLayer,
                                                AimetEqualization::EqualizationParams& currLayer,
                                                AimetEqualization::EqualizationParams& nextLayer)
{
    const int ndims = 4;
    int N           = prevLayer.weightShape[0];   // output channels

    cv::Mat weightTensor1 = cv::Mat(ndims, (int*) &prevLayer.weightShape[0], CV_32F, prevLayer.weight);
    cv::Mat biasTensor1;
    if (!prevLayer.isBiasNone)
        biasTensor1 = cv::Mat(N, 1, CV_32F, (float*) &prevLayer.bias[0]);

    cv::Mat weightTensor2        = cv::Mat(ndims, (int*) &currLayer.weightShape[0], CV_32F, currLayer.weight);
    cv::Mat flippedWeightTensor2 = TensorOperations::swapFirstTwoAxisIn4dMat(weightTensor2);
    cv::Mat biasTensor2;
    if (!currLayer.isBiasNone)
        biasTensor2 = cv::Mat(N, 1, CV_32F, (float*) &currLayer.bias[0]);

    cv::Mat weightTensor3        = cv::Mat(ndims, (int*) &nextLayer.weightShape[0], CV_32F, nextLayer.weight);
    cv::Mat flippedWeightTensor3 = TensorOperations::swapFirstTwoAxisIn4dMat(weightTensor3);

    RescalingParams* pReScalingMats =
        ScaleFactorCalculator::ForDepthWiseSeparableLayer(weightTensor1, weightTensor2, flippedWeightTensor3);


    for (size_t s = 0; s < pReScalingMats->scalingMatrix12.total(); ++s)
    {
        // Scaling Weight Matrix of prev layer with S12
        cv::Mat w1PerChannel = TensorOperations::getDataPerChannelIn4dMat(weightTensor1, s, AXIS_0);
        w1PerChannel         = w1PerChannel * (1.0f / pReScalingMats->scalingMatrix12.at<float>(s));

        // Scaling the bias of prev layer with S12
        if (!prevLayer.isBiasNone)
            biasTensor1.at<float>(s) = biasTensor1.at<float>(s) * (1.0f / pReScalingMats->scalingMatrix12.at<float>(s));

        // Scaling Weight Matrix of curr layer with S12
        cv::Mat w2PerChannel = TensorOperations::getDataPerChannelIn4dMat(weightTensor2, s, AXIS_0);
        w2PerChannel         = w2PerChannel * pReScalingMats->scalingMatrix12.at<float>(s);
    }

    for (size_t s = 0; s < pReScalingMats->scalingMatrix23.total(); ++s)
    {
        // Scaling Weight Matrix of prev layer with S23
        cv::Mat w2PerChannel = TensorOperations::getDataPerChannelIn4dMat(weightTensor2, s, AXIS_0);
        w2PerChannel         = w2PerChannel * (1.0f / pReScalingMats->scalingMatrix23.at<float>(s));

        // Scaling the bias of curr layer with S23
        if (!currLayer.isBiasNone)
            biasTensor2.at<float>(s) = biasTensor2.at<float>(s) * (1.0f / pReScalingMats->scalingMatrix23.at<float>(s));

        // Scaling Weight Matrix of curr layer with S23
        cv::Mat w3PerChannel = TensorOperations::getDataPerChannelIn4dMat(flippedWeightTensor3, s, AXIS_0);
        w3PerChannel         = w3PerChannel * pReScalingMats->scalingMatrix23.at<float>(s);
    }

    cv::Mat(TensorOperations::swapFirstTwoAxisIn4dMat(flippedWeightTensor3)).copyTo(weightTensor3);

    // return pReScalingMats as vectors
    CrossLayerScaling::RescalingParamsVectors scalingVectors;

    scalingVectors.scalingMatrix12.assign(pReScalingMats->scalingMatrix12.begin<float>(),
                                          pReScalingMats->scalingMatrix12.end<float>());

    scalingVectors.scalingMatrix23.assign(pReScalingMats->scalingMatrix23.begin<float>(),
                                          pReScalingMats->scalingMatrix23.end<float>());

    return scalingVectors;
}
}   // namespace AimetEqualization
