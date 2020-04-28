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

#include "DlEqualization/BatchNormFold.h"
#include "TensorOperations.h"
#include <iostream>

namespace AimetEqualization
{
using namespace std;

std::vector<float> BatchNormFold::fold(BNParams& bnParams, TensorParams& weightTensor, TensorParams& biasTensor,
                                       bool isBiasTensorValid, bool foldPrevLayer)
{
    // invoke weight computation with weight tensors in openCV mat format
    const int ndims = 4;

    unsigned int channelIndex = 0;
    if (foldPrevLayer)
    {
        channelIndex = 0;   // output channels
    }
    else
    {
        channelIndex = 1;   // input channels
    }

    unsigned int channels = weightTensor.shape[channelIndex];

    vector<float> bias;
    cv::Mat weightTensorMat = cv::Mat(ndims, (int*) &weightTensor.shape[0], CV_32F, weightTensor.data);

    // Sum weight matrix along 3rd and 4th axis
    cv::Mat reducedWeightMat = TensorOperations::sumAlongThirdAndFourthAxis(weightTensorMat);
    cv::Mat runningMeanMat   = cv::Mat(channels, 1, CV_32F, bnParams.runningMean);
    cv::Mat betaMat          = cv::Mat(channels, 1, CV_32F, bnParams.beta);
    cv::Mat muHat, betaHat;

    // Calculate (mu * gamma)/sigma
    vector<float> muGammaOverSigma;
    for (unsigned int i = 0; i < channels; i++)
    {
        float runningMean = 0.;
        if (foldPrevLayer && isBiasTensorValid)
            runningMean = bnParams.runningMean[i] - biasTensor.data[i];
        else
            runningMean = bnParams.runningMean[i];

        muGammaOverSigma.push_back(runningMean * bnParams.gamma[i] * (1.0f / bnParams.runningVar[i]));
    }
    cv::Mat muGammaOverSigmaMat = cv::Mat(channels, 1, CV_32F, &muGammaOverSigma[0]);

    int totalSize = 1;
    for (auto dimSize: weightTensor.shape)
        totalSize = totalSize * dimSize;

    cv::Mat reshapedWeightTensor;
    if (!foldPrevLayer)
    {
        muHat                = reducedWeightMat * muGammaOverSigmaMat;
        betaHat              = reducedWeightMat * betaMat;
        reshapedWeightTensor = TensorOperations::swapFirstTwoAxisIn4dMat(weightTensorMat);
    }
    else
    {
        muHat                = muGammaOverSigmaMat;
        betaHat              = betaMat;
        reshapedWeightTensor = weightTensorMat;
    }

    for (int i = 0; i < weightTensor.shape[0]; i++)
    {
        bias.push_back(betaHat.at<float>(i) - muHat.at<float>(i));
    }

    for (unsigned int i = 0; i < channels; i++)
    {
        cv::Mat wPerChannel = TensorOperations::getDataPerChannelIn4dMat(reshapedWeightTensor, i, AXIS_0);
        wPerChannel         = wPerChannel * bnParams.gamma[i] * (1.0f / bnParams.runningVar[i]);
    }

    if (!foldPrevLayer)
    {
        cv::Mat(TensorOperations::swapFirstTwoAxisIn4dMat(reshapedWeightTensor)).copyTo(weightTensorMat);

        if (isBiasTensorValid)
        {
            for (size_t i = 0; i < bias.size(); i++)
            {
                bias[i] += biasTensor.data[i];
            }
        }
    }

    return bias;
}

}   // namespace AimetEqualization
