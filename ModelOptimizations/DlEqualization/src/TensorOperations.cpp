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

#include "TensorOperations.h"

namespace AimetEqualization
{
using namespace std;

cv::Mat TensorOperations::swapFirstTwoAxisIn4dMat(const cv::Mat& input4dMat)
{
    cv::Mat stackedMat;
    // go through all indices on second axis
    for (int i = 0; i < input4dMat.size[DIM_1]; ++i)
    {
        // get data per channel and stack them up
        cv::Mat chData = getDataPerChannelIn4dMat(input4dMat, i, AXIS_1);
        stackedMat.push_back(chData);
    }

    // @todo - check if there is a better way to do this
    // in the new Mat we will flip the first and second dimension,look below.
    int newDim[] = {input4dMat.size[DIM_1], input4dMat.size[DIM_0], input4dMat.size[DIM_2], input4dMat.size[DIM_3]};
    cv::Mat flippedMat(NUM_DIMENSIONS_IN_WEIGHT_TENSOR, newDim, FLOAT_32_TYPE, stackedMat.data);
    flippedMat = flippedMat.clone();

    return flippedMat;
}


cv::Mat TensorOperations::getDataPerChannelIn4dMat(const cv::Mat& inputMat, const int& channelIndex, axisType axis)
{
    int outputChannels = inputMat.size[DIM_0];
    int inputChannels  = inputMat.size[DIM_1];
    int height         = inputMat.size[DIM_2];
    int width          = inputMat.size[DIM_3];
    cv::Mat dataPerChannel;

    if (0 == inputChannels || 0 == outputChannels)
    {
        std::cerr << "Invalid inputs, input channels do not match output channel" << std::endl;
        throw std::runtime_error("aborted getDataPerChannelIn4DMat");
    }

    switch (axis)
    {
    case AXIS_0:
    {
        //@todo could skip this for linear mat
        int sz[]             = {inputChannels, height, width};
        float* dataOffsetPtr = (float*) (inputMat.data + inputMat.step[DIM_0] * channelIndex);
        // extract 1 NXHxW data blob for a given M
        dataPerChannel = cv::Mat(SUB_MAT_DIMENSION, sz, FLOAT_32_TYPE, (float*) dataOffsetPtr);
    }
    break;

    case AXIS_1:
    {
        // stack up M '1xHxW' data blobs for a given N(channelIndex)
        int sz[] = {1, height, width};
        for (int m = 0; m < outputChannels; m++)
        {
            float* dataOffsetPtr =
                (float*) (inputMat.data + inputMat.step[DIM_0] * m + inputMat.step[1] * channelIndex);
            dataPerChannel.push_back(cv::Mat(SUB_MAT_DIMENSION, sz, FLOAT_32_TYPE, ((float*) dataOffsetPtr)));
        }
    }
    break;

    default:
    {
        std::cerr << "Invalid axis" << std::endl;
        throw std::runtime_error("aborted _getDataPerChannelIn4DMat");
    }
    }

    return dataPerChannel;
}


cv::Mat TensorOperations::sumAlongThirdAndFourthAxis(cv::Mat inputTensor)
{
    const int outputChannels = inputTensor.size[0];
    const int inputChannels  = inputTensor.size[1];
    const int height         = inputTensor.size[2];
    const int width          = inputTensor.size[3];

    cv::Mat reducedMat = cv::Mat::zeros(outputChannels, inputChannels, CV_32F);

    for (int i = 0; i < outputChannels; i++)
    {
        cv::Mat dataPerChannel = getDataPerChannelIn4dMat(inputTensor, i, AXIS_0);
        for (int j = 0; j < inputChannels; j++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    reducedMat.at<float>(i, j) += dataPerChannel.at<float>(j, h, w);
                }
            }
        }
    }
    return reducedMat;
}

cv::Mat TensorOperations::computeRangeAlongFirstAxis(const cv::Mat& weightTensor)
{
    int numChannels = weightTensor.size[DIM_0];

    // 1D vector to store range values across channels for a given layer
    cv::Mat rangeVec(cv::Size(1, numChannels), FLOAT_32_TYPE);

    for (int ch = 0; ch < numChannels; ++ch)
    {
        // always fixed as axis_0
        cv::Mat dataPerChannel = getDataPerChannelIn4dMat(weightTensor, ch, AXIS_0);

        double maxVal = 0;
        int maxIdx[SUB_MAT_DIMENSION];

        // Take abs only before finding max in subMatrix
        // use nullptr for min value/index params, we don't need min value/index.
        cv::minMaxIdx(abs(dataPerChannel), nullptr, &maxVal, nullptr, maxIdx);
        rangeVec.at<float>(ch) = maxVal;
    }

    return rangeVec;
}


cv::Mat TensorOperations::sumAlongSecondThirdAxis(cv::Mat inputTensor)
{
    const int outputChannels = inputTensor.size[0];
    const int height         = inputTensor.size[1];
    const int width          = inputTensor.size[2];

    cv::Mat reducedMat = cv::Mat::zeros(outputChannels, 1, CV_64F);

    for (int i = 0; i < outputChannels; i++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                reducedMat.at<double>(i, 0) += inputTensor.at<double>(i, h, w);
            }
        }
    }
    return reducedMat;
}
}