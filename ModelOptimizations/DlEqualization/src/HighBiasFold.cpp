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
#include "DlEqualization/HighBiasFold.h"
#include "TensorOperations.h"
#include <iostream>


namespace AimetEqualization
{
using namespace std;

void HighBiasFold::updateBias(AimetEqualization::LayerParams& prevLayerParams,
                              AimetEqualization::LayerParams& currLayerParams,
                              AimetEqualization::BNParamsHighBiasFold& prevLayerBNParams)
{
    int outputShape = currLayerParams.weightShape[1];
    if (outputShape == 1)
        outputShape = currLayerParams.weightShape[0];

    const int nDims = 4;
    std::vector<float> absorbBias;

    if (!prevLayerParams.activationIsRelu)
    {
        absorbBias.assign(prevLayerBNParams.beta, prevLayerBNParams.beta + outputShape);
    }
    else
    {
        for (int i = 0; i < outputShape; i++)
        {
            float c = (prevLayerBNParams.beta[i] - 3 * fabs(prevLayerBNParams.gamma[i])) > 0
                          ? (prevLayerBNParams.beta[i] - 3 * fabs(prevLayerBNParams.gamma[i]))
                          : 0;

            absorbBias.push_back(c);
        }
    }
    for (int i = 0; i < outputShape; i++)
    {
        prevLayerParams.bias[i] = prevLayerParams.bias[i] - absorbBias[i];
    }
    cv::Mat weightTensor = cv::Mat(nDims, (int*) &currLayerParams.weightShape[0], CV_32F, currLayerParams.weight);

    cv::Mat reducedWeightMat = TensorOperations::sumAlongThirdAndFourthAxis(weightTensor);
    cv::Mat absorbBiasMat    = cv::Mat(outputShape, 1, CV_32F, (float*) &absorbBias[0]);

    cv::Mat biasCorrectionMat;
    if (reducedWeightMat.size[1] == 1)
        biasCorrectionMat = reducedWeightMat.mul(absorbBiasMat);
    else
        biasCorrectionMat = reducedWeightMat * absorbBiasMat;

    cv::Mat biasCurrLayer = cv::Mat(currLayerParams.weightShape[0], 1, CV_32F, (float*) &currLayerParams.bias[0]);
    biasCurrLayer += biasCorrectionMat;
}
}