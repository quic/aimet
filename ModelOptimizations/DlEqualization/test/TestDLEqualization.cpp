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
#include "DlEqualization/BatchNormFoldForPython.h"
#include "DlEqualization/CrossLayerScalingForPython.h"
#include "DlEqualization/HighBiasFold.h"
#include <gtest/gtest.h>
#include <vector>

#include "DlEqualization/BiasCorrection.h"
#include "DlEqualization/BiasCorrectionForPython.h"
#include "DlEqualization/def.h"
#include <DlEqualization/CrossLayerScalingForPython.h>
#include <DlEqualization/HighBiasFoldForPython.h>
#include <opencv2/core/core.hpp>


class TestDLEqualization : public ::testing::Test
{
};

TEST(TestCrossLayerScaling, SanityTest)
{
    std::vector<float> testWeightTensor1, testWeightTensor2;
    for (uint8_t ic = 0; ic < 2 * 2 * 3 * 1; ++ic)
    {
        testWeightTensor1.push_back(ic + 1);
        testWeightTensor2.push_back(ic * 2 + 1);
    }


    AimetEqualization::EqualizationParamsForPython prevLayer, currLayer;

    prevLayer.weight      = testWeightTensor1;
    prevLayer.weightShape = std::vector<int> {2, 2, 3, 1};
    prevLayer.bias        = std::vector<float> {1, 2};
    prevLayer.isBiasNone  = false;

    currLayer.weight      = testWeightTensor2;
    currLayer.weightShape = std::vector<int> {2, 2, 3, 1};

    // invoke cross layer scaling api
    AimetEqualization::CrossLayerScalingForPython::scaleLayerParams(prevLayer, currLayer);
    EXPECT_FLOAT_EQ(prevLayer.weight[0], 1.0f / currLayer.weight[0]);
    EXPECT_FLOAT_EQ(prevLayer.weight[0], prevLayer.bias[0]);
}


TEST(TestCrossLayerScaling, SanityTestForDepthWiseSeparableLayer)
{
    std::vector<float> testWeightTensor1, testWeightTensor2, testWeightTensor3;
    for (uint8_t ic = 0; ic < 2 * 2 * 3 * 1; ++ic)
    {
        testWeightTensor1.push_back(ic + 1);
        testWeightTensor2.push_back(ic * 2 + 1);
        testWeightTensor3.push_back(ic + 1);
    }


    AimetEqualization::EqualizationParamsForPython prevLayer, currLayer, nextLayer;

    prevLayer.weight      = testWeightTensor1;
    prevLayer.weightShape = std::vector<int> {2, 2, 3, 1};
    prevLayer.bias        = std::vector<float> {1, 2};
    prevLayer.isBiasNone  = false;

    currLayer.weight      = testWeightTensor2;
    currLayer.weightShape = std::vector<int> {2, 2, 3, 1};
    currLayer.bias        = std::vector<float> {1, 3};
    currLayer.isBiasNone  = false;

    nextLayer.weight      = testWeightTensor3;
    nextLayer.weightShape = std::vector<int> {2, 2, 3, 1};

    // invoke cross layer scaling api
    AimetEqualization::CrossLayerScalingForPython::scaleDepthWiseSeparableLayer(prevLayer, currLayer, nextLayer);
    EXPECT_FLOAT_EQ(0.7137658, 1.0f / prevLayer.weight[0]);
    EXPECT_FLOAT_EQ(prevLayer.weight[0], prevLayer.bias[0]);
    EXPECT_FLOAT_EQ(0.93401319, nextLayer.weight[0]);
}


TEST(TestBatchNormFold, SanityTestBatchNormFold)
{
    std::vector<float> testWeightTensor, gamma, beta, mean, var, bias;

    for (uint8_t ic = 0; ic < 3 * 2 * 3 * 1; ++ic)
    {
        testWeightTensor.push_back(ic + 1);
    }

    AimetEqualization::BNParamsForPython bnLayerParams;

    AimetEqualization::TensorParamsForPython weightParams;
    weightParams.data  = testWeightTensor;
    weightParams.shape = std::vector<int> {3, 2, 3, 1};

    AimetEqualization::TensorParamsForPython biasTensor;
    biasTensor.data  = {1, 2, 3};
    biasTensor.shape = std::vector<int> {3};

    for (uint8_t ic = 0; ic < 2; ++ic)
    {
        gamma.push_back(ic * 2 + 1);
        beta.push_back(ic + 2);
        mean.push_back(ic + 1);
        var.push_back(ic + 1);
    }

    bnLayerParams.runningMean = mean;
    bnLayerParams.runningVar  = var;
    bnLayerParams.beta        = beta;
    bnLayerParams.gamma       = gamma;

    bias = AimetEqualization::BatchNormFoldForPython::fold(bnLayerParams, weightParams, biasTensor, true, true);
    EXPECT_FLOAT_EQ(biasTensor.data[0] + 1, bias[0]);
}

TEST(TestHighBiasFold, SanityTestHighBiasFoldActivationRelu)
{
    std::vector<float> testWeightTensor, gamma, prev_layer_bias, curr_layer_bias, var, bias, beta;

    for (uint8_t ic = 0; ic < 2 * 2 * 3 * 1; ++ic)
    {
        testWeightTensor.push_back(ic + 1);
    }

    for (uint8_t ic = 0; ic < 2; ++ic)
    {
        gamma.push_back(ic * 2 + 1);
        beta.push_back(ic * 3 - 2);
        prev_layer_bias.push_back(ic + 2);
        curr_layer_bias.push_back(ic + 1);
    }

    AimetEqualization::LayerParamsForPython prevLayer, currLayer;
    AimetEqualization::BNParamsHighBiasFoldForPython prevLayerBNParams;
    currLayer.weight      = testWeightTensor;
    currLayer.weightShape = std::vector<int> {2, 2, 3, 1};
    currLayer.bias        = curr_layer_bias;

    prevLayer.bias          = prev_layer_bias;
    prevLayerBNParams.gamma = gamma;
    prevLayerBNParams.beta  = beta;

    prevLayer.activationIsRelu = true;
    AimetEqualization::HighBiasFoldForPython::updateBias(prevLayer, currLayer, prevLayerBNParams);
}

TEST(TestBiasCorrection, CorrectBias)
{
    std::vector<float> o1, o2;
    AimetEqualization::TensorParam y1, y2, bias;
    std::vector<int> outputShape = std::vector<int> {2, 2, 3, 1};
    std::vector<float> biasData  = std::vector<float> {10, 10};
    for (uint8_t ic = 0; ic < 12; ++ic)
    {
        o1.push_back(ic + 1);
        o2.push_back(ic);
    }
    y1.data   = &o1[0];
    y1.shape  = outputShape;
    y2.data   = &o2[0];
    y2.shape  = outputShape;
    bias.data = &biasData[0];
    AimetEqualization::BiasCorrection obj;
    obj.storePreActivationOutput(y2);
    obj.storePreActivationOutput(y2);


    obj.storeQuantizedPreActivationOutput(y1);
    obj.storeQuantizedPreActivationOutput(y1);

    obj.correctBias(bias);

    EXPECT_FLOAT_EQ(bias.data[0], 9);
    EXPECT_FLOAT_EQ(bias.data[1], 9);
}

TEST(TestBiasCorrection, CorrectBiasBNParams)
{
    std::vector<float> quantizedWeights, weight, bias, beta, gamma;
    for (uint8_t ic = 0; ic < 3 * 2 * 3 * 1; ++ic)
    {
        quantizedWeights.push_back(ic + 1);
        weight.push_back(ic);
    }

    for (uint8_t ic = 0; ic < 2; ++ic)
    {
        gamma.push_back(ic * 2 + 1);
        beta.push_back(ic * 3 - 2);
    }

    AimetEqualization::TensorParam quantizedWeightsTensor, weightsTensor, biasTensor;
    quantizedWeightsTensor.data  = &quantizedWeights[0];
    quantizedWeightsTensor.shape = std::vector<int> {3, 2, 3, 1};

    weightsTensor.data  = &weight[0];
    weightsTensor.shape = std::vector<int> {3, 2, 3, 1};

    bias            = std::vector<float> {5.0, 5.0, 5.0};
    biasTensor.data = &bias[0];

    AimetEqualization::BnParamsBiasCorr bnParams;
    bnParams.beta  = &beta[0];
    bnParams.gamma = &gamma[0];

    // ReLu Activation

    bias            = std::vector<float> {5.0, 5.0, 5.0};
    biasTensor.data = &bias[0];

    AimetEqualization::BnBasedBiasCorrection::correctBias(biasTensor, quantizedWeightsTensor, weightsTensor, bnParams,
                                                          AimetEqualization::ActivationType::relu);

    EXPECT_FLOAT_EQ(biasTensor.data[1], -0.3135972);
    EXPECT_FLOAT_EQ(biasTensor.data[2], -0.3135972);
}


TEST(TestBiasCorrection, CorrectBiasBNParamsNoActivation)
{
    std::vector<float> quantizedWeights, weight, bias, beta, gamma;
    for (uint8_t ic = 0; ic < 3 * 2 * 3 * 1; ++ic)
    {
        quantizedWeights.push_back(ic + 1);
        weight.push_back(ic);
    }

    for (uint8_t ic = 0; ic < 2; ++ic)
    {
        gamma.push_back(ic * 2 + 1);
        beta.push_back(ic * 3 - 2);
    }

    AimetEqualization::TensorParam quantizedWeightsTensor, weightsTensor, biasTensor;
    quantizedWeightsTensor.data  = &quantizedWeights[0];
    quantizedWeightsTensor.shape = std::vector<int> {3, 2, 3, 1};

    weightsTensor.data  = &weight[0];
    weightsTensor.shape = std::vector<int> {3, 2, 3, 1};

    bias            = std::vector<float> {5.0, 5.0, 5.0};
    biasTensor.data = &bias[0];

    AimetEqualization::BnParamsBiasCorr bnParams;
    bnParams.beta  = &beta[0];
    bnParams.gamma = &gamma[0];

    // No activation
    AimetEqualization::BnBasedBiasCorrection::correctBias(biasTensor, quantizedWeightsTensor, weightsTensor, bnParams,
                                                          AimetEqualization::ActivationType::noActivation);

    EXPECT_FLOAT_EQ(biasTensor.data[1], 8);
    EXPECT_FLOAT_EQ(biasTensor.data[2], 8);
}