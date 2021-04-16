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

#include "DlEqualization/BiasCorrection.h"
#include "TensorOperations.h"
#include <iostream>


namespace AimetEqualization
{
using namespace std;

float BnBasedBiasCorrection::_phiX(const float x)
{
    return 1.0 / (sqrt(2 * M_PI)) * exp(-0.5 * x * x);
}

float BnBasedBiasCorrection::_normalCDF(float x)   // Phi(-Infinity, x) aka N(x)
{
    return std::erfc(-x / std::sqrt(2)) / 2;
}

float BnBasedBiasCorrection::calcExpectationPerChannel(const int a, const int b, const float gamma, const float beta)
{
    //      = a if x < a
    // f(x) = x if a < x < b
    //      = b is x > b
    float expectation = 0.0;
    // for relu6 b == 6; Calc Expectation for ReLu6
    if (b == 6)
    {
        float Z = _normalCDF((b - beta) / gamma) - _normalCDF((a - beta) / gamma);
        float z = _phiX((a - beta) / gamma) - _phiX((b - beta) / gamma);

        expectation =
            gamma * z + beta * Z + a * _normalCDF((a - beta) / gamma) + b * (1 - _normalCDF((b - beta) / gamma));
    }
    // Calc expectation for relu
    else
    {
        expectation = beta * (1 - _normalCDF(-beta / gamma)) + gamma * _phiX(-beta / gamma);
    }

    return expectation;
}

void BnBasedBiasCorrection::correctBias(TensorParam& bias, TensorParam& quantizedWeights, TensorParam& weights,
                                        BnParamsBiasCorr& bnParams, ActivationType activation)
{
    if (quantizedWeights.shape[0] != weights.shape[0] || quantizedWeights.shape[1] != weights.shape[1] ||
        quantizedWeights.shape[2] != weights.shape[2] || quantizedWeights.shape[3] != weights.shape[3])
    {
        std::cerr << "Dimensions for quantized weights and weights don't match " << std::endl;
        throw std::runtime_error("Aborted Bias Correction");
    }

    // Error is epsilon * E[x]
    // Calculation of epsilon term
    const int nDims        = 4;
    cv::Mat quantWeightMat = cv::Mat(nDims, (int*) &quantizedWeights.shape[0], CV_32F, quantizedWeights.data);
    cv::Mat weightMat      = cv::Mat(nDims, (int*) &weights.shape[0], CV_32F, weights.data);

    // epsilon = W' - W (W' is quantized-dequantized weights & W is original weights)
    quantWeightMat -= weightMat;
    cv::Mat epsilon = TensorOperations::sumAlongThirdAndFourthAxis(quantWeightMat);

    // Calculating E[x] term
    int outputShape = weights.shape[1];
    // For depthwise separable layers, output shape of previous layer is same as output shape of current layer
    if (weights.shape[1] == 1)
        outputShape = weights.shape[0];

    vector<float> ex;
    if (activation == ActivationType::noActivation)
    {
        ex.assign(bnParams.beta, bnParams.beta + outputShape);
    }
    else
    {
        // Defaulting a & b (Inf) parameters to ReLU activation values
        int a = 0, b = INT32_MAX;
        if (activation == ActivationType::relu6)
            b = 6;

        for (int i = 0; i < outputShape; i++)
        {
            ex.push_back(calcExpectationPerChannel(a, b, bnParams.gamma[i], bnParams.beta[i]));
        }
    }

    cv::Mat exMat = cv::Mat(outputShape, 1, CV_32F, (float*) &ex[0]);

    cv::Mat errorMat;
    // For Depthwise separable layers
    if (epsilon.size[1] == 1)
        errorMat = epsilon.mul(exMat);
    else
        errorMat = epsilon * exMat;

    for (uint i = 0; i < errorMat.total(); i++)
    {
        bias.data[i] -= errorMat.at<float>(i);
    }
}


void BiasCorrection::storePreActivationOutput(TensorParam& outputActivation)
{
    uint outputLengthBatch =
        outputActivation.shape[0] * outputActivation.shape[1] * outputActivation.shape[2] * outputActivation.shape[3];

    uint outputLength = outputActivation.shape[1] * outputActivation.shape[2] * outputActivation.shape[3];

    std::vector<double> doubleBatchAct;
    doubleBatchAct.assign(outputActivation.data, outputActivation.data + outputLengthBatch);


    cv::Mat batchOutputActivationMat = cv::Mat(outputActivation.shape[0], outputLength, CV_64F, &doubleBatchAct[0]);


    for (int i = 1; i < outputActivation.shape[0]; i++)
    {
        batchOutputActivationMat.row(0) += batchOutputActivationMat.row(i);
    }

    // Add batch data to empty outputTensors
    if (outputTensors.empty())
    {
        outputTensors.assign(&doubleBatchAct[0], &doubleBatchAct[0] + outputLength);
        outputTensorShape[1] = outputActivation.shape[1];
        outputTensorShape[2] = outputActivation.shape[2];
        outputTensorShape[3] = outputActivation.shape[3];
    }
    else
    {
        cv::Mat outputTensorsMat = cv::Mat(1, outputLength, CV_64F, &outputTensors[0]);

        outputTensorsMat += batchOutputActivationMat.row(0);
    }

    // Adding one more output to the outputTensor vector
    outputTensorShape[0] += outputActivation.shape[0];
    ;
}

void BiasCorrection::storeQuantizedPreActivationOutput(TensorParam& outputActivation)
{
    uint outputLengthBatch =
        outputActivation.shape[0] * outputActivation.shape[1] * outputActivation.shape[2] * outputActivation.shape[3];

    uint outputLength = outputActivation.shape[1] * outputActivation.shape[2] * outputActivation.shape[3];

    std::vector<double> doubleBatchAct;
    doubleBatchAct.assign(outputActivation.data, outputActivation.data + outputLengthBatch);


    cv::Mat batchOutputActivationMat = cv::Mat(outputActivation.shape[0], outputLength, CV_64F, &doubleBatchAct[0]);


    for (int i = 1; i < outputActivation.shape[0]; i++)
    {
        batchOutputActivationMat.row(0) += batchOutputActivationMat.row(i);
    }

    if (quantizedOutputTensors.empty())
    {
        quantizedOutputTensors.assign(&doubleBatchAct[0], &doubleBatchAct[0] + outputLength);

        quantizedOutputTensorShape[1] = outputActivation.shape[1];
        quantizedOutputTensorShape[2] = outputActivation.shape[2];
        quantizedOutputTensorShape[3] = outputActivation.shape[3];
    }
    else
    {
        cv::Mat quantizedOutputTensorsMat = cv::Mat(1, outputLength, CV_64F, &quantizedOutputTensors[0]);
        quantizedOutputTensorsMat += batchOutputActivationMat.row(0);
    }

    // Adding one more output to the outputTensor vector
    quantizedOutputTensorShape[0] += outputActivation.shape[0];
    ;
}

void BiasCorrection::correctBias(TensorParam& bias)
{
    if (quantizedOutputTensorShape[0] != outputTensorShape[0])
    {
        std::cerr << "Number of quantized output do not match number of pre activation outputs " << std::endl;
        throw std::runtime_error("Aborted Bias Correction");
    }
    const int nDims          = 3;
    cv::Mat outputTensorsMat = cv::Mat(nDims, &outputTensorShape[1], CV_64F, &outputTensors[0]);


    cv::Mat quantizedOutputTensorsMat =
        cv::Mat(nDims, &quantizedOutputTensorShape[1], CV_64F, &quantizedOutputTensors[0]);

    quantizedOutputTensorsMat -= outputTensorsMat;

    cv::Mat summedErrorMat = TensorOperations::sumAlongSecondThirdAxis(quantizedOutputTensorsMat);

    const int divisor = outputTensorShape[0] * outputTensorShape[2] * outputTensorShape[3];

    // Find mean of outputTensor
    cv::Mat errorMat = summedErrorMat * (1.0 / divisor);

    // Converting bias float vector to double vector to match type while subtraction
    for (uint i = 0; i < errorMat.total(); i++)
    {
        bias.data[i] -= errorMat.at<double>(i);
    }
}

}