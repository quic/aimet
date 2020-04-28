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


#ifndef AIMET_BIAS_CORRECTION_H
#define AIMET_BIAS_CORRECTION_H

#include "def.h"
#include <vector>

namespace AimetEqualization
{
class BiasCorrection
{
private:
    std::vector<double> outputTensors;
    std::vector<int> outputTensorShape {0, 0, 0, 0};

    std::vector<double> quantizedOutputTensors;
    std::vector<int> quantizedOutputTensorShape {0, 0, 0, 0};

public:
    /**
     * brief: Stores pre activation output of a layer for each image. Batch size is N
     * @param outputActivation: Struct containing batch of pre activation output of a layer
     */
    void storePreActivationOutput(AimetEqualization::TensorParam& outputActivation);

    /**
     * brief: Stores quantized pre activation output of a layer. Batch size is N
     * @param outputActivation: Struct containing batch of pre activation output of a layer
     */
    void storeQuantizedPreActivationOutput(AimetEqualization::TensorParam& outputActivation);

    /**
     * brief: Corrects bias of layer and returns corrected bias
     * @param bias: struct of bias of layer
     * @return corrected bias tensor
     */
    void correctBias(AimetEqualization::TensorParam& bias);
};

class BnBasedBiasCorrection
{
private:
    /**
     * Calculates Normal PDF of x
     * @param x input
     */
    static float _phiX(const float x);

    /**
     * Calculates Normal CDF of x
     * @param x input
     */
    static float _normalCDF(const float x);

    /**
     * Calculates E[x] per channel
     * @param a Activation function parameter
     * @param b Activation function parameter
     * @param gamma BN parameter gamma per channel
     * @param beta BN parameter beta per channel
     */
    static float calcExpectationPerChannel(const int a, const int b, const float gamma, const float beta);

public:
    /**
     * Corrects bias caused due to quantization error in activations of layer using BN parameters
     * @param bias bias parameter of layer to be corrected
     * @param quantizedWeights quantized dequantized weights of layer
     * @param weights original weights of layer
     * @param bnParams BatchNorm parameters beta and gamma of prev layer
     * @param activation If activation is of type noActivation, relu, relu6
     */
    static void correctBias(AimetEqualization::TensorParam& bias, AimetEqualization::TensorParam& quantizedWeights,
                            AimetEqualization::TensorParam& weights, AimetEqualization::BnParamsBiasCorr& bnParams,
                            ActivationType activation);
};
}   // End of namespace AimetEqualization


#endif   // AIMET_BIAS_CORRECTION_H