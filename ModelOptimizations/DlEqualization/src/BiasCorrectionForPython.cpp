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

#include <pybind11/numpy.h>

#include "DlEqualization/BiasCorrection.h"
#include "DlEqualization/BiasCorrectionForPython.h"
#include "DlEqualization/def.h"

#include <iostream>
namespace py = pybind11;

namespace AimetEqualization
{
void BiasCorrectionForPython::storePreActivationOutput(py::array_t<float> activationArr)
{
    auto npArr   = activationArr.mutable_unchecked<4>();
    auto dataPtr = (float*) npArr.mutable_data(0, 0, 0, 0);

    TensorParam outputActivation = {{npArr.shape(0), npArr.shape(1), npArr.shape(2), npArr.shape(3)}, dataPtr};
    biasCorrection.storePreActivationOutput(outputActivation);
}

void BiasCorrectionForPython::storeQuantizedPreActivationOutput(py::array_t<float> activationArr)
{
    auto npArr   = activationArr.mutable_unchecked<4>();
    auto dataPtr = (float*) npArr.mutable_data(0, 0, 0, 0);

    TensorParam outputActivation = {{npArr.shape(0), npArr.shape(1), npArr.shape(2), npArr.shape(3)}, dataPtr};
    biasCorrection.storeQuantizedPreActivationOutput(outputActivation);
}

void BiasCorrectionForPython::correctBias(AimetEqualization::TensorParamForPython& biasPython)
{
    TensorParam bias;
    bias = {biasPython.shape, &biasPython.data[0]};

    biasCorrection.correctBias(bias);
}

void BnBasedBiasCorrectionForPython::correctBias(AimetEqualization::TensorParamForPython& biasPython,
                                                 py::array_t<float>& quantizedWeightsPython,
                                                 py::array_t<float>& weightsPython,
                                                 BnParamsBiasCorrForPython& bnParamsPython,
                                                 ActivationType activationPython)
{
    TensorParam bias, weights, quantizedWeights;
    bias = {biasPython.shape, &biasPython.data[0]};


    auto npArr   = weightsPython.mutable_unchecked<4>();
    auto dataPtr = (float*) npArr.mutable_data(0, 0, 0, 0);

    //    std::cout<<"data pointer "<<dataPtr[0]<<std::endl;

    weights = {{npArr.shape(0), npArr.shape(1), npArr.shape(2), npArr.shape(3)}, dataPtr};

    auto npArrQuantized   = quantizedWeightsPython.mutable_unchecked<4>();
    auto dataPtrQuantized = (float*) npArrQuantized.mutable_data(0, 0, 0, 0);

    quantizedWeights = {
        {npArrQuantized.shape(0), npArrQuantized.shape(1), npArrQuantized.shape(2), npArrQuantized.shape(3)},
        dataPtrQuantized};
    //    std::cout<<"data pointer quant"<<quantizedWeights.data[0]<<std::endl;


    BnParamsBiasCorr bnParams;
    auto npArrBeta  = bnParamsPython.beta.mutable_unchecked<1>();
    bnParams.beta   = (float*) npArrBeta.mutable_data(0);
    auto npArrGamma = bnParamsPython.gamma.mutable_unchecked<1>();
    bnParams.gamma  = (float*) npArrGamma.mutable_data(0);
    BnBasedBiasCorrection::correctBias(bias, quantizedWeights, weights, bnParams, activationPython);
}
}