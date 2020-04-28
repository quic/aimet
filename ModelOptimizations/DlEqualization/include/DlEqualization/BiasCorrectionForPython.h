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


#ifndef AIMET_BIAS_CORRECTION_FOR_PYTHON_H
#define AIMET_BIAS_CORRECTION_FOR_PYTHON_H

#include "DlEqualization/BiasCorrection.h"
#include "def.h"
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

namespace AimetEqualization
{
/**
 * @brief Layer output parameters
 */

typedef struct
{
    // The shape of the tensor
    std::vector<int> shape;

    // Flattened Tensor data
    std::vector<float> data;

} TensorParamForPython;

typedef struct
{
    // BN layer's bias
    py::array_t<float> beta;

    // BN layer's weight
    py::array_t<float> gamma;
} BnParamsBiasCorrForPython;


class BiasCorrectionForPython
{
    AimetEqualization::BiasCorrection biasCorrection;

public:
    /**
     * brief: Stores pre activation output of a layer. Batch size is 1.
     * @param outputActivation: Struct containing batch of pre activation output of a layer
     */
    void storePreActivationOutput(py::array_t<float> activationArr);

    /**
     * brief: Stores quantized pre activation output of a layer. Batch size is 1.
     * @param outputActivation: Struct containing batch of pre activation output of a layer
     */
    void storeQuantizedPreActivationOutput(py::array_t<float> activationArr);

    /**
     * brief: Corrects bias of layer and returns corrected bias
     * @param bias: struct of bias of layer
     * @return corrected bias tensor
     */
    void correctBias(TensorParamForPython& bias);
};

class BnBasedBiasCorrectionForPython
{
public:
    /**
     * Corrects bias of layer using BN parameters
     * @param biasPython
     * @param quantizedWeightsPython
     * @param weightsPython
     * @param bnParamsPython BatchNorm parameters beta and gamma of prev layer
     * @param activationPython If activation is of type noActivation, relu, relu6
     */

    void correctBias(TensorParamForPython& biasPython, py::array_t<float>& quantizedWeightsPython,
                     py::array_t<float>& weightsPython, BnParamsBiasCorrForPython& bnParamsPython,
                     ActivationType activationPython);
};

}   // namespace AimetEqualization

// End of namespace AimetEqualization


#endif   // AIMET_BIAS_CORRECTION_FOR_PYTHON_H
