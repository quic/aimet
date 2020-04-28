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

#ifndef AIMET_CROSS_LAYER_SCALING_FOR_PYTHON_H
#define AIMET_CROSS_LAYER_SCALING_FOR_PYTHON_H

#include "DlEqualization/CrossLayerScaling.h"
#include <vector>

namespace AimetEqualization
{
/**
 * @brief EqualizationParams type comprising weightTensor shape, weights and bias
 */

typedef struct
{
    // The shape of weight tensor (output, input, height, width)
    std::vector<int> weightShape;

    // Layer's weight tensor
    std::vector<float> weight;

    // Layer's bias tensor
    std::vector<float> bias;

    // If layer bias is None
    bool isBiasNone;

} EqualizationParamsForPython;

class CrossLayerScalingForPython
{
public:
    /**
     * @brief Scales weights and bias for a depth wise separable layer
     * @param prevLayer Struct of layer weight, shape and axis shared by the previous layer with the current layer
     * @param currLayer Struct of layer weight, shape and axis shared by the current layer with the previous layer
     * @param nextLayer If curr layer is depthwise-seperable then specify:
     *                   Struct of layer weight, shape and axis shared by the current layer with the previous layer
     * @return Rescaling parameters struct. Updates the weights and biases in place
     */
    static AimetEqualization::CrossLayerScaling::RescalingParamsVectors
    scaleDepthWiseSeparableLayer(EqualizationParamsForPython& prevLayer, EqualizationParamsForPython& currLayer,
                                 EqualizationParamsForPython& nextLayer);

    /**
     * @brief Scales weights and bias for consecutive layers
     * @param prevLayer Struct of layer weight, shape and axis shared by the previous layer with the current layer
     * @param currLayer Struct of layer weight, shape and axis shared by the current layer with the previous layer
     * @return Rescaling parameters vector. Updates the weights and biases in place
     */
    static std::vector<float> scaleLayerParams(EqualizationParamsForPython& prevLayer,
                                               EqualizationParamsForPython& currLayer);
};


}   // namespace AimetEqualization

#endif   // AIMET_CROSS_LAYER_SCALING_FOR_PYTHON_H
