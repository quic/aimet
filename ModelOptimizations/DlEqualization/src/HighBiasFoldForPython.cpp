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

#include "DlEqualization/HighBiasFoldForPython.h"
#include "DlEqualization/HighBiasFold.h"

namespace AimetEqualization
{
void HighBiasFoldForPython::updateBias(AimetEqualization::LayerParamsForPython& prevLayerParamsForPython,
                                       AimetEqualization::LayerParamsForPython& currLayerParamsForPython,
                                       AimetEqualization::BNParamsHighBiasFoldForPython& prevLayerBNParams)
{
    BNParamsHighBiasFold bnParams;
    LayerParams prevLayerParams, currLayerParams;

    bnParams.beta  = &prevLayerBNParams.beta[0];
    bnParams.gamma = &prevLayerBNParams.gamma[0];

    prevLayerParams.activationIsRelu = prevLayerParamsForPython.activationIsRelu;
    prevLayerParams.bias             = &prevLayerParamsForPython.bias[0];

    currLayerParams.weightShape = currLayerParamsForPython.weightShape;
    currLayerParams.weight      = &currLayerParamsForPython.weight[0];
    currLayerParams.bias        = &currLayerParamsForPython.bias[0];

    AimetEqualization::HighBiasFold::updateBias(prevLayerParams, currLayerParams, bnParams);
}
}   // namespace AimetEqualization
