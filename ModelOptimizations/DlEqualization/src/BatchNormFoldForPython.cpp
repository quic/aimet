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
#include "DlEqualization/BatchNormFold.h"
#include <iostream>

namespace AimetEqualization
{
std::vector<float> BatchNormFoldForPython::fold(BNParamsForPython& bnParamsForPython,
                                                TensorParamsForPython& weightTensorForPython,
                                                TensorParamsForPython& biasTensorForPython, bool isBiasTensorValid,
                                                bool foldPrevLayer)
{
    BNParams bnParams;

    bnParams.beta        = &bnParamsForPython.beta[0];
    bnParams.gamma       = &bnParamsForPython.gamma[0];
    bnParams.runningMean = &bnParamsForPython.runningMean[0];
    bnParams.runningVar  = &bnParamsForPython.runningVar[0];

    TensorParams weightTensor = {weightTensorForPython.shape, &weightTensorForPython.data[0]};
    TensorParams biasTensor   = {biasTensorForPython.shape, &biasTensorForPython.data[0]};


    return BatchNormFold::fold(bnParams, weightTensor, biasTensor, isBiasTensorValid, foldPrevLayer);
}
}   // namespace AimetEqualization
