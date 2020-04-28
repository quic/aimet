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


#ifndef AIMET_BATCH_NORM_FOLD_H
#define AIMET_BATCH_NORM_FOLD_H

#include <vector>

namespace AimetEqualization
{
/**
 * @brief Batch norm layer parameters
 */

typedef struct
{
    // BN layer's bias
    float* beta;

    // BN layer's weight
    float* gamma;

    // BN layer's running_mean
    float* runningMean;

    // BN layer's running_var
    float* runningVar;

} BNParams;

/**
 * @brief Layer weight parameters
 */

typedef struct
{
    // The shape of the tensor
    std::vector<int> shape;

    // Pointer to the flattened tensor's data
    float* data;

} TensorParams;

class BatchNormFold
{
public:
    /**
     * @brief Updates a given layers weight parameters to fold a batch norm layer before or after it
     * @param bnParams Struct containing BN layer parameters
     * @param weightTensor Struct containing weight of layer to the fold the BN into
     * @param biasTensor Struct containing bias of layer to the fold the BN into
     * @param isBiasTensorValid True if the layer did have a bias tensor
     * @param foldPrevLayer True if BN layer follows the conv/linear layer, False otherwise
     * @return Updated bias vector for the layer
     */
    static std::vector<float> fold(BNParams& bnParams, TensorParams& weightTensor, TensorParams& biasTensor,
                                   bool isBiasTensorValid, bool foldPrevLayer = true);
};

}   // End of namespace AimetEqualization


#endif   // AIMET_BATCH_NORM_FOLD_H
