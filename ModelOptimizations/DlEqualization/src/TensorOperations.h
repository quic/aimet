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

#ifndef AIMET_TENSOR_OPERATIONS_H
#define AIMET_TENSOR_OPERATIONS_H

#include <iostream>
#include <opencv2/core/core.hpp>


namespace AimetEqualization
{
static constexpr int const& FLOAT_32_TYPE                   = CV_32F;
static constexpr int const& SUB_MAT_DIMENSION               = 3;
static constexpr int const& NUM_DIMENSIONS_IN_WEIGHT_TENSOR = 4;

typedef enum axisType
{
    AXIS_0 = 0,
    AXIS_1 = 1
} axisType;

/**weight
 * Four dimensions of Weight Tensor
 */
typedef enum dimension
{
    DIM_0 = 0,
    DIM_1 = 1,
    DIM_2 = 2,
    DIM_3 = 3

} dimension;

class TensorOperations
{
public:
    /**
     * Generic api to extract per channel data (without reshape on 4D matrix),
     * along a given axis and channel index.
     * @param inputMat in MxNxHxW format , where M is output channels and N is input channels
     * @param channelIndex channel index for which the data blob needs to be extracted
     * @param axis AXIS_0 extracts [NxWxH] data blob and
     *             AXIS_1 extracts [MxWxH] data blob,
     *             in a NxMxHxW inputMat.
     * @return Extracted sub matrix as open CV Mat type
     */
    static cv::Mat getDataPerChannelIn4dMat(const cv::Mat& inputMat, const int& channelIndex, axisType axis);


    /**
     * @brief Sums weightTensor along 3rd and 4th axis and returns a 2D matrix
     * @param weightTensor
     * @return 2D matrix
     */
    static cv::Mat sumAlongThirdAndFourthAxis(cv::Mat weightTensor);
    /**
     * Method to swaps data along first two axis of given input 4d mat
     * @param input4dMat in MxNxHxW format , where M is output channels and N is input channels
     * @return opencv 4d Mat with swapped data
     */

    static cv::Mat swapFirstTwoAxisIn4dMat(const cv::Mat& input4dMat);

    /**
     * Internal method to compute range along first axis of 4d input mat.
     * @param weightTensor Input tensor in correct format (MxNxHxW) / NxMxHxW
     * @return 1D Mat with range values for given set of channels
     */
    static cv::Mat computeRangeAlongFirstAxis(const cv::Mat& weightTensor);

    /**
     * Sums Tensor along 2nd and 3rd axis and returns a 1D matrix
     * @param inputTensor 2D matrix
     * @return 1D reduced matrix
     */
    static cv::Mat sumAlongSecondThirdAxis(cv::Mat inputTensor);
};

}   // namespace AimetEqualization


#endif   // AIMET_TENSOR_OPERATIONS_H
