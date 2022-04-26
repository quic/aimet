//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

#include <gtest/gtest.h>

#include "tensorflow/core/framework/tensor.h"

using namespace tensorflow;


class TestTfTensorOps : public ::testing::Test
{
};


TEST(TestTfTensorOps, TensorPerChannelMinMax)
{
    Tensor input(DT_FLOAT, TensorShape({2, 3, 4, 5}));
    std::vector<float> inputData(2 * 3 * 4 * 5, 5);
    float mean   = 2;
    float stddev = 2;
    std::normal_distribution<float> distribution(mean, stddev);
    std::mt19937 generator(10);
    for (int i = 0; i < inputData.size(); i++)
    {
        inputData[i] = distribution(generator);
    }

    std::copy_n(inputData.data(), inputData.size(), input.flat<float>().data());

    auto flat2dTensor = input.flat_inner_dims<float, 2>();
    Tensor tempScalar(DT_FLOAT, TensorShape({1}));;
    auto tempScalarTensorMap = tempScalar.scalar<float>();
    std::vector<float> minVector(5);
    std::vector<float> maxVector(5);

    for (int channel_idx = 0; channel_idx < flat2dTensor.dimension(1); channel_idx++)
    {
        tempScalarTensorMap = flat2dTensor.chip<1>(channel_idx).minimum();
        float min = tempScalarTensorMap();
        minVector[channel_idx] = min;

        tempScalarTensorMap = flat2dTensor.chip<1>(channel_idx).maximum();
        float max = tempScalarTensorMap();
        maxVector[channel_idx] = max;

        std::cout << "Channel: " << channel_idx << ", ";
        std::cout << "Min: " << min << ", ";
        std::cout << "Max: " << max << "\n";
    }

    Tensor minTensor(DT_FLOAT, TensorShape({5}));
    Tensor maxTensor(DT_FLOAT, TensorShape({5}));
    std::copy_n(minVector.data(), minVector.size(), minTensor.flat<float>().data());
    std::copy_n(maxVector.data(), maxVector.size(), maxTensor.flat<float>().data());

    std::cout << minTensor.dims() << '\n';

    std::cout << "Num dimensions: " << flat2dTensor.dimensions() << "\n";
    EXPECT_EQ(2, flat2dTensor.dimensions().size());
    EXPECT_EQ(24, flat2dTensor.dimensions()[0]);
    EXPECT_EQ(5, flat2dTensor.dimensions()[1]);
}

TEST(TestTfTensorOps, TensorPerTensorMinMax)
{
    Tensor input(DT_FLOAT, TensorShape({2, 3, 4, 5}));

    std::vector<float> inputData(2 * 3 * 4 * 5, 5);
    float mean   = 2;
    float stddev = 2;
    std::normal_distribution<float> distribution(mean, stddev);
    std::mt19937 generator(10);
    for (int i = 0; i < inputData.size(); i++)
    {
        inputData[i] = distribution(generator);
    }
    std::copy_n(inputData.data(), inputData.size(), input.flat<float>().data());

    Tensor someScalar(DT_FLOAT, TensorShape({}));;
    auto scalar = someScalar.scalar<float>();
    scalar = input.flat<float>().minimum();
    float min = scalar();
    scalar = input.flat<float>().maximum();
    float max = scalar();

    std::cout << "Min: " << min << ", ";
    std::cout << "Max: " << max << "\n";

    EXPECT_NEAR(-2.60564, min, 0.0001);
    EXPECT_NEAR(7.35337, max, 0.0001);
}
