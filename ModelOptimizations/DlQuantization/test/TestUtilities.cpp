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
#include <random>
#include "quantization_utils.hpp"

using namespace DlQuantization;

class TestUtilitiesCpu : public ::testing::Test
{
protected:

    std::vector<float> data1, data2, data3, data4;
    std::vector<uint32_t> shape1, shape2, shape3;

    void SetUp() {
        if(data1.size() == 0) {
            data1.resize(24);
            std::iota(std::begin(data1), std::end(data1), 0);
            shape1 = {2,3,2,2};
        }

        if(data2.size() == 0) {
            data2.resize(60);
            float t = -15;
            for(uint32_t i = 0; i < data2.size(); ++i) {
                data2[i] = t;
                t += 0.5;
            }
            shape2 = {1,4,5,3};
        }

        if(data3.size() == 0) {
            shape3 = {2,5,4,1};
            data3.resize(40);
            std::mt19937 eng;
            std::normal_distribution<float> dist;
            for(auto& d : data3) {
                d = dist(eng);
            }
            std::iota(std::begin(data1), std::end(data1), 0);
        }

        if (data4.size() == 0){
            float mean   = 2;
            float stddev = 2;
            std::normal_distribution<float> distribution(mean, stddev);
            std::mt19937 generator(1);

            unsigned int tensorCount = 6000;
            data4.resize(tensorCount);

            for (unsigned int i = 0; i < tensorCount; i++)
            {
                data4[i] = distribution(generator);
            }

        }

    }
};
//Slice testing
TEST_F(TestUtilitiesCpu, SANITY_SliceAndDice) {

    // Test data1, axis 1
    int32_t axis = 1;
    std::vector<uint32_t> outputShape;
    std::vector<uint32_t> expectedOutputShape = {2,1,2,2};
    std::vector<std::vector<float>> outputData;
    std::vector<std::vector<float>> expectedOutputData(3);
    expectedOutputData[0] = {0,  1,  2,  3, 12, 13, 14, 15 };
    expectedOutputData[1] = {4,  5,  6,  7, 16, 17, 18, 19 };
    expectedOutputData[2] = {8,  9, 10, 11, 20, 21, 22, 23 };

    slice(this->data1.data(), this->shape1, axis, outputData, outputShape);
    ASSERT_EQ(outputData.size(), expectedOutputData.size());
    ASSERT_EQ(outputData[0], expectedOutputData[0]);
    ASSERT_EQ(outputData[1], expectedOutputData[1]);
    ASSERT_EQ(outputData[2], expectedOutputData[2]);
}

TEST_F(TestUtilitiesCpu, SANITY_SliceAndDice2) {

    int32_t axis = 3;
    std::vector<uint32_t> outputShape;
    std::vector<uint32_t> expectedOutputShape= {2,3,2,1};
    std::vector<std::vector<float>> outputData;
    std::vector<std::vector<float>> expectedOutputData(2);
    expectedOutputData[0] = {0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22 };
    expectedOutputData[1] = {1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23 };

    slice(this->data1.data(), this->shape1, axis, outputData, outputShape);
    ASSERT_EQ(outputData.size(), expectedOutputData.size());
    ASSERT_EQ(outputData[0], expectedOutputData[0]);
    ASSERT_EQ(outputData[1], expectedOutputData[1]);

}

TEST_F(TestUtilitiesCpu, SANITY_SliceAndDice3) {

    int32_t axis = 1;
    std::vector<uint32_t> outputShape;
    std::vector<uint32_t> expectedOutputShape= {1,1,5,3};
    std::vector<std::vector<float>> outputData;
    std::vector<std::vector<float>> expectedOutputData(4);
    expectedOutputData[0] = {-15., -14.5, -14., -13.5, -13., -12.5, -12., -11.5, -11., -10.5, -10., -9.5, -9., -8.5, -8.};
    expectedOutputData[1] = {-7.5, -7. , -6.5, -6. , -5.5, -5. , -4.5, -4. , -3.5, -3. , -2.5, -2. , -1.5, -1. , -0.5};
    expectedOutputData[2] = {0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7.};
    expectedOutputData[3] = {7.5,  8. ,  8.5,  9. ,  9.5, 10. , 10.5, 11. , 11.5, 12. , 12.5, 13. , 13.5, 14. , 14.5};

    slice(this->data2.data(), this->shape2, axis, outputData, outputShape);
    ASSERT_EQ(outputData.size(), expectedOutputData.size());
    ASSERT_EQ(outputData[0], expectedOutputData[0]);
    ASSERT_EQ(outputData[1], expectedOutputData[1]);
    ASSERT_EQ(outputData[2], expectedOutputData[2]);
    ASSERT_EQ(outputData[3], expectedOutputData[3]);

}

TEST_F(TestUtilitiesCpu, SANITY_SliceAndDice4) {

    int32_t axis = -2;
    std::vector<uint32_t> outputShape;
    std::vector<uint32_t> expectedOutputShape= {1,4,1,3};
    std::vector<std::vector<float>> outputData;
    std::vector<std::vector<float>> expectedOutputData(5);
    expectedOutputData[0] = {-15. , -14.5, -14. ,  -7.5,  -7. ,  -6.5,   0. ,   0.5,   1. ,  7.5,   8. ,   8.5};
    expectedOutputData[1] = {-13.5, -13. , -12.5,  -6. ,  -5.5,  -5. ,   1.5,   2. ,   2.5,  9. ,   9.5,  10.};
    expectedOutputData[2] = {-12. , -11.5, -11. ,  -4.5,  -4. ,  -3.5,   3. ,   3.5,   4., 10.5,  11. ,  11.5};
    expectedOutputData[3] = {-10.5, -10. ,  -9.5,  -3. ,  -2.5,  -2. ,   4.5,   5. ,   5.5, 12. ,  12.5,  13.};
    expectedOutputData[4] = {-9. , -8.5, -8. , -1.5, -1. , -0.5,  6. ,  6.5,  7. , 13.5, 14., 14.5};

    slice(this->data2.data(), this->shape2, axis, outputData, outputShape);
    ASSERT_EQ(outputData.size(), expectedOutputData.size());
    ASSERT_EQ(outputData[0], expectedOutputData[0]);
    ASSERT_EQ(outputData[1], expectedOutputData[1]);
    ASSERT_EQ(outputData[2], expectedOutputData[2]);
    ASSERT_EQ(outputData[3], expectedOutputData[3]);
    ASSERT_EQ(outputData[4], expectedOutputData[4]);

}

TEST_F(TestUtilitiesCpu, SANITY_SliceAndDiceSingleDim) {

    int32_t axis = 3;
    std::vector<uint32_t> outputShape;
    std::vector<uint32_t> expectedOutputShape= {2,5,4,1};
    std::vector<std::vector<float>> outputData;

    // Slicing on an axis where the dimension == 1 means the output "slice" is really just the input
    slice(this->data3.data(), this->shape3, axis, outputData, outputShape);
    ASSERT_EQ(outputData.size(), 1);
    ASSERT_EQ(outputData[0].size(), this->data3.size());
    ASSERT_EQ(outputData[0], this->data3);
}

//Concat testing
TEST_F(TestUtilitiesCpu, SANITY_Concat) {

    int32_t axis = 1;
    std::vector<uint32_t> outputShape;
    std::vector<float> outputData(this->data1.size());
    std::vector<uint32_t> splitShape = {2,1,2,2};
    std::vector<std::vector<float>> inputData(3);
    inputData[0] = {0,  1,  2,  3, 12, 13, 14, 15 };
    inputData[1] = {4,  5,  6,  7, 16, 17, 18, 19 };
    inputData[2] = {8,  9, 10, 11, 20, 21, 22, 23 };

    concat(inputData, splitShape, axis, outputData.data(), outputShape);

    ASSERT_EQ(outputData.size(), this->data1.size());
    ASSERT_EQ(outputData, this->data1);
    ASSERT_EQ(outputShape, this->shape1);
}

TEST_F(TestUtilitiesCpu, SANITY_Concat2) {

    int32_t axis = 1;
    std::vector<uint32_t> outputShape;
    std::vector<float> outputData(this->data2.size());;
    std::vector<uint32_t> splitShape= {1,1,5,3};
    std::vector<std::vector<float>> inputData(4);
    inputData[0] = {-15., -14.5, -14., -13.5, -13., -12.5, -12., -11.5, -11., -10.5, -10., -9.5, -9., -8.5, -8.};
    inputData[1] = {-7.5, -7. , -6.5, -6. , -5.5, -5. , -4.5, -4. , -3.5, -3. , -2.5, -2. , -1.5, -1. , -0.5};
    inputData[2] = {0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7.};
    inputData[3] = {7.5,  8. ,  8.5,  9. ,  9.5, 10. , 10.5, 11. , 11.5, 12. , 12.5, 13. , 13.5, 14. , 14.5};

    concat(inputData, splitShape, axis, outputData.data(), outputShape);

    ASSERT_EQ(outputData.size(), this->data2.size());
    ASSERT_EQ(outputData, this->data2);
    ASSERT_EQ(outputShape, this->shape2);
}

TEST_F(TestUtilitiesCpu, SANITY_Concat3) {

    int32_t axis = 3;
    std::vector<uint32_t> splitShape;
    std::vector<std::vector<float>> splitData;
    slice(this->data1.data(), this->shape1, axis, splitData, splitShape);

    std::vector<uint32_t> outputShape;
    std::vector<float> outputData(this->data1.size());
    concat(splitData, splitShape, axis, outputData.data(), outputShape);

    ASSERT_EQ(outputData.size(), this->data1.size());
    ASSERT_EQ(outputData, this->data1);
    ASSERT_EQ(outputShape, this->shape1);
}

TEST_F(TestUtilitiesCpu, SANITY_Concat4) {

    int32_t axis = -2;
    std::vector<uint32_t> splitShape;
    std::vector<std::vector<float>> splitData;
    slice(this->data2.data(), this->shape2, axis, splitData, splitShape);
    std::vector<uint32_t> expectedOutputShape= {1,4,1,3};
    ASSERT_EQ(splitShape, expectedOutputShape);

    std::vector<uint32_t> outputShape;
    std::vector<float> outputData(this->data2.size());
    concat(splitData, splitShape, axis, outputData.data(), outputShape);

    ASSERT_EQ(outputData.size(), this->data2.size());
    ASSERT_EQ(outputData, this->data2);
    ASSERT_EQ(outputShape, this->shape2);

}