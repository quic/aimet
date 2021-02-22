//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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

#include <random>
#include <DlQuantization/TensorQuantizer.h>
#include <gtest/gtest.h>

#include "test_quantization_lib.hpp"

using namespace DlQuantization;

class TestTensorQuantizer : public ::testing::Test
{
};

TEST(TestTensorQuantizer, SanityTestCpu)
{
    TensorQuantizer tensorQuantizer(QuantizationMode::QUANTIZATION_TF_ENHANCED, ROUND_NEAREST);

    float mean   = 2;
    float stddev = 2;
    std::normal_distribution<float> distribution(mean, stddev);
    std::mt19937 generator(1);

    unsigned int tensorCount = 6000;
    std::vector<float> statsTensor(tensorCount);

    for (unsigned int i = 0; i < tensorCount; i++)
    {
        statsTensor[i] = distribution(generator);
    }

    tensorQuantizer.updateStats(statsTensor.data(), statsTensor.size(), false);
    EXPECT_FALSE(tensorQuantizer.isEncodingValid);
    TfEncoding encoding = tensorQuantizer.computeEncoding(8, false, false, false);
    EXPECT_TRUE(tensorQuantizer.isEncodingValid);

    std::vector<float> inputTensor(tensorCount, 5);
    std::vector<float> quantizedTensor(tensorCount);

    tensorQuantizer.quantizeDequantize(inputTensor.data(), inputTensor.size(), quantizedTensor.data(),
                                       encoding.min, encoding.max, 8, false);

    std::cout << "Encoding min=" << encoding.min << ", max=" << encoding.max
              << std::endl;
    EXPECT_NEAR(encoding.min, -6.52711, 0.001);
    EXPECT_NEAR(encoding.max, 8.88412, 0.001);

    std::cout << "input-data=" << inputTensor.data()[0] << ", quantized-data=" << quantizedTensor.data()[0]
              << std::endl;

    EXPECT_NE(inputTensor.data()[0], quantizedTensor.data()[0]);
    EXPECT_NEAR(quantizedTensor.data()[0], 5.0162, 0.001);
}

#ifdef GPU_QUANTIZATION_ENABLED

TEST(TestTensorQuantizer, SanityTestGpu)
{
    TensorQuantizer tensorQuantizer(QuantizationMode::QUANTIZATION_TF_ENHANCED, ROUND_NEAREST);

    float mean   = 2;
    float stddev = 2;
    std::normal_distribution<float> distribution(mean, stddev);
    std::mt19937 generator(1);

    int tensorCount = 6000;
    std::vector<float> statsTensor(tensorCount);

    for (unsigned int i = 0; i < tensorCount; i++)
    {
        statsTensor[i] = distribution(generator);
    }
    Blob<GpuDevice<float>> statsTensorBlob(statsTensor.data(), tensorCount);

    tensorQuantizer.updateStats(statsTensorBlob.getDataPtrOnDevice(), statsTensor.size(), true);
    EXPECT_FALSE(tensorQuantizer.isEncodingValid);
    TfEncoding encoding = tensorQuantizer.computeEncoding(8, false, false, false);
    EXPECT_TRUE(tensorQuantizer.isEncodingValid);

    std::vector<float> inputTensor(tensorCount, 5);
    std::vector<float> quantizedTensor(tensorCount);
    tensorQuantizer.quantizeDequantize(inputTensor.data(), inputTensor.size(), quantizedTensor.data(),
                                       encoding.min, encoding.max, 8, false);

    std::cout << "Encoding min=" << encoding.min << ", max=" <<
    encoding.max << std::endl;
    EXPECT_NEAR(encoding.min, -6.52711, 0.001);
    EXPECT_NEAR(encoding.max, 8.88412, 0.001);

    std::cout << "input-data=" << inputTensor.data()[0] << ", quantized-data=" <<
    quantizedTensor.data()[0]<< std::endl;
    EXPECT_NE(inputTensor.data()[0], quantizedTensor.data()[0]);
    EXPECT_NEAR(quantizedTensor.data()[0], 5.0162, 0.001);
}

#endif