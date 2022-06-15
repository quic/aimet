//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2020 - 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

    tensorQuantizer.setStrictSymmetric(false);
    tensorQuantizer.setUnsignedSymmetric(false);
    tensorQuantizer.updateStats(statsTensor.data(), statsTensor.size(), false);
    EXPECT_FALSE(tensorQuantizer.isEncodingValid);
    TfEncoding encoding = tensorQuantizer.computeEncoding(8, false);
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


TEST(TestTensorQuantizer, SanityTestComputeEncodingFromDataAsymmetricTFEnhanced)
{
    TensorQuantizer tensorQuantizer(QuantizationMode::QUANTIZATION_TF_ENHANCED, ROUND_NEAREST);

    float mean   = 2;
    float stddev = 2;
    std::normal_distribution<float> distribution(mean, stddev);
    std::mt19937 generator(1);

    unsigned int tensorCount = 6000;
    std::vector<float> paramTensor(tensorCount);

    for (unsigned int i = 0; i < tensorCount; i++)
    {
        paramTensor[i] = distribution(generator);
    }

    TfEncoding encoding{};
    tensorQuantizer.computeEncodingFromData(8, paramTensor.data(),
                                            tensorCount, encoding, ComputationMode::COMP_MODE_CPU, false, false, false);

    std::cout << "Encoding min=" << encoding.min << ", max=" << encoding.max
              << std::endl;
    EXPECT_NEAR(encoding.min, -6.527, 0.001);
    EXPECT_NEAR(encoding.max, 8.884, 0.001);
}

TEST(TestTensorQuantizer, SanityTestComputeEncodingFromDataSymmetricTF)
{
    TensorQuantizer tensorQuantizer(QuantizationMode::QUANTIZATION_TF, ROUND_NEAREST);

    float mean   = 2;
    float stddev = 2;
    std::normal_distribution<float> distribution(mean, stddev);
    std::mt19937 generator(100);

    unsigned int tensorCount = 6000;
    std::vector<float> paramTensor(tensorCount);

    for (int i = 0; i < tensorCount; i++)
    {
        paramTensor[i] = distribution(generator);
    }

    TfEncoding encoding{};
    tensorQuantizer.computeEncodingFromData(8, paramTensor.data(),
                                            tensorCount, encoding, ComputationMode::COMP_MODE_CPU,
                                            true, false, true);

    std::cout << "Encoding min=" << encoding.min << ", max=" << encoding.max
              << std::endl;

    float expected_max = std::max(std::abs(*std::min_element(paramTensor.begin(), paramTensor.end())),
                                  std::abs(*std::max_element(paramTensor.begin(), paramTensor.end())));

    // Min and Max will get adjusted slightly to represent an exact zero with one of the quantized values
    // Adjustment is expected to be less than half a delta worth
    EXPECT_NEAR(encoding.max, expected_max, encoding.delta / 2 + 1e-4);
    EXPECT_EQ(encoding.max, -encoding.min);

    // Check that the center value is absolute 0
    EXPECT_NEAR(encoding.min + encoding.delta * (-encoding.offset), 0, 1e-7);

    EXPECT_FLOAT_EQ(encoding.delta, (encoding.max - encoding.min) / 254);

    // Check that offset is -127 - another check for strict symmetric encodings
    EXPECT_NEAR(encoding.offset, -127, 0);
    EXPECT_EQ(encoding.bw, 8);
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

    tensorQuantizer.setStrictSymmetric(false);
    tensorQuantizer.setUnsignedSymmetric(false);
    tensorQuantizer.updateStats(statsTensorBlob.getDataPtrOnDevice(), statsTensor.size(), true);
    EXPECT_FALSE(tensorQuantizer.isEncodingValid);
    TfEncoding encoding = tensorQuantizer.computeEncoding(8, false);
    EXPECT_TRUE(tensorQuantizer.isEncodingValid);

    std::vector<float> inputTensor(tensorCount, 5);
    Blob<GpuDevice<float>> inputTensorBlob(inputTensor.data(), tensorCount);

    std::vector<float> quantizedTensor(tensorCount, 0);
    Blob<GpuDevice<float>> quantTensorBlob(quantizedTensor.data(), tensorCount);

    tensorQuantizer.quantizeDequantize(inputTensorBlob.getDataPtrOnDevice(), inputTensor.size(),
                                       quantTensorBlob.getDataPtrOnDevice(),
                                       encoding.min, encoding.max, 8, true);

    std::cout << "Encoding min=" << encoding.min << ", max=" <<
    encoding.max << std::endl;
    EXPECT_NEAR(encoding.min, -6.52711, 0.001);
    EXPECT_NEAR(encoding.max, 8.88412, 0.001);

    std::cout << "input-data=" << inputTensorBlob.getDataPtrOnCpu()[0] << ", quantized-data=" <<
    quantTensorBlob.getDataPtrOnCpu()[0]<< std::endl;
    EXPECT_NE(inputTensorBlob.getDataPtrOnCpu()[0], quantTensorBlob.getDataPtrOnCpu()[0]);
    EXPECT_NEAR(quantTensorBlob.getDataPtrOnCpu()[0], 5.0162, 0.001);
}

#endif