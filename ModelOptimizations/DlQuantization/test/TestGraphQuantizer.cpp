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

#include <gtest/gtest.h>
#include <random>

#include "DlQuantization/GraphQuantizer.h"
#include "test_quantization_lib.hpp"

using namespace DlQuantization;

class TestGraphQuantizer : public ::testing::Test
{
protected:
    std::vector<float> data1, data2;
    std::vector<float> symData1, symData2;
    std::vector<std::string> tensorNames {"tensor1", "tensor2"};
    const std::map<std::string, int> bws = {{"tensor1", 8}, {"tensor2", 8}};
    std::unique_ptr<GraphQuantizer> enhancedGraphQuant;
    std::unique_ptr<GraphQuantizer> tfGraphQuant;

    void SetUp()
    {
        if (data1.size() == 0)
        {
            data1.resize(10);
            std::iota(std::begin(data1), std::end(data1), 0);
        }

        if (data2.size() == 0)
        {
            data2.resize(15);
            std::iota(std::begin(data2), std::end(data2), 0);
        }


        if (symData1.size() == 0)
        {
            symData1.resize(60);
            float t = -15;
            for (uint32_t i = 0; i < symData1.size(); ++i)
            {
                symData1[i] = t;
                t += 0.5;
            }
        }

        if (symData2.size() == 0)
        {
            float mean   = 2;
            float stddev = 2;
            std::normal_distribution<float> distribution(mean, stddev);
            std::mt19937 generator(1);

            unsigned int tensorCount = 6000;
            symData2.resize(tensorCount);

            for (unsigned int i = 0; i < tensorCount; i++)
            {
                symData2[i] = distribution(generator);
            }
        }

        enhancedGraphQuant.reset(new GraphQuantizer(tensorNames, ComputationMode::COMP_MODE_CPU,
                                                    QuantizationMode::QUANTIZATION_TF_ENHANCED));
        tfGraphQuant.reset(
            new GraphQuantizer(tensorNames, ComputationMode::COMP_MODE_CPU, QuantizationMode::QUANTIZATION_TF));
    }

    static bool compareEncodingMap(const std::map<std::string, TfEncoding>& encodings,
                                   const std::map<std::string, TfEncoding>& expectedEncodings)
    {
        return std::equal(
            encodings.begin(), encodings.end(), expectedEncodings.begin(),
            [&](const std::pair<std::string, TfEncoding>& enc1, const std::pair<std::string, TfEncoding>& enc2)
            {
                bool isEqual = compareEncodings(enc1.second, enc2.second);
                if (!isEqual)
                {
                    std::cout << std::endl << "Got:   ";
                    printEncoding(enc1.second);
                    std::cout << std::endl << "Expected:   ";
                    printEncoding(enc2.second);
                }
                return enc1.first == enc2.first && isEqual;
            });
    }
};

TEST_F(TestGraphQuantizer, SanityTestGetEncodingsTfEnhancedCpu)
{
    enhancedGraphQuant->setUnsignedSymmetric(false);

    enhancedGraphQuant->updateStats(tensorNames[0], data1.data(), data1.size());
    EXPECT_TRUE(enhancedGraphQuant->getTensorQuantizer(tensorNames[0])->hasValidStats());

    enhancedGraphQuant->updateStats(tensorNames[1], data2.data(), data2.size());
    EXPECT_TRUE(enhancedGraphQuant->getTensorQuantizer(tensorNames[1])->hasValidStats());


    std::map<std::string, TfEncoding> encodings;
    // get Encodings
    enhancedGraphQuant->getEncodings(bws, encodings, false);
    std::map<std::string, TfEncoding> expectedEncodings = {{"tensor1", getTfEncoding(0, 8.98242, 8)},
                                                           {"tensor2", getTfEncoding(0, 14, 8)}};

    for (const auto& tensorName: tensorNames)
    {
        ASSERT_NO_FATAL_FAILURE(compareEncodings(encodings[tensorName], expectedEncodings[tensorName]));
    }
}

TEST_F(TestGraphQuantizer, SanityTestGetEncodingsTfCpu)
{
    tfGraphQuant->setUnsignedSymmetric(false);

    tfGraphQuant->updateStats(tensorNames[0], data1.data(), data1.size());
    EXPECT_TRUE(tfGraphQuant->getTensorQuantizer(tensorNames[0])->hasValidStats());

    tfGraphQuant->updateStats(tensorNames[1], data2.data(), data2.size());
    EXPECT_TRUE(tfGraphQuant->getTensorQuantizer(tensorNames[1])->hasValidStats());


    std::map<std::string, TfEncoding> encodings;
    // get Encodings
    tfGraphQuant->getEncodings(bws, encodings, false);
    std::map<std::string, TfEncoding> expectedEncodings = {{"tensor1", getTfEncoding(0, 9, 8)},
                                                           {"tensor2", getTfEncoding(0, 14, 8)}};

    EXPECT_TRUE(compareEncodingMap(encodings, expectedEncodings));
}

TEST_F(TestGraphQuantizer, SanityTestGetEncodingsTfCpu_Symmetric)
{
    tfGraphQuant->setStrictSymmetric(true);
    tfGraphQuant->setUnsignedSymmetric(false);
    tfGraphQuant->updateStats(tensorNames[0], symData1.data(), symData1.size());
    EXPECT_TRUE(tfGraphQuant->getTensorQuantizer(tensorNames[0])->hasValidStats());

    tfGraphQuant->updateStats(tensorNames[1], symData2.data(), symData2.size());
    EXPECT_TRUE(tfGraphQuant->getTensorQuantizer(tensorNames[1])->hasValidStats());


    std::map<std::string, TfEncoding> encodings;
    // get Encodings
    tfGraphQuant->getEncodings(bws, encodings, true);
    std::map<std::string, TfEncoding> expectedEncodings = {{"tensor1", getTfSymmetricEncoding(15, 8)},
                                                           {"tensor2", getTfSymmetricEncoding(8.89245, 8)}};

    for (const auto& tensorName: tensorNames)
    {
        ASSERT_NO_FATAL_FAILURE(compareEncodings(encodings[tensorName], expectedEncodings[tensorName]));
    }
}

TEST_F(TestGraphQuantizer, SanityTestGetEncodingsFailsIfNoUpdateStats)
{
    auto tensorName = tensorNames[0];

    std::map<std::string, TfEncoding> encodings;
    // verify get Encodings fails
    EXPECT_THROW(tfGraphQuant->getEncodings(bws, encodings, false), std::runtime_error);
}