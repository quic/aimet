//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2016-2017, Qualcomm Innovation Center, Inc. All rights reserved.
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


#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "DlQuantization/QuantizerFactory.hpp"
#include "test_quantization_lib.hpp"

using namespace std;
using namespace DlQuantization;

template <typename TypeParam>
class TestQuantizationLibTfEnhancedCpuGpu : public ::testing::Test
{
protected:
    void PrintEncoding(TfEncoding encoding)
    {
        cout << "Encoding: min: " << encoding.min << ", max: " << encoding.max << ", delta: " << encoding.delta
             << ", offset: " << encoding.offset << ", bw: " << encoding.bw << endl;
    }
};

// Test on CPU and GPU with float and double
TYPED_TEST_CASE(TestQuantizationLibTfEnhancedCpuGpu, TestDataTypesAndDevices);

// Test quantization of activations to 8-bit.
// Gather stats for one blob and two iterations: Gaussian distribution.
// Test that SQNR of gaussian distribution is reasonably high.
// In this test case we test the following methods of the IQuantizationAlgorithm
// interface: UpdateStatsModeSpecific(), StatsToFxpFormat().
TYPED_TEST(TestQuantizationLibTfEnhancedCpuGpu, SANITY_Gaussian)
{
    if (!CheckRunTest<TypeParam>())
        return;
    typedef typename TypeParam::dataType DataType;
    // Create quantizer object
    vector<string> layer_names;
    string layer_name("conv0");
    layer_names.push_back(layer_name);
    vector<int> bw_activations;
    int bw = 8;
    bw_activations.push_back(bw);
    unique_ptr<IQuantizer<DataType>> iq = unique_ptr<IQuantizer<DataType>>(
        GetQuantizerInstance<DataType>(layer_names, TypeParam::modeCpuGpu, bw_activations, QUANTIZATION_TF_ENHANCED));
    // Prepare data for gathering statistics: Gaussian distribution.
    const int cnt_acts = 1000;
    vector<DataType> acts_iter0(cnt_acts);
    vector<DataType> acts_iter1(cnt_acts);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    double mean   = 2;
    double stddev = 2;
    normal_distribution<double> distribution(mean, stddev);
    for (int i = 0; i < cnt_acts; ++i)
    {
        acts_iter0[i] = distribution(generator);
        acts_iter1[i] = distribution(generator);
    }
    // Pass data into library to gather statistics.
    vector<size_t> count;
    count.push_back(cnt_acts);
    vector<const DataType*> acts_vector;
    // Iteration Nr. 0.
    Blob<TypeParam> blob_acts_iter0(acts_iter0.data(), cnt_acts);
    acts_vector.push_back(blob_acts_iter0.getDataPtrOnDevice());
    iq->UpdateStats(layer_name, LAYER_INPUT, acts_vector, count);
    // Iteration Nr. 1.
    Blob<TypeParam> blob_acts_iter1(acts_iter1.data(), cnt_acts);
    acts_vector.clear();
    acts_vector.push_back(blob_acts_iter1.getDataPtrOnDevice());
    iq->UpdateStats(layer_name, LAYER_INPUT, acts_vector, count);
    // Get Encoding from library.
    map<string, TfEncodingLayer> encodings;
    map<string, int> bws;
    bws.insert(std::make_pair(layer_name, bw));
    iq->GetEncoding(bws, encodings);
    EXPECT_NE(0, encodings.count(layer_name));
    EXPECT_EQ(1, encodings[layer_name].in.size());
    TfEncoding encoding = encodings[layer_name].in[0];
    // We know we have a normal distribution. We expect the encoding to cover
    // at least 2 standard deviations, and at most 6.
    EXPECT_GT(encoding.min, mean - 6 * stddev);
    EXPECT_LT(encoding.min, mean - 2 * stddev);
    EXPECT_GT(encoding.max, mean + 2 * stddev);
    EXPECT_LT(encoding.max, mean + 6 * stddev);
    // Compute SQNR.
    // We use the first activation blob from before.
    vector<DataType> reference = acts_iter0;
    Blob<TypeParam> data_blob(acts_iter0.data(), cnt_acts);
    vector<DataType*> acts {data_blob.getDataPtrOnDevice()};
    vector<TfEncoding> encoding_acts;
    // Quantize the activation blob.
    iq->QuantizeDequantizeActs(layer_name, LAYER_INPUT, bw, acts, count, acts, encoding_acts);
    // Compare the quantized blob to the original blob.
    DataType signal          = 0;
    DataType noise           = 0;
    DataType* data_quantized = data_blob.getDataPtrOnCpu();
    for (int i = 0; i < cnt_acts; ++i)
    {
        signal += pow(reference[i], 2);
        noise += pow(data_quantized[i] - reference[i], 2.0);
    }
    DataType sqnr = 10 * log10(signal / noise);
    // Check that our SQNR is at least 35dB.
    EXPECT_GT(sqnr, 35);
}

// Test quantization of weights which all have the same value.
// In this test case we test the following methods of the IQuantizationAlgorithm
// interface: NumberDistributionToFxpFormat().
TYPED_TEST(TestQuantizationLibTfEnhancedCpuGpu, SANITY_AllSameValue)
{
    if (!CheckRunTest<TypeParam>())
        return;
    typedef typename TypeParam::dataType DataType;

    // Create quantizer object
    vector<string> layer_names;
    vector<int> bw_activations;
    int bw                              = 8;
    unique_ptr<IQuantizer<DataType>> iq = unique_ptr<IQuantizer<DataType>>(
        GetQuantizerInstance<DataType>(layer_names, TypeParam::modeCpuGpu, bw_activations, QUANTIZATION_TF_ENHANCED));

    // Prepare data for gathering statistics.
    const int cnt = 8;
    TfEncoding encoding;

    // Second test: all data points are 10.
    vector<DataType> params1(8, 10);
    vector<DataType> expected1(8, 10);

    // Do second test.
    Blob<TypeParam> blob1(params1.data(), cnt);
    iq->QuantizeDequantizeParams(bw, blob1.getDataPtrOnDevice(), cnt, ROUND_NEAREST, blob1.getDataPtrOnDevice(),
                                 encoding);
    // Check if zero is in range.
    EXPECT_LE(encoding.min, 0);
    for (int i = 0; i < cnt; ++i)
    {
        EXPECT_NEAR(blob1.getDataPtrOnCpu()[i], expected1[i], 1e-3);
    }

    // Third test: all data points are -7.
    vector<DataType> params2(8, -7);
    vector<DataType> expected2(8, -7);

    // Do third test.
    Blob<TypeParam> blob2(params2.data(), cnt);
    iq->QuantizeDequantizeParams(bw, blob2.getDataPtrOnDevice(), cnt, ROUND_NEAREST, blob2.getDataPtrOnDevice(),
                                 encoding);
    // Check if zero is in range.
    EXPECT_GE(encoding.max, 0);
    for (int i = 0; i < cnt; ++i)
    {
        EXPECT_NEAR(blob2.getDataPtrOnCpu()[i], expected2[i], 1e-3);
    }
}

// Test quantization of weights with a gaussian distribution. The data
// distribution has a non-zero mean. We add some outliers very far away from the
// mean.
// We test quantization to 6 bit.
// To allow for a qualitative comparison, we quantize the same number
// distribution in QUANTIZATION_TF and QUANTIZATION_SQNR mode. Our
// implementation should yield a higher SQNR.
// In this test case we test the following methods of the IQuantizationAlgorithm
// interface: NumberDistributionToFxpFormat().
TYPED_TEST(TestQuantizationLibTfEnhancedCpuGpu, SANITY_CompareToOtherQuantizers)
{
    if (!CheckRunTest<TypeParam>())
        return;
    typedef typename TypeParam::dataType DataType;
    // Create quantizer object
    vector<string> layer_names;
    vector<int> bw_activations;
    int bw                                          = 6;
    unique_ptr<IQuantizer<DataType>> iq_tf_enhanced = unique_ptr<IQuantizer<DataType>>(
        GetQuantizerInstance<DataType>(layer_names, TypeParam::modeCpuGpu, bw_activations, QUANTIZATION_TF_ENHANCED));
    unique_ptr<IQuantizer<DataType>> iq_tf = unique_ptr<IQuantizer<DataType>>(
        GetQuantizerInstance<DataType>(layer_names, TypeParam::modeCpuGpu, bw_activations, QUANTIZATION_TF));

    // Prepare data for gathering statistics: Gaussian distribution with mean
    // 20 and a standard deviation of 5.
    const int cnt = 10000;
    vector<DataType> reference(cnt);
    unsigned seed = 0;
    std::mt19937 generator(seed);
    double mean   = 20;
    double stddev = 5;
    normal_distribution<double> distribution(mean, stddev);
    for (int i = 0; i < cnt - 2; ++i)
    {
        reference[i] = distribution(generator);
    }
    // Add two extreme outliers to the number distribution.
    reference[cnt - 2] = mean + 10 * stddev;
    reference[cnt - 1] = mean - 10 * stddev;
    // Prepare input vectors.
    vector<DataType> data_tf_enhanced = reference;
    vector<DataType> data_tf          = reference;

    Blob<TypeParam> blob_tf_enhanced(data_tf_enhanced.data(), cnt);
    Blob<TypeParam> blob_tf(data_tf.data(), cnt);

    // Do quantization.
    TfEncoding encoding_tf_enhanced;
    TfEncoding encoding_tf;
    iq_tf_enhanced->QuantizeDequantizeParams(bw, blob_tf_enhanced.getDataPtrOnDevice(), cnt, ROUND_NEAREST,
                                             blob_tf_enhanced.getDataPtrOnDevice(), encoding_tf_enhanced);
    iq_tf->QuantizeDequantizeParams(bw, blob_tf.getDataPtrOnDevice(), cnt, ROUND_NEAREST, blob_tf.getDataPtrOnDevice(),
                                    encoding_tf);

    // Perform a basic test for the encoding.
    // We compare the encodings of QUANTIZATION_TF_ENHANCED and QUANTIZATION_TF.
    // The first second one will choose the min and max according to the full
    // number distribution. In comparison, the first one should be more resilient
    // to outliers in the distribution, and thus the min should be larger and
    // the max smaller.
    EXPECT_GT(encoding_tf_enhanced.min, encoding_tf.min);
    EXPECT_LT(encoding_tf_enhanced.max, encoding_tf.max);

    std::cout << "TfEnhanced (min/max): " << encoding_tf_enhanced.min << "," << encoding_tf_enhanced.max << "\n";
    std::cout << "Tf         (min/max): " << encoding_tf.min << "," << encoding_tf.max << "\n";
}

// Test the computation of delta and offset, given a min and max.
// In this test case we test the following methods of the IQuantizationAlgorithm
// interface: ComputeDeltaAndOffsetModeSpecific().
TYPED_TEST(TestQuantizationLibTfEnhancedCpuGpu, SANITY_ComputeDeltaOffset)
{
    if (!CheckRunTest<TypeParam>())
        return;
    typedef typename TypeParam::dataType DataType;
    vector<string> layer_names;
    vector<int> bw_activations;
    unique_ptr<IQuantizer<DataType>> iq = unique_ptr<IQuantizer<DataType>>(
        GetQuantizerInstance<DataType>(layer_names, TypeParam::modeCpuGpu, bw_activations, QUANTIZATION_TF_ENHANCED));
    double min = -18;
    double max = 10;
    int bw     = 6;
    double delta, offset;
    double delta_expected  = 0.4444;
    double offset_expected = -41;
    double min_expected    = -18.2222;
    double max_expected    = 9.7778;
    iq->ComputeDeltaAndOffset(bw, min, max, delta, offset);
    EXPECT_NEAR(min, min_expected, 1e-4);
    EXPECT_NEAR(max, max_expected, 1e-4);
    EXPECT_NEAR(delta, delta_expected, 1e-4);
    EXPECT_NEAR(offset, offset_expected, 1e-4);
}
