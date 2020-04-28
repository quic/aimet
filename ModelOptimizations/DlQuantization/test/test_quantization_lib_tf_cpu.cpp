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


#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <math.h>
#include <memory>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "DlQuantization/IQuantizer.hpp"
#include "DlQuantization/Quantization.hpp"
#include "DlQuantization/QuantizerFactory.hpp"
#include "test_quantization_lib.hpp"

using namespace std;
using namespace DlQuantization;

template <typename TypeParam>
class TestQuantizationLibTfCpu : public ::testing::Test
{
protected:
    // Get TensorFlow encoding.
    // Assume a non-skewed distribution.
    // Round offset to fixed point, and adjust min and max accordingly.
    void GetTfEncoding(double min, double max, int bw, TfEncoding& encoding)
    {
        int steps       = pow(2, bw) - 1;
        encoding.delta  = (max - min) / (double) steps;
        encoding.offset = round(min / encoding.delta);
        encoding.min    = encoding.delta * encoding.offset;
        encoding.max    = encoding.delta * steps + encoding.min;
        encoding.bw     = bw;
    }

    bool CompareEncodings(TfEncoding e0, TfEncoding e1)
    {
        return e0.min == e1.min && e0.max == e1.max && e0.delta == e1.delta && e0.offset == e1.offset && e0.bw == e1.bw;
    }

    void PrintEncoding(TfEncoding encoding)
    {
        cout << "Encoding: min: " << encoding.min << ", max: " << encoding.max << ", delta: " << encoding.delta
             << ", offset: " << encoding.offset << ", bw: " << encoding.bw << endl;
    }
};

// Test on CPU with float and double
TYPED_TEST_CASE(TestQuantizationLibTfCpu, TestDtypes);

// Test UpdateStats with empty activations vector.
// Test QuantizeActs where stats are such that min == max.
TYPED_TEST(TestQuantizationLibTfCpu, SANITY_SpecialInputV0)
{
    // Create quantizer object
    vector<string> layer_names;
    string layer_name("conv0");
    layer_names.push_back(layer_name);
    vector<int> bw_activations;
    int bw = 8;
    bw_activations.push_back(bw);
    unique_ptr<IQuantizer<TypeParam>> iq = unique_ptr<IQuantizer<TypeParam>>(
        GetQuantizerInstance<TypeParam>(layer_names, COMP_MODE_CPU, bw_activations, QUANTIZATION_TF));
    // Prepare data for gathering statistics
    TypeParam acts0[] = {};
    int cnt0          = 0;
    TypeParam acts1[] = {5, 5, 5, 5};
    int cnt1          = 4;
    // Pass data into library to gather statistics
    // Iteration 0: empty activations vector
    vector<const TypeParam*> acts_vector;
    acts_vector.push_back(acts0);
    vector<size_t> count;
    count.push_back(cnt0);
    iq->UpdateStats(layer_name, LAYER_INPUT, acts_vector, count);
    // Iteration 1: many activations with all the same value
    acts_vector.clear();
    acts_vector.push_back(acts1);
    count.clear();
    count.push_back(cnt1);
    iq->UpdateStats(layer_name, LAYER_INPUT, acts_vector, count);
    // Quantize the same data we used to gather statistics: the value 5.
    TypeParam data[]   = {5};
    int cnt2           = 1;
    TypeParam expected = 5;
    vector<TypeParam*> data_vector;
    data_vector.push_back(data);
    vector<TfEncoding> encoding_vector;
    count.clear();
    count.push_back(cnt2);
    iq->QuantizeDequantizeActs(layer_name, LAYER_INPUT, 8, data_vector, count, data_vector, encoding_vector);
    // Check quantized data is the same as original data.
    EXPECT_EQ(data_vector[0][0], expected);
}

// Test ComputeDeltaAndOffset().
// Test the conversion of min/max into delta/offset.
// Test different corner cases.
TYPED_TEST(TestQuantizationLibTfCpu, SANITY_ComputeDeltaOffset)
{
    // Create quantizer object
    vector<string> layer_names;
    vector<int> bw_activations;
    int bw = 8;
    double min;
    double max;
    double delta;
    double offset;
    double min_expected;
    double max_expected;
    double delta_expected;
    double offset_expected;
    unique_ptr<IQuantizer<TypeParam>> iq = unique_ptr<IQuantizer<TypeParam>>(
        GetQuantizerInstance<TypeParam>(layer_names, COMP_MODE_CPU, bw_activations, QUANTIZATION_TF));
    // Test case 1): min/max = -40/80. This is the "normal" case.
    min             = -40;
    max             = 80;
    min_expected    = min;
    max_expected    = max;
    delta_expected  = 0.4706;
    offset_expected = -85;
    iq->ComputeDeltaAndOffset(bw, min, max, delta, offset);
    EXPECT_EQ(min, min_expected);
    EXPECT_EQ(max, max_expected);
    EXPECT_NEAR(delta, delta_expected, 1e-4);
    EXPECT_EQ(offset, offset_expected);
    // Test case 2): min/max = 10/10. Here we test special case where min==max.
    // Min will get adjusted to include 0 in the range.
    min             = 10;
    max             = 10;
    min_expected    = 0;
    max_expected    = max;
    delta_expected  = 0.0392;
    offset_expected = 0;
    iq->ComputeDeltaAndOffset(bw, min, max, delta, offset);
    EXPECT_EQ(min, min_expected);
    EXPECT_EQ(max, max_expected);
    EXPECT_NEAR(delta, delta_expected, 1e-4);
    EXPECT_EQ(offset, offset_expected);
    // Test case 3): min/max = 0/0. Here we test special case where min==max==0.
    // What should happen is that max gets slightly increased so we have a real
    // range of values.
    min             = 0;
    max             = 0;
    min_expected    = 0;
    max_expected    = 0.01;
    delta_expected  = 0.01 / 255;
    offset_expected = 0;
    iq->ComputeDeltaAndOffset(bw, min, max, delta, offset);
    EXPECT_EQ(min, min_expected);
    EXPECT_EQ(max, max_expected);
    EXPECT_NEAR(delta, delta_expected, 1e-4);
    EXPECT_EQ(offset, offset_expected);
    // Test case 3): Test skewed number distribution. min/max = 1000/1200.
    // Min should get adjusted to 0, so 0 is representable.
    min             = 1000;
    max             = 1200;
    min_expected    = 0;
    max_expected    = max;
    delta_expected  = 4.7059;
    offset_expected = 0;
    iq->ComputeDeltaAndOffset(bw, min, max, delta, offset);
    EXPECT_EQ(min, min_expected);
    EXPECT_EQ(max, max_expected);
    EXPECT_NEAR(delta, delta_expected, 1e-4);
    EXPECT_EQ(offset, offset_expected);
    // Test case 4): Test skewed number distribution. min/max = -8/-5.
    // Max should get adjusted to 0, so 0 is representable.
    min             = -8;
    max             = -5;
    min_expected    = min;
    max_expected    = 0;
    delta_expected  = 0.0314;
    offset_expected = -255;
    iq->ComputeDeltaAndOffset(bw, min, max, delta, offset);
    EXPECT_EQ(min, min_expected);
    EXPECT_EQ(max, max_expected);
    EXPECT_NEAR(delta, delta_expected, 1e-4);
    EXPECT_EQ(offset, offset_expected);
}