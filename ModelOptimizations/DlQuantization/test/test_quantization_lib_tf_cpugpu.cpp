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


#include <cstdint>
#include <iostream>
#include <map>
#include <math.h>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "DlQuantization/IQuantizer.hpp"
#include "DlQuantization/Quantization.hpp"
#include "DlQuantization/QuantizerFactory.hpp"
#include "test_quantization_lib.hpp"

using namespace std;
using namespace DlQuantization;

template <typename TypeParam>
class TestQuantizationLibTfCpuGpu : public ::testing::Test
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

// Test on CPU and GPU with float and double
TYPED_TEST_CASE(TestQuantizationLibTfCpuGpu, TestDataTypesAndDevices);

// Test parameter quantization for bw=8, number range: -50...80
TYPED_TEST(TestQuantizationLibTfCpuGpu, SANITY_UnpackedParam)
{
    if (!CheckRunTest<TypeParam>())
        return;
    typedef typename TypeParam::dataType DataType;
    vector<string> layer_names;
    vector<int> bw_activations;
    unique_ptr<IQuantizer<DataType>> iq = unique_ptr<IQuantizer<DataType>>(
        GetQuantizerInstance<DataType>(layer_names, TypeParam::modeCpuGpu, bw_activations, QUANTIZATION_TF));
    DataType data[]     = {-40, -1, 0, 1, 2, -50, 80};
    DataType expected[] = {-39.7647, -1.0196, 0, 1.0196, 2.0392, -49.9608, 80.0392};
    int cnt             = 7;
    int bw              = 8;
    Blob<TypeParam> blob(data, cnt);
    TfEncoding encoding;
    iq->QuantizeDequantizeParams(bw, blob.getDataPtrOnDevice(), cnt, ROUND_NEAREST, blob.getDataPtrOnDevice(),
                                 encoding);
    for (int i = 0; i < cnt; ++i)
    {
        EXPECT_NEAR(blob.getDataPtrOnCpu()[i], expected[i], 1e-4);
    }
}

// Activation quantization, encoding given externally.
// Test various quantization corner cases.
TYPED_TEST(TestQuantizationLibTfCpuGpu, SANITY_UnpackedActsExternal)
{
    if (!CheckRunTest<TypeParam>())
        return;
    typedef typename TypeParam::dataType DataType;
    // Create quantizer object
    vector<string> layer_names;
    string layer_name("conv1");
    layer_names.push_back(layer_name);
    vector<int> bw_activations;
    int bw = 8;
    bw_activations.push_back(bw);
    unique_ptr<IQuantizer<DataType>> iq = unique_ptr<IQuantizer<DataType>>(
        GetQuantizerInstance<DataType>(layer_names, TypeParam::modeCpuGpu, bw_activations, QUANTIZATION_TF));
    // Create fixed point encoding
    // encoding: delta=1, offset=0, min=0, max=255, bw=8
    TfEncoding encoding = {0, 255, 1, 0, 8};
    vector<TfEncoding> encoding_in;
    encoding_in.push_back(encoding);
    vector<TfEncoding> encoding_out;
    TfEncodingLayer encoding_layer = {encoding_in, encoding_out};
    map<string, TfEncodingLayer> encoding_map;
    encoding_map[layer_name] = encoding_layer;
    iq->SetEncoding(encoding_map);
    // Prepare test data:
    // underflow x2, smallest possible, tie round up, tie round up, exact,
    // round down, round up, largest possible, overflow, overflow
    DataType data[]     = {-10, -0.5, 0, 1.5, 2.5, 3, 3.1, 3.9, 255, 255.1, 300};
    DataType expected[] = {0, 0, 0, 2, 3, 3, 3, 4, 255, 255, 255};
    int cnt             = 11;
    vector<DataType*> data_vec;
    Blob<TypeParam> blob(data, cnt);
    data_vec.push_back(blob.getDataPtrOnDevice());
    vector<size_t> count_vec;
    count_vec.push_back(cnt);
    // Perform quantization
    vector<TfEncoding> encoding_result_vec;
    iq->QuantizeDequantizeActs(layer_name, LAYER_INPUT, bw, data_vec, count_vec, data_vec, encoding_result_vec);
    for (int i = 0; i < cnt; ++i)
    {
        EXPECT_EQ(blob.getDataPtrOnCpu()[i], expected[i]);
    }
}

// Activation quantization using statistical data.
// Data to be quantized: underflow, overflow, round.
TYPED_TEST(TestQuantizationLibTfCpuGpu, SANITY_UnpackedActsStats)
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
        GetQuantizerInstance<DataType>(layer_names, TypeParam::modeCpuGpu, bw_activations, QUANTIZATION_TF));
    // Prepare data for gathering statistics
    DataType acts0[] = {-100, 0, 125};
    int cnt0         = 3;
    vector<const DataType*> acts0_vector;
    Blob<TypeParam> blob0(acts0, cnt0);
    acts0_vector.push_back(blob0.getDataPtrOnDevice());
    // Pass data into library to gather statistics
    vector<size_t> count;
    count.push_back(cnt0);
    iq->UpdateStats(layer_name, LAYER_INPUT, acts0_vector, count);
    // Prepare data to be quantized:
    // Overflow, round, round, underflow
    DataType acts1[]    = {200, 100, -50, -500};
    DataType expected[] = {125.2941, 99.7059, -50.2941, -99.7059};
    int cnt1            = 4;
    count.clear();
    count.push_back(cnt1);
    vector<DataType*> acts1_vector;
    Blob<TypeParam> blob1(acts1, cnt1);
    acts1_vector.push_back(blob1.getDataPtrOnDevice());
    vector<TfEncoding> encoding_vector;
    iq->QuantizeDequantizeActs(layer_name, LAYER_INPUT, bw, acts1_vector, count, acts1_vector, encoding_vector);
    // Check encoding from QuantizeActs()
    TfEncoding encoding_expected;
    this->GetTfEncoding(-100, 125, bw, encoding_expected);
    EXPECT_TRUE(this->CompareEncodings(encoding_expected, encoding_vector[0]));
    // Check quantized data
    for (int i = 0; i < cnt1; ++i)
    {
        EXPECT_NEAR(blob1.getDataPtrOnCpu()[i], expected[i], 1e-4);
    }
}
