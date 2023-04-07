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

#include <cstdint>
#include "QuantizeDequantizeUtils.hpp"
#include "AimetOpUtils.h"
#include "cuda_runtime_api.h"
#include <gtest/gtest.h>
#include <limits>
#include <DlQuantization/Quantization.hpp>
#include <DlQuantization/QuantizerFactory.hpp>
#include "OnnxOpUtils.h"
#include <DlQuantization/TensorQuantizerOpFacade.h>
#include <DlQuantization/TensorQuantizer.h>
#include <cmath>

double computeDelta(double encodingMin, double encodingMax, double numSteps)
{
    double delta = (encodingMax - encodingMin) / numSteps;
    return delta;
}


double computeOffset(double encodingMin, double delta)
{
    double offset = round(encodingMin / delta);

    return offset;
}


class TestOnnxTensorOps : public ::testing::Test
{
};


TEST(TestOnnxTensorOps, TensorChannelSlice)
{
    int ch_out = 16;
    int ch_in = 8;
    int k = 3;
    int total_elements = ch_out * ch_in * k * k;
    std::vector<int64_t> dimensions{ch_out, ch_in, k, k};
    float data[ch_out][ch_in][k][k];
    for (int c_o = 0; c_o < ch_out; c_o++){
        for (int c_i = 0; c_i < ch_in; c_i++){
            for (int i = 0; i < k; i++){
                for (int j = 0; j < k; j++){
                    data[c_o][c_i][i][j] = (float) c_i;
                }
            }
        }
    }
    void* device_data;
    cudaMalloc(&device_data, sizeof(float) * total_elements);
    cudaMemcpy(device_data, data, total_elements * sizeof(float), cudaMemcpyHostToDevice);
    void* result_data;
    int copy_len = ch_out * k * k;
    cudaMalloc(&result_data, sizeof(float) * copy_len);
    int channel = 3;
    long copy_width = k * k;
    long output_stride = copy_width;
    long iters = ch_out;
    long input_stride = ch_in * copy_width;
    long input_offset = channel * copy_width;
    long output_offset = 0;
    cudaDeviceSynchronize();


    sliceTensorChannelGPU((float*) device_data, (float*) result_data, iters, copy_width, input_stride,
                               output_stride, input_offset, output_offset);
    float* result = static_cast<float*>(malloc(sizeof(float) * copy_len));

    cudaMemcpy(result, result_data, copy_len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < copy_len; i++){
        EXPECT_EQ((float) channel, result[i]);
    }


    cudaFree(device_data);
    cudaFree(result_data);
    free(result);
}


TEST(TestOnnxTensorOps, TensorChannelSliceCPU)
{
    int axis = 0;
    int ch_out = 16;
    int ch_in = 8;
    int k = 3;
    int total_elements = ch_out * ch_in * k * k;
    std::vector<int64_t> dimensions{ch_out, ch_in, k, k};
    float data[ch_out][ch_in][k][k];
    float output[total_elements/dimensions[axis]];
    for (int c_o = 0; c_o < ch_out; c_o++){
        for (int c_i = 0; c_i < ch_in; c_i++){
            for (int i = 0; i < k; i++){
                for (int j = 0; j < k; j++){
                    data[c_o][c_i][i][j] = (float) c_o;
                }
            }
        }
    }


    int copy_len = total_elements/dimensions[axis];
    int channel = 3;
    cudaDeviceSynchronize();


    sliceTensorAlongAxis<float>((float *)data, dimensions, axis, channel, (float*)output, false);



    cudaDeviceSynchronize();
    for (int i = 0; i < copy_len; i++){
        EXPECT_EQ((float) channel, output[i]);
    }

}


TEST(TestOnnxTensorOps, PerChannelQuantDequant)
{
    OnnxCudaAllocator _allocator;
    OnnxCudaAllocator* allocator = &_allocator;
    const int ch_out = 4;
    const int ch_in = 6;
    const int k = 1;
    int total_elements = ch_out * ch_in * k * k;
    std::vector<int64_t> dimensions{ch_out, ch_in, k, k};
    float data[ch_out][ch_in][k][k] = {{-7, -5, -3, 0, .1, 2.5},
                                        {-7, -5, -3, 0, .1, 2.5},
                                        {-7, -5, -3, 0, .1, 2.5},
                                        {-7, -5, -3, 0, .1, 2.5}};

    float expected_out[ch_out][ch_in][k][k] = {{-3.84, -3.84, -3, 0, .089999996, 2.49},
                                               {-3.84, -3.84, -3, 0, .089999996, 2.49},
                                               {-3.84, -3.84, -3, 0, .089999996, 2.49},
                                               {-6.4, -5, -3, 0, .1, 2.5}};

    std::vector<DlQuantization::TfEncoding*> encodings;
    for (int i = 0; i < ch_out; i++) {
        auto enc = new DlQuantization::TfEncoding();
        encodings.push_back(enc);
        encodings[i]->max = 3.81;
        encodings[i]->min = -3.84;
        encodings[i]->delta = 0.03;
        encodings[i]->bw = 8;
    }
    encodings[3]->bw = 8;
    encodings[3]->max = 6.35;
    encodings[3]->min = -6.4;
    encodings[3]->delta = 0.05;
    DlQuantization::TensorQuantizer tensorQuant = DlQuantization::TensorQuantizer(DlQuantization::QuantizationMode::QUANTIZATION_TF, DlQuantization::RoundingMode::ROUND_NEAREST);
    tensorQuant.setStrictSymmetric(false);
    tensorQuant.setUnsignedSymmetric(false);
    bool useSymmetricEncoding = false;
    DlQuantization::TensorQuantizer* tensorQuantizerReference = &tensorQuant;
    void* device_data;
    cudaMalloc(&device_data, sizeof(float) * total_elements);
    cudaMemcpy(device_data, data, total_elements * sizeof(float), cudaMemcpyHostToDevice);
    void* result_data;
    int axis = 0;
    cudaMalloc(&result_data, sizeof(float) * total_elements);

    quantizeDequantizePerChannelGPU((float*)device_data,(float*) result_data, dimensions, axis, encodings,
                                    allocator);
    float result[ch_out][ch_in][k][k];

    cudaMemcpy(result, result_data, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int c_o = 0; c_o < ch_out; c_o++){
        for (int c_i = 0; c_i < ch_in; c_i++){
            for (int i = 0; i < k; i++){
                for (int j = 0; j < k; j++){
                    EXPECT_NEAR(expected_out[c_o][c_i][i][j], result[c_o][c_i][i][j], 0.0001);
                }
            }
        }
    }

    for (auto & encoding : encodings)
    {
        delete encoding;
    }
    encodings.clear();
    cudaFree(device_data);
    cudaFree(result_data);
}

TEST(TestOnnxTensorOps, PerChannelQuantDequantTranspose)
{
    OnnxCudaAllocator _allocator;
    OnnxCudaAllocator* allocator = &_allocator;
    const int ch_out = 6;
    const int ch_in = 4;
    const int k = 1;
    int total_elements = ch_out * ch_in * k * k;
    std::vector<int64_t> dimensions{ch_out, ch_in, k, k};
    float data[ch_out][ch_in][k][k] = {{-7, -7, -7, -7},
                                       {-5, -5, -5, -5},
                                       {-3, -3, -3, -3},
                                       {0, 0, 0, 0},
                                       {.1, .1, .1, .1},
                                       {2.5, 2.5, 2.5, 2.5}};


    float expected_out[ch_in][ch_out][k][k] = {{-3.84, -3.84, -3, 0, .089999996, 2.49},
                                               {-3.84, -3.84, -3, 0, .089999996, 2.49},
                                               {-3.84, -3.84, -3, 0, .089999996, 2.49},
                                               {-6.4, -5, -3, 0, .1, 2.5}};

    std::vector<DlQuantization::TfEncoding*> encodings;
    for (int i = 0; i < ch_out; i++) {
        auto enc = new DlQuantization::TfEncoding();
        encodings.push_back(enc);
        encodings[i]->max = 3.81;
        encodings[i]->min = -3.84;
        encodings[i]->delta = 0.03;
        encodings[i]->bw = 8;
    }
    encodings[3]->bw = 8;
    encodings[3]->max = 6.35;
    encodings[3]->min = -6.4;
    encodings[3]->delta = 0.05;
    DlQuantization::TensorQuantizer tensorQuant = DlQuantization::TensorQuantizer(DlQuantization::QuantizationMode::QUANTIZATION_TF, DlQuantization::RoundingMode::ROUND_NEAREST);
    tensorQuant.setStrictSymmetric(false);
    tensorQuant.setUnsignedSymmetric(false);
    bool useSymmetricEncoding = false;
    DlQuantization::TensorQuantizer* tensorQuantizerReference = &tensorQuant;
    void* device_data;
    cudaMalloc(&device_data, sizeof(float) * total_elements);
    cudaMemcpy(device_data, data, total_elements * sizeof(float), cudaMemcpyHostToDevice);
    void* result_data;
    int axis = 1;
    cudaMalloc(&result_data, sizeof(float) * total_elements);

    quantizeDequantizePerChannelGPU((float*)device_data,(float*) result_data, dimensions, axis, encodings,
                                    allocator);
    float result[ch_out][ch_in][k][k];

    cudaMemcpy(result, result_data, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int c_o = 0; c_o < ch_out; c_o++){
        for (int c_i = 0; c_i < ch_in; c_i++){
            for (int i = 0; i < k; i++){
                for (int j = 0; j < k; j++){
                    EXPECT_NEAR(expected_out[c_i][c_o][i][j], result[c_o][c_i][i][j], 0.0001);
                }
            }
        }
    }

    for (auto & encoding : encodings)
    {
        delete encoding;
    }
    encodings.clear();

    cudaFree(device_data);
    cudaFree(result_data);
}

TEST(TestOnnxTensorOps, PerChannelQuantDequantCPU)
{
    OnnxCudaAllocator _allocator;
    OnnxCudaAllocator* allocator = &_allocator;
    const int ch_out = 4;
    const int ch_in = 6;
    const int k = 1;
    int total_elements = ch_out * ch_in * k * k;
    std::vector<int64_t> dimensions{ch_out, ch_in, k, k};
    float data[ch_out][ch_in][k][k] = {{-7, -5, -3, 0, .1, 2.5},
                                       {-7, -5, -3, 0, .1, 2.5},
                                       {-7, -5, -3, 0, .1, 2.5},
                                       {-7, -5, -3, 0, .1, 2.5}};

    float expected_out[ch_out][ch_in][k][k] = {{-3.84, -3.84, -3, 0, .089999996, 2.49},
                                               {-3.84, -3.84, -3, 0, .089999996, 2.49},
                                               {-3.84, -3.84, -3, 0, .089999996, 2.49},
                                               {-6.4, -5, -3, 0, .1, 2.5}};

    std::vector<DlQuantization::TfEncoding*> encodings;
    for (int i = 0; i < ch_out; i++) {
        auto enc = new DlQuantization::TfEncoding();
        encodings.push_back(enc);
        encodings[i]->max = 3.81;
        encodings[i]->min = -3.84;
        encodings[i]->bw = 8;
    }
    encodings[3]->bw = 8;
    encodings[3]->max = 6.35;
    encodings[3]->min = -6.4;
    DlQuantization::TensorQuantizer tensorQuant = DlQuantization::TensorQuantizer(DlQuantization::QuantizationMode::QUANTIZATION_TF, DlQuantization::RoundingMode::ROUND_NEAREST);
    tensorQuant.setStrictSymmetric(false);
    tensorQuant.setUnsignedSymmetric(false);
    bool useSymmetricEncoding = false;
    DlQuantization::TensorQuantizer* tensorQuantizerReference = &tensorQuant;

    int axis = 0;


    float result[ch_out][ch_in][k][k];
    quantizeDequantizePerChannelCPU((float*)data,(float*) result, dimensions, axis, encodings);

    ;
    cudaDeviceSynchronize();
    for (int c_o = 0; c_o < ch_out; c_o++){
        for (int c_i = 0; c_i < ch_in; c_i++){
            for (int i = 0; i < k; i++){
                for (int j = 0; j < k; j++){
                    EXPECT_NEAR(expected_out[c_o][c_i][i][j], result[c_o][c_i][i][j], 0.0001);
                }
            }
        }
    }
    for (auto & encoding : encodings)
    {
        delete encoding;
    }
    encodings.clear();

}


