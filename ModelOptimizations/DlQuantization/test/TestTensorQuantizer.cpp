// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <random>
#include <type_traits>

#include "DlQuantization/TensorQuantizer.h"
#include "test_quantization_lib.hpp"

using namespace DlQuantization;

class TestTensorQuantizer : public ::testing::Test
{
protected:
    std::vector<float> data1, data2, data3, data4;
    std::vector<uint32_t> shape1, shape2, shape3;

    void SetUp()
    {
        if (data1.size() == 0)
        {
            data1.resize(24);
            std::iota(std::begin(data1), std::end(data1), 0);
            shape1 = {2, 3, 2, 2};
        }

        if (data2.size() == 0)
        {
            data2.resize(60);
            float t = -15;
            for (uint32_t i = 0; i < data2.size(); ++i)
            {
                data2[i] = t;
                t += 0.5;
            }
            shape2 = {1, 4, 5, 3};
        }

        if (data3.size() == 0)
        {
            shape3 = {2, 5, 4, 1};
            data3.resize(40);
            std::mt19937 eng;
            std::normal_distribution<float> dist;
            for (auto& d: data3)
            {
                d = dist(eng);
            }
            std::iota(std::begin(data1), std::end(data1), 0);
        }

        if (data4.size() == 0)
        {
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

// qtype should stay stable even when legacy bitwidth metadata changes.
TEST_F(TestTensorQuantizer, SetEncodingsDoesNotChangeFp8QuantizationType)
{
    BlockTensorQuantizer tensorQuantizer({}, QuantizationType::Fp8E4M3FN(), QUANTIZATION_TF);

    TfEncoding encoding;
    encoding.bw     = 4;
    encoding.delta  = 1.0;
    encoding.offset = 0.0;
    encoding.min    = -448.0;
    encoding.max    = 448.0;

    tensorQuantizer.setEncodings({encoding});

    EXPECT_TRUE(tensorQuantizer.getQuantizationType().isFloat());
    EXPECT_EQ(tensorQuantizer.getQuantizationType().bitwidth(), 8);
    EXPECT_EQ(tensorQuantizer.bitwidth, 4);
}

TEST_F(TestTensorQuantizer, LegacyBitwidthMutationDoesNotChangeQuantizationType)
{
    BlockTensorQuantizer tensorQuantizer({}, 8, QUANTIZATION_TF);

    tensorQuantizer.bitwidth = 4;

    EXPECT_TRUE(tensorQuantizer.getQuantizationType().isInt());
    EXPECT_EQ(tensorQuantizer.getQuantizationType().bitwidth(), 8);
    EXPECT_EQ(tensorQuantizer.bitwidth, 4);
}

TEST_F(TestTensorQuantizer, SanityTestTfEnhancedPerTensorQdqCpu)
{
    BlockTensorQuantizer tensorQuantizer({}, 8, QUANTIZATION_TF_ENHANCED);
    tensorQuantizer.setStrictSymmetric(false);
    tensorQuantizer.setUnsignedSymmetric(false);
    TensorDims tensorShape = {(TensorDim) data4.size()};

    tensorQuantizer.updateStats(data4.data(), tensorShape, false);
    EXPECT_FALSE(tensorQuantizer.isEncodingValid);
    auto encodings = tensorQuantizer.computeEncodings(false);
    tensorQuantizer.setEncodings(encodings);
    EXPECT_TRUE(tensorQuantizer.isEncodingValid);
    auto encoding = encodings[0];

    std::vector<float> inputTensor(data4.size(), 5);
    std::vector<float> quantizedTensor(data4.size());
    tensorQuantizer.quantizeDequantize(inputTensor.data(), quantizedTensor.data(), tensorShape, false);

    double minVal = 0;
    double maxVal = 0;
    for (auto x : data4)
    {
        minVal = std::min(minVal, static_cast<double>(x));
        maxVal = std::max(maxVal, static_cast<double>(x));
    }

    size_t PDF_SIZE = 512 * 3;
    double HISTOGRAM_BUCKET_SIZE = 3 * (maxVal - minVal) / PDF_SIZE;
    // Allow for worst case 2 x HISTOGRAM_BUCKET_SIZE error on (maxVal - minVal)
    EXPECT_NEAR(encoding.delta, (maxVal - minVal) / 255, 2 * HISTOGRAM_BUCKET_SIZE / 255);
    // Allow for worst case 1 x HISTOGRAM_BUCKET_SIZE error on minValue and 0.5 x delta from rounding offset
    EXPECT_NEAR(encoding.min, minVal, HISTOGRAM_BUCKET_SIZE + 0.5 * encoding.delta);
    EXPECT_NEAR(encoding.max, maxVal, HISTOGRAM_BUCKET_SIZE + 0.5 * encoding.delta);

    EXPECT_NE(inputTensor[0], quantizedTensor[0]);
    EXPECT_NEAR(quantizedTensor[0], inputTensor[0], 0.5 * encoding.delta);
}


TEST_F(TestTensorQuantizer, SanityTestComputeEncodingsAsymmetricTfEnhanced)
{
    auto paramTensor = this->data4;
    BlockTensorQuantizer tensorQuantizer({}, 8, QUANTIZATION_TF_ENHANCED);
    tensorQuantizer.setStrictSymmetric(false);
    tensorQuantizer.setUnsignedSymmetric(false);
    TensorDims tensorShape = {(TensorDim) paramTensor.size()};

    tensorQuantizer.updateStats(paramTensor.data(), tensorShape, false);
    auto encoding = tensorQuantizer.computeEncodings(false)[0];

    double MAX = *std::max_element(paramTensor.begin(), paramTensor.end());
    double MIN = *std::min_element(paramTensor.begin(), paramTensor.end());
    size_t PDF_SIZE = 512;
    double HISTOGRAM_BUCKET_SIZE = 3 * (MAX - MIN) / PDF_SIZE;
    EXPECT_NEAR(encoding.min, MIN, HISTOGRAM_BUCKET_SIZE + 0.5 * encoding.delta);
    EXPECT_NEAR(encoding.max, MAX, HISTOGRAM_BUCKET_SIZE + 0.5 * encoding.delta);
    EXPECT_EQ(encoding.bw, 8);
    EXPECT_GT(encoding.delta, 0);
    EXPECT_LE(encoding.offset, 0);
}


// Check that existing tests pass with BlockTensorQuantizer
TEST_F(TestTensorQuantizer, SanityTestComputeEncodingFromDataSymmetricTFBlocked)
{
    auto paramTensor = this->data4;
    BlockTensorQuantizer tensorQuantizer({}, 8, QUANTIZATION_TF);
    tensorQuantizer.setStrictSymmetric(true);
    TensorDims tensorShape = {(TensorDim) paramTensor.size()};
    ASSERT_FALSE(tensorQuantizer.hasValidStats());
    tensorQuantizer.updateStats(paramTensor.data(), tensorShape, false);\
    ASSERT_TRUE(tensorQuantizer.hasValidStats());
    ASSERT_FALSE(tensorQuantizer.isEncodingValid);
    auto encoding = tensorQuantizer.computeEncodings(true)[0];

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


TEST_F(TestTensorQuantizer, SANITY_GeneratePerChannelParamsBlocked)
{
    BlockTensorQuantizer tensorQuantizer({3, 1, 1}, 8, QUANTIZATION_TF);
    tensorQuantizer.updateStats(data1.data(), {2, 3, 2, 2}, false);
    auto encodings = tensorQuantizer.computeEncodings(false);

    std::vector<TfEncoding> expectedEncodings(3);
    expectedEncodings[0] = getTfEncoding(0, 15, 8);
    expectedEncodings[1] = getTfEncoding(0, 19, 8);
    expectedEncodings[2] = getTfEncoding(0, 23, 8);

    for (uint32_t i = 0; i < encodings.size(); ++i)
    {
        EXPECT_TRUE(compareEncodings(encodings[i], expectedEncodings[i]));
    }
}


// 1. QuantizeDequantize per channel using Asymmetric mode
TEST_F(TestTensorQuantizer, SANITY_QuantizeDequantizePerChannelTensorBlocked)
{
    TensorDims tensorShape = {1, 4, 5, 3};
    BlockTensorQuantizer tensorQuantizer({3}, 8, QUANTIZATION_TF);

    std::vector<TfEncoding> expectedEncodings(3);
    expectedEncodings[0] = getTfEncoding(-15, 13.5, 8);
    expectedEncodings[1] = getTfEncoding(-14.5, 14, 8);
    expectedEncodings[2] = getTfEncoding(-14, 14.5, 8);

    std::vector<float> expectedParams = {
        -14.9765, -14.5294, -13.9706, -13.5235, -12.9647, -12.5176, -11.9588, -11.5118, -10.9529, -10.5059,
        -9.94706, -9.5,     -9.05294, -8.49412, -8.04706, -7.48824, -7.04118, -6.48235, -6.03529, -5.47647,
        -5.02941, -4.47059, -4.02353, -3.46471, -3.01765, -2.45882, -2.01176, -1.45294, -1.00588, -0.447059,
        0,        0.447059, 1.00588,  1.45294,  2.01176,  2.45882,  3.01765,  3.46471,  4.02353,  4.47059,
        5.02941,  5.47647,  6.03529,  6.48235,  7.04118,  7.48824,  8.04706,  8.49412,  9.05294,  9.5,
        9.94706,  10.5059,  10.9529,  11.5118,  11.9588,  12.5176,  12.9647,  13.5235,  13.9706,  14.5294};

    tensorQuantizer.updateStats(this->data2.data(), tensorShape, false);
    auto encodings = tensorQuantizer.computeEncodings(false);
    tensorQuantizer.setEncodings(encodings);

    std::vector<float> params_quantized(this->data2.size());
    tensorQuantizer.quantizeDequantize(this->data2.data(), params_quantized.data(), tensorShape, false);

    ASSERT_EQ(encodings.size(), expectedEncodings.size());
    for (uint32_t i = 0; i < encodings.size(); ++i)
    {
        EXPECT_TRUE(compareEncodings(encodings[i], expectedEncodings[i]));
    }

    ASSERT_EQ(params_quantized.size(), expectedParams.size());
    for (uint32_t i = 0; i < expectedParams.size(); ++i)
    {
        EXPECT_NEAR(params_quantized[i], expectedParams[i], 0.001);
        EXPECT_NEAR(params_quantized[i], this->data2[i], 0.06);
    }
}

// QuantizeDequantize per channel using Symmetric mode
TEST_F(TestTensorQuantizer, SANITY_QuantizeDequantizePerChannelTensorSymmetricBlocked)
{
    TensorDims tensorShape = {1, 4, 5, 3};
    BlockTensorQuantizer tensorQuantizer({1, 1, 5, 1}, 8, QUANTIZATION_TF);
    tensorQuantizer.setStrictSymmetric(true);
    tensorQuantizer.setUnsignedSymmetric(false);

    std::vector<TfEncoding> expectedEncodings(5);
    expectedEncodings[0] = getTfSymmetricEncoding(15, 8);
    expectedEncodings[1] = getTfSymmetricEncoding(13.5, 8);
    expectedEncodings[2] = getTfSymmetricEncoding(12, 8);
    expectedEncodings[3] = getTfSymmetricEncoding(13, 8);
    expectedEncodings[4] = getTfSymmetricEncoding(14.5, 8);

    std::vector<float> expectedParams = {
        -15,      -14.5276, -14.0551, -13.5,    -12.9685, -12.5433, -12,      -11.5276, -10.9606, -10.5433,
        -10.0315, -9.51968, -9.01968, -8.44882, -7.99213, -7.44094, -6.9685,  -6.49606, -5.95276, -5.52756,
        -4.99606, -4.53543, -3.9685,  -3.49606, -2.9685,  -2.45669, -2.04724, -1.48425, -1.02756, -0.456693,
        0,        0.472441, 0.944882, 1.48819,  2.01969,  2.55118,  3.02362,  3.49606,  3.9685,   4.50394,
        5.01575,  5.52756,  6.05118,  6.50787,  6.96457,  7.55906,  8.0315,   8.50394,  9.03543,  9.46063,
        9.99213,  10.4882,  10.9606,  11.5276,  11.9764,  12.4882,  13,       13.4724,  14.0433,  14.5};

    tensorQuantizer.updateStats(this->data2.data(), tensorShape, false);
    auto encodings = tensorQuantizer.computeEncodings(true);
    tensorQuantizer.setEncodings(encodings);

    std::vector<float> params_quantized(this->data2.size());
    tensorQuantizer.quantizeDequantize(this->data2.data(), params_quantized.data(), tensorShape, false);

    ASSERT_EQ(encodings.size(), expectedEncodings.size());
    for (uint32_t i = 0; i < encodings.size(); ++i)
    {
        EXPECT_TRUE(compareEncodings(encodings[i], expectedEncodings[i]));
    }

    ASSERT_EQ(params_quantized.size(), expectedParams.size());
    for (uint32_t i = 0; i < expectedParams.size(); ++i)
    {
        EXPECT_NEAR(params_quantized[i], expectedParams[i], 0.0001);
        EXPECT_NEAR(params_quantized[i], this->data2[i], 0.06);
    }
}

#ifdef GPU_QUANTIZATION_ENABLED

// FP8 QDQ should be bit-exact across CPU and GPU.
TEST_F(TestTensorQuantizer, Fp8PerTensorQdqCpuGpuBitExact)
{
    const TensorDims tensorShape = {(TensorDim) data4.size()};

    TfEncoding encoding;
    encoding.bw     = 8;
    encoding.delta  = 0.5;
    encoding.offset = 0;
    encoding.min    = -448.0 * encoding.delta;
    encoding.max    = 448.0 * encoding.delta;

    BlockTensorQuantizer cpuQuantizer({}, QuantizationType::Fp8E4M3FN(), QUANTIZATION_TF);
    cpuQuantizer.setEncodings({encoding});
    std::vector<float> cpuOutput(data4.size(), 0);
    cpuQuantizer.quantizeDequantize(data4.data(), cpuOutput.data(), tensorShape, false);

    BlockTensorQuantizer gpuQuantizer({}, QuantizationType::Fp8E4M3FN(), QUANTIZATION_TF);
    gpuQuantizer.setEncodings({encoding});
    Blob<GpuDevice<float>> inputBlob(data4.data(), data4.size());
    std::vector<float> gpuOutput(data4.size(), 0);
    Blob<GpuDevice<float>> outputBlob(gpuOutput.data(), gpuOutput.size());
    gpuQuantizer.quantizeDequantize(inputBlob.getDataPtrOnDevice(), outputBlob.getDataPtrOnDevice(), tensorShape, true);
    const float* gpuResult = outputBlob.getDataPtrOnCpu();

    for (size_t i = 0; i < data4.size(); i++)
    {
        EXPECT_EQ(cpuOutput[i], gpuResult[i]) << "index " << i << " input " << data4[i];
    }
}

// Per-channel FP8 exercises broadcast scale selection.
TEST_F(TestTensorQuantizer, Fp8PerChannelQdqCpuGpuBitExact)
{
    const TensorDims tensorShape = {2, 3, 2, 2};
    const TensorDims blockShape  = {1, 3, 1, 1};

    Encodings encodings;
    for (int channel = 0; channel < 3; channel++)
    {
        TfEncoding encoding;
        encoding.bw     = 8;
        encoding.delta  = 0.25 * (channel + 1);
        encoding.offset = 0;
        encoding.min    = -448.0 * encoding.delta;
        encoding.max    = 448.0 * encoding.delta;
        encodings.push_back(encoding);
    }

    BlockTensorQuantizer cpuQuantizer(blockShape, QuantizationType::Fp8E4M3FN(), QUANTIZATION_TF);
    cpuQuantizer.setEncodings(encodings);
    std::vector<float> cpuOutput(data1.size(), 0);
    cpuQuantizer.quantizeDequantize(data1.data(), cpuOutput.data(), tensorShape, false);

    BlockTensorQuantizer gpuQuantizer(blockShape, QuantizationType::Fp8E4M3FN(), QUANTIZATION_TF);
    gpuQuantizer.setEncodings(encodings);
    Blob<GpuDevice<float>> inputBlob(data1.data(), data1.size());
    std::vector<float> gpuOutput(data1.size(), 0);
    Blob<GpuDevice<float>> outputBlob(gpuOutput.data(), gpuOutput.size());
    gpuQuantizer.quantizeDequantize(inputBlob.getDataPtrOnDevice(), outputBlob.getDataPtrOnDevice(), tensorShape, true);
    const float* gpuResult = outputBlob.getDataPtrOnCpu();

    for (size_t i = 0; i < data1.size(); i++)
    {
        EXPECT_EQ(cpuOutput[i], gpuResult[i]) << "index " << i << " input " << data1[i];
    }
}


// E5M2 exercises a different exponent and mantissa layout.
TEST_F(TestTensorQuantizer, Fp8E5M2PerTensorQdqCpuGpuBitExact)
{
    const TensorDims tensorShape = {(TensorDim) data4.size()};

    TfEncoding encoding;
    encoding.bw     = 8;
    encoding.delta  = 0.5;
    encoding.offset = 0;
    encoding.min    = -57344.0 * encoding.delta;
    encoding.max    = 57344.0 * encoding.delta;

    BlockTensorQuantizer cpuQuantizer({}, QuantizationType::Fp8E5M2(), QUANTIZATION_TF);
    cpuQuantizer.setEncodings({encoding});
    std::vector<float> cpuOutput(data4.size(), 0);
    cpuQuantizer.quantizeDequantize(data4.data(), cpuOutput.data(), tensorShape, false);

    BlockTensorQuantizer gpuQuantizer({}, QuantizationType::Fp8E5M2(), QUANTIZATION_TF);
    gpuQuantizer.setEncodings({encoding});
    Blob<GpuDevice<float>> inputBlob(data4.data(), data4.size());
    std::vector<float> gpuOutput(data4.size(), 0);
    Blob<GpuDevice<float>> outputBlob(gpuOutput.data(), gpuOutput.size());
    gpuQuantizer.quantizeDequantize(inputBlob.getDataPtrOnDevice(), outputBlob.getDataPtrOnDevice(), tensorShape, true);
    const float* gpuResult = outputBlob.getDataPtrOnCpu();

    for (size_t i = 0; i < data4.size(); i++)
    {
        EXPECT_EQ(cpuOutput[i], gpuResult[i]) << "index " << i << " input " << data4[i];
    }
}


TEST_F(TestTensorQuantizer, Fp8E5M2PerChannelQdqCpuGpuBitExact)
{
    const TensorDims tensorShape = {2, 3, 2, 2};
    const TensorDims blockShape  = {1, 3, 1, 1};

    Encodings encodings;
    for (int channel = 0; channel < 3; channel++)
    {
        TfEncoding encoding;
        encoding.bw     = 8;
        encoding.delta  = 0.25 * (channel + 1);
        encoding.offset = 0;
        encoding.min    = -57344.0 * encoding.delta;
        encoding.max    = 57344.0 * encoding.delta;
        encodings.push_back(encoding);
    }

    BlockTensorQuantizer cpuQuantizer(blockShape, QuantizationType::Fp8E5M2(), QUANTIZATION_TF);
    cpuQuantizer.setEncodings(encodings);
    std::vector<float> cpuOutput(data1.size(), 0);
    cpuQuantizer.quantizeDequantize(data1.data(), cpuOutput.data(), tensorShape, false);

    BlockTensorQuantizer gpuQuantizer(blockShape, QuantizationType::Fp8E5M2(), QUANTIZATION_TF);
    gpuQuantizer.setEncodings(encodings);
    Blob<GpuDevice<float>> inputBlob(data1.data(), data1.size());
    std::vector<float> gpuOutput(data1.size(), 0);
    Blob<GpuDevice<float>> outputBlob(gpuOutput.data(), gpuOutput.size());
    gpuQuantizer.quantizeDequantize(inputBlob.getDataPtrOnDevice(), outputBlob.getDataPtrOnDevice(), tensorShape, true);
    const float* gpuResult = outputBlob.getDataPtrOnCpu();

    for (size_t i = 0; i < data1.size(); i++)
    {
        EXPECT_EQ(cpuOutput[i], gpuResult[i]) << "index " << i << " input " << data1[i];
    }
}


TEST_F(TestTensorQuantizer, Fp8QdqCpuGpuBitExactAtEdgeCases)
{
    const std::vector<QuantizationType> qtypes = {
        QuantizationType::Fp8E4M3FN(),
        QuantizationType::Fp8E5M2(),
    };

    for (const auto& qtype: qtypes)
    {
        const auto& fp8Spec           = qtype.floatSpec();
        const float maxValue          = static_cast<float>(fp8Spec.maxValue);
        const float scale             = 0.3f;
        const float stepAtOne         = std::ldexp(1.0f, -fp8Spec.mantissaBits);
        const float smallestSubnormal = std::ldexp(1.0f, fp8Spec.exponentMin - fp8Spec.mantissaBits);
        std::vector<float> input = {
            -std::numeric_limits<float>::infinity(),
            -2.0f * maxValue * scale,
            -maxValue * scale,
            -(1.0f + 1.5f * stepAtOne) * scale,
            -(1.0f + 0.5f * stepAtOne) * scale,
            -0.5f * smallestSubnormal * scale,
            -0.0f,
            0.0f,
            0.5f * smallestSubnormal * scale,
            (1.0f + 0.5f * stepAtOne) * scale,
            (1.0f + 1.5f * stepAtOne) * scale,
            maxValue * scale,
            2.0f * maxValue * scale,
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(),
        };
        const TensorDims tensorShape = {static_cast<TensorDim>(input.size())};

        TfEncoding encoding;
        encoding.bw     = 8;
        encoding.delta  = scale;
        encoding.offset = 0;
        encoding.min    = -maxValue * scale;
        encoding.max    = maxValue * scale;

        BlockTensorQuantizer cpuQuantizer({}, qtype, QUANTIZATION_TF);
        cpuQuantizer.setEncodings({encoding});
        std::vector<float> cpuOutput(input.size(), 0);
        cpuQuantizer.quantizeDequantize(input.data(), cpuOutput.data(), tensorShape, false);

        BlockTensorQuantizer gpuQuantizer({}, qtype, QUANTIZATION_TF);
        gpuQuantizer.setEncodings({encoding});
        Blob<GpuDevice<float>> inputBlob(input.data(), input.size());
        std::vector<float> gpuOutput(input.size(), 0);
        Blob<GpuDevice<float>> outputBlob(gpuOutput.data(), gpuOutput.size());
        gpuQuantizer.quantizeDequantize(inputBlob.getDataPtrOnDevice(), outputBlob.getDataPtrOnDevice(), tensorShape,
                                        true);
        const float* gpuResult = outputBlob.getDataPtrOnCpu();

        for (size_t i = 0; i < input.size(); i++)
        {
            if (std::isnan(cpuOutput[i]))
            {
                EXPECT_TRUE(std::isnan(gpuResult[i])) << "index " << i;
            }
            else
            {
                EXPECT_EQ(cpuOutput[i], gpuResult[i]) << "index " << i << " input " << input[i];
                if (cpuOutput[i] == 0.0f)
                {
                    EXPECT_EQ(std::signbit(cpuOutput[i]), std::signbit(gpuResult[i])) << "index " << i;
                }
            }
        }
    }
}


TEST_F(TestTensorQuantizer, SanityTestGpuBlocked)
{
    BlockTensorQuantizer tensorQuantizer({}, 8, QuantizationMode::QUANTIZATION_TF_ENHANCED);

    float mean   = 2;
    float stddev = 2;
    std::normal_distribution<float> distribution(mean, stddev);
    std::mt19937 generator(1);

    int tensorCount = 6000;
    std::vector<float> statsTensor(tensorCount);
    TensorDims tensorShape = {tensorCount};

    for (unsigned int i = 0; i < tensorCount; i++)
    {
        statsTensor[i] = distribution(generator);
    }
    Blob<GpuDevice<float>> statsTensorBlob(statsTensor.data(), tensorCount);

    tensorQuantizer.setStrictSymmetric(false);
    tensorQuantizer.setUnsignedSymmetric(false);
    tensorQuantizer.updateStats(statsTensorBlob.getDataPtrOnDevice(), tensorShape, true);
    EXPECT_FALSE(tensorQuantizer.isEncodingValid);
    auto encodings = tensorQuantizer.computeEncodings(false);
    tensorQuantizer.setEncodings(encodings);
    TfEncoding encoding = encodings[0];
    EXPECT_TRUE(tensorQuantizer.isEncodingValid);

    std::vector<float> inputTensor(tensorCount, 5);
    Blob<GpuDevice<float>> inputTensorBlob(inputTensor.data(), tensorCount);

    std::vector<float> quantizedTensor(tensorCount, 0);
    Blob<GpuDevice<float>> quantTensorBlob(quantizedTensor.data(), tensorCount);

    tensorQuantizer.quantizeDequantize(inputTensorBlob.getDataPtrOnDevice(), quantTensorBlob.getDataPtrOnDevice(),
                                       tensorShape, true);

    double MAX = *std::max_element(statsTensor.begin(), statsTensor.end());
    double MIN = *std::min_element(statsTensor.begin(), statsTensor.end());
    size_t PDF_SIZE = 512 * 3;
    double HISTOGRAM_BUCKET_SIZE = 3 * (MAX - MIN) / PDF_SIZE;
    EXPECT_NEAR(encoding.min, MIN, HISTOGRAM_BUCKET_SIZE + 0.5 * encoding.delta);
    EXPECT_NEAR(encoding.max, MAX, HISTOGRAM_BUCKET_SIZE + 0.5 * encoding.delta);

    EXPECT_NE(inputTensorBlob.getDataPtrOnCpu()[0], quantTensorBlob.getDataPtrOnCpu()[0]);
    EXPECT_NEAR(quantTensorBlob.getDataPtrOnCpu()[0], 5.0162, HISTOGRAM_BUCKET_SIZE);
}

#endif

template <typename TypeParam>
class TestBlockQuantizerCpuGpu : public ::testing::Test
{};

TYPED_TEST_SUITE(TestBlockQuantizerCpuGpu, TestDeviceTypes);

TYPED_TEST(TestBlockQuantizerCpuGpu, TestBlockQuantizerPerTensorQdq)
{
    if (!CheckRunTest<TypeParam>())
        return;

    typedef typename TypeParam::dataType DataType;

    BlockTensorQuantizer tensorQuantizer({}, 8, QUANTIZATION_TF);
    Encodings encodings(1);
    float min = -5;
    float max = 1;
    float delta = 6 / 255.;
    float offset = min / delta;
    int bw = 8;
    encodings[0] = {min, max, delta, offset, bw};
    tensorQuantizer.setEncodings(encodings);

    float inputF32[6] = {-7, -5, -3, 0, .1, 2.5};
    DataType input[6];
    DataType output[6];
    for (int i = 0; i < 6; i++)
        input[i] = DataType(inputF32[i]);

    Blob<TypeParam> inputBlob(input, 6);
    Blob<TypeParam> outputBlob(output, 6);
    bool useCuda = TypeParam::modeCpuGpu == COMP_MODE_GPU;
    tensorQuantizer.quantizeDequantize(inputBlob.getDataPtrOnDevice(), outputBlob.getDataPtrOnDevice(),
                                       {6, 1}, useCuda);

    for (int i = 0; i < 6; i++)
    {
        float inp = float(input[i]);
        float clipped = std::max(std::min(inp, max), min);
        DataType expected = DataType((std::round(clipped / delta - offset) + offset) * delta);
        EXPECT_NEAR(outputBlob.getDataPtrOnCpu()[i], expected, 0.001);
    }
}

TYPED_TEST(TestBlockQuantizerCpuGpu, TestBlockQuantizationEndToEnd)
{
    if (!CheckRunTest<TypeParam>())
        return;

    typedef typename TypeParam::dataType DataType;

    bool symmetric = true;
    constexpr int numElements = 12;
    TensorDims inputShape = {2, 6};
    TensorDims quantizerShape = {2, 2};
    BlockTensorQuantizer tensorQuantizer(quantizerShape, 8, QUANTIZATION_TF);

    float inF32[numElements] = {
        -5.4f, 10.f, -2.f,
        3.5f, 23.1f, 2.f,
        -10.f, -2.f, -1.f,
        -.1f, 0.3f, 0.1f
    };
    DataType in[numElements];
    DataType out[numElements];
    for (int i = 0; i < numElements; i++)
        in[i] = DataType(inF32[i]);

    Blob<TypeParam> inputBlob(in, numElements);
    Blob<TypeParam> outputBlob(out, numElements);
    bool useCuda = TypeParam::modeCpuGpu == COMP_MODE_GPU;

    tensorQuantizer.updateStats(inputBlob.getDataPtrOnDevice(), inputShape, useCuda);

    auto encodings = tensorQuantizer.computeEncodings(symmetric);
    tensorQuantizer.setEncodings(encodings);

    // Expected max values go through DataType cast for fp16 representability
    float expectedMax[4] = {float(DataType(10.f)), float(DataType(23.1f)), float(float(DataType(10.f)) * 127.f / 128.f),
                            float(DataType(.3f))};
    for (size_t i = 0; i < 4; i++)
    {
        auto enc = encodings[i];
        EXPECT_NEAR(enc.max, expectedMax[i], 0.001);
        EXPECT_NEAR(enc.min + encodings[i].max, -1 * encodings[i].delta, 0.001);
        EXPECT_EQ(enc.offset, -128);
        EXPECT_NEAR(enc.delta, enc.max / 127, 0.001);
    }

    tensorQuantizer.quantizeDequantize(inputBlob.getDataPtrOnDevice(), outputBlob.getDataPtrOnDevice(),
                                       inputShape, useCuda);

    for (int i = 0; i < numElements; i++)
    {
        auto enc = encodings[i / 3];
        float min = enc.min; float max = enc.max; float delta = enc.delta; float offset = enc.offset;
        float inp = float(in[i]);
        float clipped = std::max(std::min(inp, max), min);
        DataType expected = DataType((std::round(clipped / delta - offset) + offset) * delta);
        EXPECT_NEAR(outputBlob.getDataPtrOnCpu()[i], expected, 0.001);

    }

    EXPECT_THROW(tensorQuantizer.setPercentileValue(90.), std::runtime_error);
    EXPECT_THROW(tensorQuantizer.getPercentileValue(), std::runtime_error);
    EXPECT_THROW(tensorQuantizer.getStatsHistogram(), std::runtime_error);

}


TYPED_TEST(TestBlockQuantizerCpuGpu, TestQuantizerZeroPointShift)
{
    if (!CheckRunTest<TypeParam>())
        return;

    typedef typename TypeParam::dataType DataType;

    bool symmetric = true;
    constexpr int numElements = 12;
    TensorDims inputShape = {2, 6};
    TensorDims quantizerShape = {2, 1};
    BlockTensorQuantizer tensorQuantizer(quantizerShape, 2, QUANTIZATION_TF);
    tensorQuantizer.setZeroPointShift(0.5);

    float inF32[numElements] = {
        -3.f, 0.1f, -2.1f, 1.3f, 1.8f, 2.5f,
        -5.1f, -4.1f, -0.1f, 1.3f, 1.8f, 6.f
    };
    float expectedOutF32[numElements] = {
        -3.f, 1.f, -3.f, 1.f, 1.f, 3.f,
        -6.f, -6.f, -2.f, 2.f, 2.f, 6.f
    };
    DataType in[numElements];
    DataType out[numElements];
    for (int i = 0; i < numElements; i++)
        in[i] = DataType(inF32[i]);

    Blob<TypeParam> inputBlob(in, numElements);
    Blob<TypeParam> outputBlob(out, numElements);
    bool useCuda = TypeParam::modeCpuGpu == COMP_MODE_GPU;

    tensorQuantizer.updateStats(inputBlob.getDataPtrOnDevice(), inputShape, useCuda);

    auto encodings = tensorQuantizer.computeEncodings(symmetric);
    tensorQuantizer.setEncodings(encodings);

    float expectedMax[2] = {float(DataType(3.0f)), float(DataType(6.0f))};
    for (size_t i = 0; i < 2; i++)
    {
        auto enc = encodings[i];
        EXPECT_NEAR(enc.max, expectedMax[i], 0.001);
        EXPECT_NEAR(enc.min, -encodings[i].max, 0.001);
        EXPECT_EQ(enc.offset, -1.5);
        EXPECT_NEAR(enc.delta, (enc.max - enc.min) / 3, 0.001);
    }

    tensorQuantizer.quantizeDequantize(inputBlob.getDataPtrOnDevice(), outputBlob.getDataPtrOnDevice(),
                                       inputShape, useCuda);

    for (int i = 0; i < numElements; i++)
    {
        EXPECT_NEAR(outputBlob.getDataPtrOnCpu()[i], DataType(expectedOutF32[i]), 0.0001);
    }

    EXPECT_THROW(tensorQuantizer.computeEncodings(false), std::runtime_error);
}


TYPED_TEST(TestBlockQuantizerCpuGpu, TestZeroPointShiftTFEError)
{
    if (!CheckRunTest<TypeParam>())
        return;

    typedef typename TypeParam::dataType DataType;

    if constexpr (std::is_same_v<DataType, float>)
    {
        bool symmetric = true;
        constexpr int numElements = 12;
        TensorDims inputShape = {2, 6};
        TensorDims quantizerShape = {2, 1};
        BlockTensorQuantizer tensorQuantizer(quantizerShape, 2, QUANTIZATION_TF_ENHANCED);
        tensorQuantizer.setZeroPointShift(0.5);

        DataType in[numElements];
        DataType out[numElements];

        Blob<TypeParam> inputBlob(in, numElements);
        bool useCuda = TypeParam::modeCpuGpu == COMP_MODE_GPU;
        tensorQuantizer.updateStats(inputBlob.getDataPtrOnDevice(), inputShape, useCuda);
        EXPECT_THROW(tensorQuantizer.computeEncodings(symmetric), std::runtime_error);
    }
}
