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

// This test class only makes sense with GPU support.
#ifdef GPU_QUANTIZATION_ENABLED

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gtest/gtest.h"

#include "DlQuantization/QuantizerFactory.hpp"
#include "SaturatingAdder.hpp"
#include "cuda_util.hpp"
#include "math_functions.hpp"

using namespace std;
using namespace DlQuantization;

/**
 * Test the CUDA fixed point GEMM routine GemmPerform_gpu().
 */
template <typename TypeParam>
class TestGemmFxpAcc : public ::testing::Test
{
protected:
    /**
     * @brief Matrix multiplication test helper.
     * @param m, n, k: The dimension of matrices A, B, and C.
     * @param useDeterministicSeed Use a fix seed to get a deterministic result.
     * @param minRandomData, maxRandomData: The test matrices A and B will get
     * initialized with random data in this interval.
     * @param satAdder Whether to quantize intermediate results or not.
     * @param encoding The fixed point format of the accumulator.
     * @param transposeB Transpose matrix B for matrix matrix multiplication.
     * @param printMatrices Print the matrices for logging purpose.
     * @param elapsedTimeSeconds Return the duration of the matrix multiplication.
     * @param cudaCallsOk Return true if all CUDA calls succeeded.
     * @param maxError Return the maximum absolute error between matrix elements.
     * @param mse Return the mean square error when comparing all matrix elements.
     */
    void MatMultTest(int m, int n, int k, bool useDeterministicSeed, float minRandomData, float maxRandomData,
                     bool satAdder, TfEncoding encoding, bool transposeB, bool printMatrices, float& elapsedTimeSeconds,
                     bool& cudaCallsOk, float& maxError, float& mse)
    {
        // Keep track if all CUDA calls succeed.
        cudaCallsOk = true;

        // Create device matrices
        float *d_A, *d_B, *d_C_ours, *d_C_reference;
        d_A           = (float*) MemoryAllocation(COMP_MODE_GPU, m * k * sizeof(float));
        d_B           = (float*) MemoryAllocation(COMP_MODE_GPU, k * n * sizeof(float));
        d_C_ours      = (float*) MemoryAllocation(COMP_MODE_GPU, m * n * sizeof(float));
        d_C_reference = (float*) MemoryAllocation(COMP_MODE_GPU, m * n * sizeof(float));

        // Create host matrices
        float* h_A           = (float*) malloc(m * k * sizeof(float));
        float* h_B           = (float*) malloc(k * n * sizeof(float));
        float* h_C_ours      = (float*) malloc(m * n * sizeof(float));
        float* h_C_reference = (float*) malloc(m * n * sizeof(float));

        // Fill host matrices with random data in range
        // [minRandomData, maxRandomData].
        if (useDeterministicSeed)
        {
            // Use a fix seed so we know what data will be in matrix A and B.
            srand(987);
        }
        else
        {
            srand(time(NULL));
        }
        for (int i = 0; i < m * k; ++i)
        {
            h_A[i] = rand() * (maxRandomData - minRandomData) / RAND_MAX - minRandomData;
        }
        for (int i = 0; i < k * n; ++i)
        {
            h_B[i] = rand() * (maxRandomData - minRandomData) / RAND_MAX - minRandomData;
        }

        // Copy matrix A and B to device.
        cudaCallsOk &= CudaMemCpy(d_A, h_A, m * k * sizeof(float), CudaMemcpyDirection::HOST_TO_DEVICE);
        cudaCallsOk &= CudaMemCpy(d_B, h_B, k * n * sizeof(float), CudaMemcpyDirection::HOST_TO_DEVICE);

        // Do reference matrix multiplication using cuBLAS.
        cudaCallsOk &= GemmFloat_gpu(m, n, k, d_A, d_B, d_C_reference, transposeB);
        // Copy the result to host.
        cudaCallsOk &=
            CudaMemCpy(h_C_reference, d_C_reference, m * n * sizeof(float), CudaMemcpyDirection::DEVICE_TO_HOST);

        // Start clock.
        struct timespec start;
        clock_gettime(CLOCK_REALTIME, &start);

        // Do matrix multiplication with our self-made GEMM routine.
        GemmPerform_gpu(satAdder, encoding, transposeB, m, n, k, d_A, d_B, d_C_ours);
        // Make sure kernels have finished.
        cudaCallsOk &= CudaSynchronize();

        // Stop clock.
        struct timespec stop;
        clock_gettime(CLOCK_REALTIME, &stop);
        elapsedTimeSeconds = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) / 1e9f;

        // Copy the result to host.
        cudaCallsOk &= CudaMemCpy(h_C_ours, d_C_ours, m * n * sizeof(float), CudaMemcpyDirection::DEVICE_TO_HOST);

        // Print the matrices, if requested.
        /*if (printMatrices && m <= 100 && n <= 100 && k <= 100) {
          printf("\nMatrix A:\n");
          for (int mm = 0; mm < m; ++mm) {
            for (int kk = 0; kk < k; ++kk) {
              printf("%f ", h_A[kk + mm * k]);
            }
            printf("\n");
          }
          printf("\nMatrix B:\n");
          for (int kk = 0; kk < k; ++kk) {
            for (int nn = 0; nn < n; ++nn) {
              printf("%f ", h_B[nn + kk * n]);
            }
            printf("\n");
          }
          printf("\nMatrix C: cuBLAS float / our fxp GEMM:\n");
          for (int mm = 0; mm < m; ++mm) {
            for (int nn = 0; nn < n; ++nn) {
              printf("%2.2f/%2.2f ", h_C_reference[mm * n + nn], h_C_ours[mm * n + nn]);
            }
            printf("\n");
          }
        }*/
        (void) printMatrices;

        // Compare reference result to our result: compute maximum absolute error
        // and average squared error.
        maxError = mse = 0;
        for (int i = 0; i < m * n; ++i)
        {
            maxError = std::max(maxError, std::abs(h_C_reference[i] - h_C_ours[i]));
            mse += pow(h_C_reference[i] - h_C_ours[i], 2);
        }
        mse /= m * n;

        // Free all CPU memory
        free(h_A);
        free(h_B);
        free(h_C_ours);
        free(h_C_reference);

        // Free all GPU memory
        MemoryFree(COMP_MODE_GPU, d_A);
        MemoryFree(COMP_MODE_GPU, d_B);
        MemoryFree(COMP_MODE_GPU, d_C_ours);
        MemoryFree(COMP_MODE_GPU, d_C_reference);
    }

    void ComputeDeltaOffset(int bw, double& min, double& max, double& delta, double& offset)
    {
        vector<string> layer_names;
        vector<int> bw_activations;
        unique_ptr<IQuantizer<float>> iq = unique_ptr<IQuantizer<float>>(
            GetQuantizerInstance<float>(layer_names, COMP_MODE_CPU, bw_activations, QUANTIZATION_TF));
        iq->ComputeDeltaAndOffset(bw, min, max, delta, offset);
    }

    void PrintEncoding(TfEncoding tf)
    {
        printf("Encoding: min: %f, max: %f, delta: %f, offset: %f, bw: %d\n", tf.min, tf.max, tf.delta, tf.offset,
               tf.bw);
    }
};

TYPED_TEST_CASE(TestGemmFxpAcc, ::testing::Types<float>);

// Do full precision matrix multiplication with our own GEMM and cuBLAS,
// compare results.
TYPED_TEST(TestGemmFxpAcc, SANITY_SmallMatrixFloat)
{
    // Define matrix dimensions: <=100
    srand(time(NULL));
    int m                     = std::max(2, rand() % 100);
    int n                     = std::max(2, rand() % 100);
    int k                     = std::max(2, rand() % 100);
    bool useDeterministicSeed = false;
    float minRandomData       = 0;
    float maxRandomData       = 1;
    bool satAdder             = false;
    TfEncoding encoding;
    bool transposeB    = false;
    bool printMatrices = false;
    float elapsedTimeSeconds;
    bool cudaCallsOk;
    float maxError;
    float mse;

    // Perform matrix multiplication test.
    this->MatMultTest(m, n, k, useDeterministicSeed, minRandomData, maxRandomData, satAdder, encoding, transposeB,
                      printMatrices, elapsedTimeSeconds, cudaCallsOk, maxError, mse);

    // Check results.
    EXPECT_TRUE(cudaCallsOk);
    float maxErrorExpected = 1e-4;
    EXPECT_LT(maxError, maxErrorExpected);
}

// Do full precision matrix multiplication with our own GEMM and cuBLAS,
// compare results.
// Matrix B is transposed.
TYPED_TEST(TestGemmFxpAcc, SANITY_SmallMatrixFloatTranspose)
{
    // Define matrix dimensions: <=100
    srand(time(NULL));
    int m                     = std::max(1, rand() % 100);
    int n                     = std::max(1, rand() % 100);
    int k                     = std::max(1, rand() % 100);
    bool useDeterministicSeed = false;
    float minRandomData       = 0;
    float maxRandomData       = 1;
    bool satAdder             = false;
    TfEncoding encoding;
    bool transposeB    = true;
    bool printMatrices = false;
    float elapsedTimeSeconds;
    bool cudaCallsOk;
    float maxError;
    float mse;

    // Perform matrix multiplication test.
    this->MatMultTest(m, n, k, useDeterministicSeed, minRandomData, maxRandomData, satAdder, encoding, transposeB,
                      printMatrices, elapsedTimeSeconds, cudaCallsOk, maxError, mse);

    // Check results.
    EXPECT_TRUE(cudaCallsOk);
    float maxErrorExpected = 1e-4;
    EXPECT_LT(maxError, maxErrorExpected);
}

// Do fixed point matrix multiplication with our own GEMM and float matrix
// multiplication with cuBLAS, compare results.
// Use large matrices.
TYPED_TEST(TestGemmFxpAcc, PERFORMANCE_LargeMatrixFxp)
{
    // Define matrix dimensions: [1024, 2048]
    srand(time(NULL));
    int m                     = rand() % 1024 + 1024;
    int n                     = rand() % 1024 + 1024;
    int k                     = rand() % 1024 + 1024;
    bool useDeterministicSeed = true;
    float minRandomData       = 0;
    float maxRandomData       = 1;
    bool satAdder             = true;
    int bw                    = 16;
    double min, max, delta, offset;
    min = 0;
    max = 2048;
    this->ComputeDeltaOffset(bw, min, max, delta, offset);
    TfEncoding encoding {min, max, delta, offset, bw};
    bool transposeB    = false;
    bool printMatrices = true;
    float elapsedTimeSeconds;
    bool cudaCallsOk;
    float maxError;
    float mse;

    // Perform matrix multiplication test.
    this->MatMultTest(m, n, k, useDeterministicSeed, minRandomData, maxRandomData, satAdder, encoding, transposeB,
                      printMatrices, elapsedTimeSeconds, cudaCallsOk, maxError, mse);

    // Check results.
    EXPECT_TRUE(cudaCallsOk);
    float maxErrorExpected = 10;
    EXPECT_LT(maxError, maxErrorExpected);
    printf("Elapsed time for GEMM with M/N/K=%d/%d/%d: %f sec\n", m, n, k, elapsedTimeSeconds);
}

// Do fixed point matrix multiplication with our own GEMM and float matrix
// multiplication with cuBLAS, compare results.
// Error should be in reasonable tolerance. Print the small matrices.
TYPED_TEST(TestGemmFxpAcc, SANITY_SmallMatrixFxp)
{
    // Define matrix dimensions.
    srand(time(NULL));
    int m = 4;
    int n = 4;
    int k = 4;
    // Use fix data for matrices A and B, so we can have a better prediction of
    // the quantization error.
    bool useDeterministicSeed = true;
    float minRandomData       = 0;
    float maxRandomData       = 1;
    bool satAdder             = true;
    int bw                    = 8;
    double min, max, delta, offset;
    min = 0;
    max = 4;
    this->ComputeDeltaOffset(bw, min, max, delta, offset);
    TfEncoding encoding {min, max, delta, offset, bw};
    bool transposeB    = false;
    bool printMatrices = true;
    float elapsedTimeSeconds;
    bool cudaCallsOk;
    float maxError;
    float mse;

    // Perform matrix multiplication test.
    this->MatMultTest(m, n, k, useDeterministicSeed, minRandomData, maxRandomData, satAdder, encoding, transposeB,
                      printMatrices, elapsedTimeSeconds, cudaCallsOk, maxError, mse);

    // Check results.
    EXPECT_TRUE(cudaCallsOk);
    // Tests use deterministic data. Experiments show the maximal error is always
    // below 0.03.
    float maxErrorExpected = 0.03;
    EXPECT_LT(maxError, maxErrorExpected);
}

#endif   // GPU_QUANTIZATION_ENABLED
