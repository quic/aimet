//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2016-2021, Qualcomm Innovation Center, Inc. All rights reserved.
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


#ifndef UTIL_MATH_FUNCTIONS_H_
#define UTIL_MATH_FUNCTIONS_H_

#include <algorithm>
#include <limits>
#include <map>
#include <math.h>
#include <stdexcept>

#include "DlQuantization/Quantization.hpp"

namespace DlQuantization
{
/**
 * @brief A probability density distribution.
 *
 * We use this structure to record the dynamic range of a given number
 * distribution.
 */
struct PDF
{
    // The left sides of the buckets.
    std::vector<double> xLeft;
    // The probability for each bucket.
    std::vector<double> pdf;
    // The histogram for each bucket.
    std::vector<double> hist;
    // This PDF holds the average data for this many iterations.
    int iterations;
};

// Number of buckets in histogram which we use to capture the dynamic range.
const int PDF_SIZE = 512;

// The probability density functions for all input and output blobs of a network
// layer.
struct StatsLayerPdf
{
    std::vector<PDF> in;
    std::vector<PDF> out;
};

template <typename DTYPE>
DTYPE GetMax(const DTYPE* data, int cnt, ComputationMode cpuGpuMode);

template <typename DTYPE>
DTYPE GetMin(const DTYPE* data, int cnt, ComputationMode cpuGpuMode);

// Android compiler doesn't have std::log2, so define it here
double logBase2(double d);

/**
 * @brief Creat a probability density function and average it into the
 * data we have so far.
 * @param data The number distribution for which we create a density function.
 * @param cnt The number of data points.
 * @param mode_cpu_gpu The 'data' buffer is either in CPU or GPU memory.
 * @param signed_vals If true, we create a histogram of the actual values. If
 * set to false, we create a histogram of the absolute values.
 * @param pdf Compute a probability density function. If this PDF already
 * contains values, update the PDF by averaging it with the new number
 * distribution.
 */
template <typename DTYPE>
void UpdatePdf(const DTYPE* data, int cnt, ComputationMode mode_cpu_gpu, bool signed_vals, PDF& pdf);

/**
 * @brief Allocate memory.
 * @param modeCpuGpu Allocate memory for CPU or GPU.
 * @param bytes The size of the memory in bytes.
 *
 * The memory is not initialized.
 */
void* MemoryAllocation(ComputationMode modeCpuGpu, size_t bytes);

/**
 * @brief Release previously allocated memory.
 * @param modeCpuGpu Allocate memory for CPU or GPU.
 * @param data Pointer to memory which was previously allocated with
 * MemoryAllocation().
 */
void MemoryFree(ComputationMode modeCpuGpu, void* data);

/**
 * @brief Multiply all inputs by a scale factor.
 * @param cnt The number of floating point values.
 * @param factor The numbers get scaled by this factor.
 */
void ElementwiseMult(ComputationMode modeCpuGpu, const float* in, size_t cnt, float factor, float* out);

/**
 * @brief Perform a matrix multiplication in floating point.
 */
void GemmFloat(ComputationMode modeCpuGpu, bool transposeB, size_t m, size_t n, size_t k, const float* A,
               const float* B, float* C);

// CPU implementations...
template <typename DTYPE>
DTYPE GetMax_cpu(const DTYPE* data, int cnt);

template <typename DTYPE>
DTYPE GetMin_cpu(const DTYPE* data, int cnt);

void ElementwiseMult_cpu(const float* in, size_t cnt, float factor, float* out);

void* MemoryAllocation_cpu(size_t bytes);

void MemoryFree_cpu(void* data);

template <typename DTYPE>
void UpdatePdfSigned_cpu(const DTYPE* data, int cnt, bool signed_vals, PDF& pdf);

template <typename DTYPE>
void UpdatePdfUnsigned_cpu(const DTYPE* data, int cnt, bool signed_vals, PDF& pdf);

// GPU implementations...
#ifdef GPU_QUANTIZATION_ENABLED

template <typename DTYPE>
DTYPE GetMax_gpu(const DTYPE* data, int cnt);

template <typename DTYPE>
DTYPE GetMin_gpu(const DTYPE* data, int cnt);

void ElementwiseMult_gpu(const float* in, size_t cnt, float factor, float* out);

/**
 * @brief Perform matrix-matrix multiplication using cuBLAS.
 * @return True if CUDA call succeeds.
 */
bool GemmFloat_gpu(int M, int N, int K, const float* A, const float* B, float* C, bool transposeB);

void* MemoryAllocation_gpu(size_t bytes);

bool MemoryFree_gpu(void* data);

#endif   // GPU_QUANTIZATION_ENABLED

}   // End of namespace DlQuantization

#endif   // UTIL_MATH_FUNCTIONS_H_
