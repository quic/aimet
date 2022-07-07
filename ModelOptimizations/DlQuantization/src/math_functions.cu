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

#include <cublas_v2.h>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include "cuda_util.hpp"
#include "math_functions.hpp"


namespace DlQuantization
{
template <typename DTYPE>
DTYPE GetMax_gpu(const DTYPE* data, int cnt)
{
    const thrust::device_ptr<const DTYPE> ptr = thrust::device_pointer_cast(data);
    return thrust::reduce(ptr, ptr + cnt, std::numeric_limits<DTYPE>::lowest(), thrust::maximum<DTYPE>());
}

template <typename DTYPE>
DTYPE GetMin_gpu(const DTYPE* data, int cnt)
{
    const thrust::device_ptr<const DTYPE> ptr = thrust::device_pointer_cast(data);
    return thrust::reduce(ptr, ptr + cnt, std::numeric_limits<DTYPE>::max(), thrust::minimum<DTYPE>());
}

__global__ void ElementwiseMult_kernel(const float* in, size_t cnt, float factor, float* out)
{
    CUDA_KERNEL_LOOP(i, cnt)
    {
        out[i] = in[i] * factor;
    }
}

void ElementwiseMult_gpu(const float* in, size_t cnt, float factor, float* out)
{
    ElementwiseMult_kernel<<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS>>>(in, cnt, factor, out);
}

bool GemmFloat_gpu(int M, int N, int K, const float* A, const float* B, float* C, bool transposeB)
{
    cublasHandle_t handle;
    bool success             = (CUBLAS_STATUS_SUCCESS == cublasCreate(&handle));
    const float alpha        = 1;
    const float beta         = 0;
    cublasOperation_t transB = !transposeB ? CUBLAS_OP_N : CUBLAS_OP_T;
    int ldb                  = !transposeB ? N : K;
    // Note that cuBLAS uses column major order, whereas C uses row major order.
    success &=
        (CUBLAS_STATUS_SUCCESS == cublasSgemm(handle, transB, CUBLAS_OP_N, N, M, K, &alpha, B, ldb, A, K, &beta, C, N));
    // cudaDeviceSynchronize();
    return success;
}

void* MemoryAllocation_gpu(size_t bytes)
{
    void* devPtr;
    cudaMalloc(&devPtr, bytes);
    return devPtr;
}

bool MemoryFree_gpu(void* data)
{
    return cudaSuccess == cudaFree(data);
}

// Explicit instantiations
template double GetMax_gpu(const double* data, int cnt);

template float GetMax_gpu(const float* data, int cnt);

template double GetMin_gpu(const double* data, int cnt);

template float GetMin_gpu(const float* data, int cnt);


using pdf_elem_type = float;

template <typename DTYPE>
__global__ void UpdatePdfKernel(const DTYPE* data,
                                pdf_elem_type* pdf_this_iter,
                                const size_t cnt,
                                const DTYPE pdf_offset,
                                const DTYPE bucket_size)
{
    // This offset is used to help map numbers to histogram buckets.
    // Go through all data points and add them to the histogram.
    CUDA_KERNEL_LOOP(i, cnt)
    {
        // Map a floating point number to the appropriate bucket.
        int index = round(data[i] / bucket_size - pdf_offset);
        // Add to histogram, if inside the histogram range.
        if (index >= 0 && index < PDF_SIZE)
        {
            int idx = PDF_SIZE * (blockIdx.x * blockDim.x + threadIdx.x) + index;
            pdf_this_iter[idx] += 1;
        }
    }
}

template <typename DTYPE>
void UpdatePdfSigned_gpu(const DTYPE* data, int cnt, PDF& pdf)
{
    // Check if we need to initialize the PDF
    if (0 == pdf.xLeft.size())
    {
        // Define the range over which we want to calculate the PDF.
        DTYPE min_val = GetMin(data, cnt, COMP_MODE_GPU);
        DTYPE max_val = GetMax(data, cnt, COMP_MODE_GPU);

        if ((min_val == 0) && (max_val == 0))
        {
            // Special case, we don't have a histogram initialized, but we have a zero tensor here
            // No point in trying to initialize the histogram using this
            return;
        }

        // Make sure we have a non-zero range.
        if (min_val == max_val)
        {
            max_val = std::max(max_val, min_val + static_cast<DTYPE>(0.01));
        }
        // Enlarge the range by factor 3, to be on the safe side.
        DTYPE center = (max_val + min_val) / 2;
        min_val      = center - 3 * (center - min_val);
        max_val      = center + 3 * (max_val - center);
        // Initialize the PDF's buckets.
        DTYPE bucket_size = (max_val - min_val) / PDF_SIZE;
        pdf.xLeft.resize(PDF_SIZE);
        for (int i = 0; i < PDF_SIZE; ++i)
        {
            pdf.xLeft[i] = min_val + i * bucket_size;
        }
        // Initialize the rest of the PDF structure.
        pdf.pdf.resize(PDF_SIZE);
        pdf.iterations = 0;
    }

    // Create the histogram of this number distribution.
    // The histogram's range is min_val to max_val.
    DTYPE min_val     = pdf.xLeft[0];
    DTYPE bucket_size = pdf.xLeft[1] - pdf.xLeft[0];

    uint32_t pdf_size = PDF_SIZE * CUDA_NUM_BLOCKS(cnt) * CUDA_NUM_THREADS;
    pdf_elem_type* pdf_this_iter = (pdf_elem_type*) MemoryAllocation_gpu(sizeof(pdf_elem_type) * pdf_size);
    cudaMemset(pdf_this_iter, 0x00, sizeof(pdf_elem_type) * pdf_size);

    // This offset is used to help map numbers to histogram buckets.
    DTYPE pdf_offset = min_val / bucket_size;
    // Go through all data points and add them to the histogram.
    UpdatePdfKernel<<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS>>>(data,
                                                                pdf_this_iter,
                                                                cnt,
                                                                pdf_offset,
                                                                bucket_size);

    // pdf_elem_type* pdf_this_iter_cpu = pdf_this_iter;
    pdf_elem_type* pdf_this_iter_cpu = (pdf_elem_type*) malloc(sizeof(pdf_elem_type) * pdf_size);
    cudaMemcpy(pdf_this_iter_cpu,
               pdf_this_iter,
               sizeof(pdf_elem_type) * pdf_size,
               cudaMemcpyDefault);

    std::vector<pdf_elem_type> pdf_sum(PDF_SIZE, 0);

    for (int i = 0; i < CUDA_NUM_BLOCKS(cnt) * CUDA_NUM_THREADS; i++) {
        for (int j = 0; j < PDF_SIZE; j++) {
            pdf_sum.at(j) += pdf_this_iter_cpu[i * PDF_SIZE + j];
        }
    }

    // Average this histogram into the average of all batches.
    // DTYPE sum = 0.0;
    for (int i = 0; i < PDF_SIZE; ++i)
    {
        // Convert histogram to probability density function.
        pdf_sum[i] /= cnt;
        // Average this PDF into the running average.
        pdf.pdf.at(i) = (pdf.pdf.at(i) * pdf.iterations + pdf_sum[i]) / (pdf.iterations + 1);
        // sum += pdf.pdf.at(i);
    }
    // assert(0.999 < sum);
    // assert(sum < 1.001);
        
    pdf.iterations++;
    MemoryFree_gpu(pdf_this_iter);
}

template <typename DTYPE>
void UpdatePdfUnsigned_gpu(const DTYPE* data, int cnt, PDF& pdf)
{
    // Check if we need to initialize the PDF.
    // if (0 == pdf.xLeft.size())
    // {
    //     // Define the range over which we want to calculate the PDF.
    //     DTYPE min_val = GetMin(data, cnt, COMP_MODE_GPU);
    //     DTYPE max_val = GetMax(data, cnt, COMP_MODE_GPU);

    //     if ((min_val == 0) && (max_val == 0))
    //     {
    //         // Special case, we don't have a histogram initialized, but we have a zero tensor here
    //         // No point in trying to initialize the histogram using this
    //         return;
    //     }

    //     // Make sure we have a non-zero range.
    //     if (min_val == max_val)
    //     {
    //         max_val = std::max(max_val, min_val + static_cast<DTYPE>(0.01));
    //     }
    //     // Enlarge the range by factor 3, to be on the safe side.
    //     DTYPE center = (max_val + min_val) / 2;
    //     min_val      = center - 3 * (center - min_val);
    //     max_val      = center + 3 * (max_val - center);
    //     // Initialize the PDF's buckets.
    //     DTYPE max_abs_val = std::max(std::abs(max_val), std::abs(min_val));
    //     DTYPE bucket_size = max_abs_val / PDF_SIZE;
    //     pdf.xLeft.resize(PDF_SIZE);
    //     for (int i = 0; i < PDF_SIZE; ++i)
    //     {
    //         pdf.xLeft[i] = i * bucket_size;
    //     }
    //     // Initialize the rest of the PDF structure.
    //     pdf.pdf.resize(PDF_SIZE);
    //     pdf.iterations = 0;
    // }

    // // Create the histogram of this number distribution.
    // DTYPE bucket_size = pdf.xLeft[1] - pdf.xLeft[0];

    // DTYPE* pdf_this_iter = (DTYPE*) MemoryAllocation_gpu(sizeof(DTYPE) * PDF_SIZE);
    // cudaMemset(pdf_this_iter, 0, sizeof(DTYPE) * PDF_SIZE);

    // UpdatePdfKernel<<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS>>>(data,
    //                                                             pdf_this_iter,
    //                                                             cnt,
    //                                                             static_cast<DTYPE>(0.0),
    //                                                             bucket_size);

    // DTYPE* pdf_this_iter_cpu = (DTYPE*) malloc(sizeof(DTYPE) * PDF_SIZE);
    // CudaMemCpy(pdf_this_iter_cpu,
    //            pdf_this_iter,
    //            sizeof(DTYPE) * PDF_SIZE,
    //            CudaMemcpyDirection::DEVICE_TO_HOST);

    // // Average this histogram into the average of all batches.
    // for (int i = 0; i < PDF_SIZE; ++i)
    // {
    //     // Convert histogram to probability density function.
    //     pdf_this_iter_cpu[i] /= cnt;
    //     // Average this PDF into the running average.
    //     pdf.pdf[i] = (pdf.pdf[i] * pdf.iterations + pdf_this_iter_cpu[i]) / (pdf.iterations + 1);
    // }
    // pdf.iterations++;
    // MemoryFree_gpu(pdf_this_iter);
    // free(pdf_this_iter_cpu);
}

template void UpdatePdfSigned_gpu(const float* data, int cnt, PDF& pdf);
template void UpdatePdfSigned_gpu(const double* data, int cnt, PDF& pdf);

template void UpdatePdfUnsigned_gpu(const float* data, int cnt, PDF& pdf);
template void UpdatePdfUnsigned_gpu(const double* data, int cnt, PDF& pdf);

}   // End of namespace DlQuantization
