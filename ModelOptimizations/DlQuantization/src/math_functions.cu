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


template <typename DTYPE>
__global__ void countKernel(const DTYPE* data,
                            DTYPE* pdf_per_thread,
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
            pdf_per_thread[idx] += 1;
        }
    }
}


template <typename DTYPE>
__global__ void reduceSumKernel(const DTYPE* pdf_per_thread,
                                DTYPE* pdf_this_iter,
                                const size_t cnt,
                                const size_t stride)
{
    if (blockIdx.x == 0 && threadIdx.x < stride)
    {
        for (int i = threadIdx.x; i < cnt; i += stride)
        {
            pdf_this_iter[threadIdx.x] += pdf_per_thread[i];
        }
    }
}


static const int PDF_MAX_BUFF_BYTES = (1 << 25); // 32MB

#define GET_PDF_BUFF_SIZE(tensor_size, DTYPE)\
    sizeof(DTYPE) * CUDA_NUM_BLOCKS(tensor_size) * CUDA_NUM_THREADS * PDF_SIZE < PDF_MAX_BUFF_BYTES ?\
    CUDA_NUM_BLOCKS(tensor_size) :\
    PDF_MAX_BUFF_BYTES / (sizeof(DTYPE) * PDF_SIZE * CUDA_NUM_THREADS)


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

    // Limit the number of thread blocks for performance based on heuristics
    const size_t CUDA_NUM_BLOCKS_ = GET_PDF_BUFF_SIZE(cnt, DTYPE);
    const size_t buff_size = PDF_SIZE * CUDA_NUM_BLOCKS_ * CUDA_NUM_THREADS;

    DTYPE* pdf_per_thread = (DTYPE*) MemoryAllocation_gpu(sizeof(DTYPE) * buff_size);
    cudaMemset(pdf_per_thread, 0x00, sizeof(DTYPE) * buff_size);

    // This offset is used to help map numbers to histogram buckets.
    DTYPE pdf_offset = min_val / bucket_size;
    // Go through all data points and add them to the histogram.
    countKernel<<<CUDA_NUM_BLOCKS_, CUDA_NUM_THREADS>>>(data,
                                                        pdf_per_thread,
                                                        cnt,
                                                        pdf_offset,
                                                        bucket_size);

    DTYPE* pdf_this_iter = (DTYPE*) MemoryAllocation_gpu(sizeof(DTYPE) * PDF_SIZE);
    cudaMemset(pdf_this_iter, 0x00, sizeof(DTYPE) * PDF_SIZE);

    reduceSumKernel<<<1, PDF_SIZE>>>(pdf_per_thread, pdf_this_iter, buff_size, PDF_SIZE);

    DTYPE pdf_this_iter_cpu[PDF_SIZE];
    cudaMemcpy(pdf_this_iter_cpu,
               pdf_this_iter,
               sizeof(DTYPE) * PDF_SIZE,
               cudaMemcpyDefault);

    // Average this histogram into the average of all batches.
    for (int i = 0; i < PDF_SIZE; ++i)
    {
        // Average this PDF into the running average.
        pdf.pdf.at(i) = (pdf.pdf.at(i) * pdf.iterations + pdf_this_iter_cpu[i] / cnt) / (pdf.iterations + 1);
    }
        
    pdf.iterations++;
    MemoryFree_gpu(pdf_this_iter);
    MemoryFree_gpu(pdf_per_thread);
}

template <typename DTYPE>
void UpdatePdfUnsigned_gpu(const DTYPE* data, int cnt, PDF& pdf)
{
    // TODO
}

template void UpdatePdfSigned_gpu(const float* data, int cnt, PDF& pdf);
template void UpdatePdfSigned_gpu(const double* data, int cnt, PDF& pdf);

template void UpdatePdfUnsigned_gpu(const float* data, int cnt, PDF& pdf);
template void UpdatePdfUnsigned_gpu(const double* data, int cnt, PDF& pdf);

}   // End of namespace DlQuantization
