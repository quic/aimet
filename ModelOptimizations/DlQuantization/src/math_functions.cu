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

}   // End of namespace DlQuantization
