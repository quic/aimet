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

#ifndef UTIL_CUDA_UTIL_H_
#define UTIL_CUDA_UTIL_H_

namespace DlQuantization
{
enum class CudaMemcpyDirection : char
{
    DEVICE_TO_HOST,
    HOST_TO_DEVICE
};

/**
 * @brief Copy memory between host and device.
 * @return True if CUDA call succeeds.
 *
 * The memory allocation has to happen outside of this function.
 */
bool CudaMemCpy(void* dest, const void* src, size_t bytes, CudaMemcpyDirection direction);

/**
 * @brief Find out at runtime if there exists a GPU with CUDA support.
 * @return True if CUDA is supported, false otherwise.
 */
bool CudaSupportedHelper();

/**
 * @brief Make sure all kernels have finished.
 *
 * This allows for accurate timing measurements.
 */
bool CudaSynchronize();

// Always use 512 threads per block
const int CUDA_NUM_THREADS = 512;

// Compute the number of blocks based on the total number of threads.
inline int CUDA_NUM_BLOCKS(const int N)
{
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// Loop over data in kernel (stride: grid)
#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

}   // End of namespace DlQuantization

#endif   // UTIL_CUDA_UTIL_H_
