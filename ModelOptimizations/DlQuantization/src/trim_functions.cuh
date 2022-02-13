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

#include <curand_kernel.h>

#include "DlQuantization/Quantization.hpp"

namespace DlQuantization
{
// This file contains the definition of quantizeToFxpDevice(): a CUDA kernel
// which we use from different .cu files.

// Returns a random number in (0,1].
// Even though the repetitive initialization of a curand state might look
// suboptimal, the performance is actually nearly the same as when using global
// states.
__device__ __forceinline__

__device__ float rand_uniform(int seed)
{
    curandState state;
    curand_init(static_cast<unsigned long long>(clock()) + seed, 0, 0, &state);
    return curand_uniform(&state);
}

__device__ double rand_uniform_double(int seed)
{
    curandState state;
    curand_init(static_cast<unsigned long long>(clock()) + seed, 0, 0, &state);
    return curand_uniform_double(&state);
}

__device__ inline float clamp(float val, float min, float max)
{
    return fmaxf(fminf(val, max), min);
}

__device__ inline double clamp(double val, double min, double max)
{
    return fmax(fmin(val, max), min);
}

__device__ inline float round_nearest(float val)
{
    return roundf(val);
}

__device__ inline double round_nearest(double val)
{
    return round(val);
}

__device__ inline float round_stochastic(float val, int seed)
{
    return __float2int_rd(val + rand_uniform(seed));
}

__device__ inline double round_stochastic(double val, int seed)
{
    return __double2int_rd(val + rand_uniform_double(seed));
}

/**
 * @brief Quantize a floating point number to fixed point.
 * @param in Pointer to the floating point number to be quantized.
 * @param out Compute the result of quantization.
 * @param encoding_min The minimum value for clipping.
 * @param encoding_max The maximum value for clipping.
 * @param encoding_delta The fixed point scale.
 * @param encoding_offset The fixed point offset.
 * @param rounding_mode The rounding mode to use for quantization to fixed
 * point.
 * @param seed This number is solely used to generate random numbers in
 * stochastic rounding mode.
 */
template <typename DTYPE>
__device__ void quantizeToFxpDevice(const DTYPE* in, DTYPE* out,
                                    DTYPE encoding_min, DTYPE encoding_max,
                                    DTYPE encoding_delta, DTYPE encoding_offset,
                                    RoundingMode rounding_mode, int seed)
{
    // Saturate
    *out = clamp(*in, encoding_min, encoding_max);
    // Scale and add offset to get something in the range [0,2^bw-1]
    *out = *out / encoding_delta - encoding_offset;
    // Round
    switch (rounding_mode)
    {
        case ROUND_NEAREST:
        {
            *out = round_nearest(*out);
            break;
        }
        case ROUND_STOCHASTIC:
        {
            *out = round_stochastic(*out, seed);
            break;
        }
        default:
        {
            break;
        }
    }
}

/**
 * @brief Dequantize a fixed point number to floating point.
 * @param out Compute the result of dequantization.
 * @param encoding_delta The fixed point scale.
 * @param encoding_offset The fixed point offset.
 */
template <typename DTYPE>
__device__ void dequantizeFromFxpDevice(DTYPE* out,
                                        DTYPE encoding_delta,
                                        DTYPE encoding_offset)
{
    // De-quantize
    *out = encoding_delta * (*out + encoding_offset);
}

}   // end of namespace DlQuantization
