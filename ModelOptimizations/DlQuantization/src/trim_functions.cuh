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

double randUniformDevice(int seed)
{
    curandState state;
    curand_init(static_cast<unsigned long long>(clock()) + seed, 0, 0, &state);
    return curand_uniform_double(&state);
}

/**
 * @brief Quantize a floating point number to fixed point.
 * @param in Pointer to the floating point number to be quantized.
 * @param seed This number is solely used to generate random numbers in
 * stochastic rounding mode.
 * @param encoding The fixed point format.
 * @param out Compute the result of quantization.
 * @param rounding_mode The rounding mode to use for quantization to fixed
 * point.
 */
template <typename DTYPE>
__device__ void quantizeToFxpDevice(const DTYPE* in, int seed, TfEncoding encoding, DTYPE* out,
                                    RoundingMode rounding_mode)
{
    // Saturate
    *out = (DTYPE) fmax(fmin((double) *in, encoding.max), encoding.min);
    // Scale and add offset to get something in the range [0,2^bw-1]
    *out = *out / encoding.delta - encoding.offset;
    // Round
    switch (rounding_mode)
    {
    case ROUND_NEAREST:
    {
        *out = roundf(*out);
        break;
    }
    case ROUND_STOCHASTIC:
    {
        *out = __float2int_rd(*out + randUniformDevice(seed));
        break;
    }
    default:
    {
        break;
    }
    }
}

/**
 * @brief Quantize a floating point number to fixed point.
 * @param in Pointer to the floating point number to be quantized.
 * @param seed This number is solely used to generate random numbers in
 * stochastic rounding mode.
 * @param encoding The fixed point format.
 * @param out Compute the result of quantization.
 * @param rounding_mode The rounding mode to use for quantization to fixed
 * point.
 */
template <typename IN_DTYPE, typename OUT_DTYPE>
__device__ void quantizeToFxpDeviceWithInt(const IN_DTYPE* in, int seed, TfEncoding encoding, OUT_DTYPE* out,
                                           RoundingMode rounding_mode)
{
    // Saturate
    double out_float = fmax(fmin((double) *in, encoding.max), encoding.min);
    // Scale and add offset to get something in the range [0,2^bw-1]
    out_float = out_float / encoding.delta - encoding.offset;
    // Round
    switch (rounding_mode)
    {
    case ROUND_NEAREST:
    {
        *out = 	__float2ll_rn(out_float);
        break;
    }
    case ROUND_STOCHASTIC:
    {
        *out = __float2ll_rn(out_float + randUniformDevice(seed));
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
 * @param encoding The fixed point format.
 * @param out Compute the result of dequantization.
 */
template <typename DTYPE>
__device__ void dequantizeFromFxpDevice(TfEncoding encoding, DTYPE* out)
{
    // De-quantize
    *out = encoding.delta * (*out + encoding.offset);
}

}   // end of namespace DlQuantization
