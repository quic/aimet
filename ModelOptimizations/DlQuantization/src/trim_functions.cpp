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


#include <algorithm>
#include <cstdint>
#include <math.h>
#include <stdexcept>
#include <stdlib.h>
#include <thread>
#include <vector>

#include "DlQuantization/Quantization.hpp"
#include "trim_functions.hpp"

namespace DlQuantization
{
using namespace std;

inline double RandUniform_cpu()
{
    return rand() / (RAND_MAX + 1.0);
}

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


template <class Lambda>
Lambda Parallelize(const uint32_t number_of_threads, Lambda lambda)
{
    std::vector<std::thread> threads(number_of_threads);
    for (uint32_t i = 0; i < number_of_threads; ++i)
    {
        threads[i] = std::thread(lambda, i);
    }
    for (uint32_t i = 0; i < number_of_threads; ++i)
    {
        threads[i].join();
    }
    return lambda;
};


// encoding: TF: rounded
template <typename DTYPE>
void QuantizeDequantize(const DTYPE* in, int cnt, const TfEncoding& encoding, DTYPE* out,
                           ComputationMode mode_cpu_gpu, RoundingMode rounding_mode)
{
    switch (mode_cpu_gpu)
    {
    case COMP_MODE_CPU:
        QuantizeDequantize_CPU(in, cnt, encoding, out, rounding_mode);
        break;
    case COMP_MODE_GPU:
#ifdef GPU_QUANTIZATION_ENABLED
        QuantizeDequantize_GPU(in, cnt, encoding, out, rounding_mode);
#else
        throw runtime_error("Not compiled for GPU mode.");
#endif
        break;
    default:
        throw runtime_error("Unknown computation mode.");
        break;
    }
}

// encoding: TF: rounded
template <typename DTYPE>
void QuantizeToFxp(const DTYPE* in, int cnt, const TfEncoding& encoding, DTYPE* out, ComputationMode mode_cpu_gpu,
                   RoundingMode rounding_mode)
{
    switch (mode_cpu_gpu)
    {
        case COMP_MODE_CPU:
            QuantizeToFxp_CPU(in, cnt, encoding, out, rounding_mode);
            break;
        case COMP_MODE_GPU:
#ifdef GPU_QUANTIZATION_ENABLED
            QuantizeToFxp_GPU(in, cnt, encoding, out, rounding_mode);
#else
            throw runtime_error("Not compiled for GPU mode.");
#endif
            break;
        default:
            throw runtime_error("Unknown computation mode.");
            break;
    }
}

// CPU implementations

template <typename DTYPE>
void QuantizeDequantize_CPU(const DTYPE* in, int cnt, const TfEncoding& encoding, DTYPE* out,
                            RoundingMode rounding_mode)
{
    for (int i = 0; i < cnt; ++i)
    {
        QuantizeValue_CPU(&in[i], encoding, &out[i], rounding_mode);
        DequantizeValue_CPU(encoding, &out[i]);
    }
}

template <typename DTYPE>
void QuantizeToFxp_CPU(const DTYPE* in, int cnt, const TfEncoding& encoding, DTYPE* out, RoundingMode rounding_mode)
{
    for (int i = 0; i < cnt; ++i)
    {
        QuantizeValue_CPU(&in[i], encoding, &out[i], rounding_mode);
    }
}

template <typename DTYPE>
inline void QuantizeValue_CPU(const DTYPE* in, const TfEncoding& encoding, DTYPE* out, RoundingMode rounding_mode)
{
    *out = (DTYPE) max(min((double) *in, encoding.max), encoding.min);
    // Scale and add offset to get something in the range [0,2^bw-1]
    *out = round(*out / encoding.delta) - encoding.offset;

    switch (rounding_mode)
    {
        case ROUND_NEAREST:
        {
            break;
        }
        case ROUND_STOCHASTIC:
        {
            *out = floor(*out + RandUniform_cpu());
            break;
        }
        default:
        {
            throw runtime_error("Unknown rounding mode.");
        }
    }
}

template <typename DTYPE>
inline void DequantizeValue_CPU(const TfEncoding& encoding, DTYPE* out)
{
    *out = encoding.delta * (*out + encoding.offset);
}


// Explicit instantiations
template void QuantizeDequantize(const double* in, int cnt, const TfEncoding& encoding, double* out,
                                 ComputationMode mode_cpu_gpu, RoundingMode rounding_mode);

template void QuantizeDequantize(const float* in, int cnt, const TfEncoding& encoding, float* out,
                                 ComputationMode mode_cpu_gpu, RoundingMode rounding_mode);

template void QuantizeToFxp(const double* in, int cnt, const TfEncoding& encoding, double* out,
                            ComputationMode mode_cpu_gpu, RoundingMode rounding_mode);

template void QuantizeToFxp(const float* in, int cnt, const TfEncoding& encoding, float* out,
                            ComputationMode mode_cpu_gpu, RoundingMode rounding_mode);

}   // End of namespace DlQuantization
