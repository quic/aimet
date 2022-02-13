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
#include <cmath>
#include <stdexcept>
#include <stdlib.h>
#include <thread>
#include <vector>

#include "DlQuantization/Quantization.hpp"
#include "trim_functions.hpp"

namespace DlQuantization
{
using namespace std;

template <typename DTYPE>
inline DTYPE randUniformCpu()
{
    return rand() / (RAND_MAX + static_cast<DTYPE>(1.0));
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
Lambda parallelize(const uint32_t number_of_threads, Lambda lambda)
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
void quantizeDequantize(const DTYPE* in, int cnt, const TfEncoding& encoding, DTYPE* out,
                        ComputationMode mode_cpu_gpu, RoundingMode rounding_mode)
{
    switch (mode_cpu_gpu)
    {
    case COMP_MODE_CPU:
        quantizeDequantizeCpu(in, cnt, encoding, out, rounding_mode);
        break;
    case COMP_MODE_GPU:
#ifdef GPU_QUANTIZATION_ENABLED
            quantizeDequantizeGpu(in, cnt, encoding, out, rounding_mode);
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
void quantizeToFxp(const DTYPE* in, int cnt, const TfEncoding& encoding, DTYPE* out, ComputationMode mode_cpu_gpu,
                   RoundingMode rounding_mode, bool shiftToSigned)
{
    switch (mode_cpu_gpu)
    {
        case COMP_MODE_CPU:
            quantizeToFxpCpu(in, cnt, encoding, out, rounding_mode, shiftToSigned);
            break;
        case COMP_MODE_GPU:
#ifdef GPU_QUANTIZATION_ENABLED
            quantizeToFxpGpu(in, cnt, encoding, out, rounding_mode, shiftToSigned);
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
inline void quantizeValueCpu(const DTYPE* in, DTYPE* out,
                             DTYPE encoding_min, DTYPE encoding_max,
                             DTYPE encoding_delta, DTYPE encoding_offset,
                             RoundingMode rounding_mode)
{
    *out = fmax(fmin(*in, encoding_max), encoding_min);
    // Scale and add offset to get something in the range [0,2^bw-1]
    *out = round(*out / encoding_delta) - encoding_offset;

    switch (rounding_mode)
    {
        case ROUND_NEAREST:
        {
            break;
        }
        case ROUND_STOCHASTIC:
        {
            *out = floor(*out + randUniformCpu<DTYPE>());
            break;
        }
        default:
        {
            throw runtime_error("Unknown rounding mode.");
        }
    }
}

template <typename DTYPE>
inline void dequantizeValueCpu(DTYPE* out, DTYPE encoding_delta, DTYPE encoding_offset)
{
    *out = encoding_delta * (*out + encoding_offset);
}

template <typename DTYPE>
void quantizeDequantizeCpu(const DTYPE* in, int cnt, const TfEncoding& encoding, DTYPE* out,
                           RoundingMode rounding_mode)
{
    for (int i = 0; i < cnt; ++i)
    {
        quantizeValueCpu<DTYPE>(&in[i], &out[i],
                                encoding.min, encoding.max,
                                encoding.delta, encoding.offset,
                                rounding_mode);
        dequantizeValueCpu<DTYPE>(&out[i], encoding.delta, encoding.offset);
    }
}

template <typename DTYPE>
void quantizeToFxpCpu(const DTYPE* in, int cnt, const TfEncoding& encoding, DTYPE* out, RoundingMode rounding_mode,
                      bool shiftToSigned)
{
    // Using unsigned int to account for case of signed symmetric 32 bit, when shift will be 2^31
    unsigned int shift = 0;
    if (shiftToSigned) {
        shift = pow(2, encoding.bw - 1);
    }
    for (int i = 0; i < cnt; ++i)
    {
        quantizeValueCpu<DTYPE>(&in[i], &out[i],
                                encoding.min, encoding.max,
                                encoding.delta, encoding.offset,
                                rounding_mode);
        out[i] -= shift;
    }
}


// Explicit instantiations
template void quantizeDequantize(const double* in, int cnt, const TfEncoding& encoding, double* out,
                                 ComputationMode mode_cpu_gpu, RoundingMode rounding_mode);

template void quantizeDequantize(const float* in, int cnt, const TfEncoding& encoding, float* out,
                                 ComputationMode mode_cpu_gpu, RoundingMode rounding_mode);

template void quantizeToFxp(const double* in, int cnt, const TfEncoding& encoding, double* out,
                            ComputationMode mode_cpu_gpu, RoundingMode rounding_mode, bool shiftToSigned);

template void quantizeToFxp(const float* in, int cnt, const TfEncoding& encoding, float* out,
                            ComputationMode mode_cpu_gpu, RoundingMode rounding_mode, bool shiftToSigned);

}   // End of namespace DlQuantization
