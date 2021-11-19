//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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

#include <math.h>

#include "trim_functions.hpp"
#include "TensorQuantizationSim.h"

namespace DlQuantization
{
template <typename DTYPE>
TensorQuantizationSim<DTYPE>::TensorQuantizationSim()
{
}

template <typename DTYPE>
void TensorQuantizationSim<DTYPE>::gateMinMax(double& encodingMin, double& encodingMax)
{

        double epsilon = 1e-5;
        // Additional handling to retain zero in range
        // encodingMin can be at maximum 0.0
        encodingMin = std::min(encodingMin, 0.0);

        // encodingMax can be at minimum 0.0
        encodingMax = std::max(encodingMax, 0.0);

        // handle case where encodingMin == encodingMax
        encodingMax = std::max(encodingMax, encodingMin + epsilon);
}

template <typename DTYPE>
void TensorQuantizationSim<DTYPE>::_fillQuantizeInfo(TfEncoding& encoding, DlQuantization::ComputationMode& cpuGpuMode,
                                                     uint8_t bw, double encodingMin, double encodingMax, bool use_cuda)
{
    gateMinMax(encodingMin, encodingMax);
    encoding.min = encodingMin;
    encoding.max = encodingMax;
    encoding.bw = bw;

    // Detect if we are in strict-symmetric mode
    double numSteps = pow(2, bw) - 1;
    if (encodingMin == -encodingMax)
    {
        numSteps -= 1;  // in case of 8-bits, strict symmetric means we use 254 int values, instead of 255
    }

    // compute offset and delta on the fly
    encoding.delta = computeDelta(encodingMin, encodingMax, numSteps);
    encoding.offset = computeOffset(encodingMin, encoding.delta);

    if (use_cuda)
        cpuGpuMode = DlQuantization::ComputationMode::COMP_MODE_GPU;
    else
        cpuGpuMode = DlQuantization::ComputationMode::COMP_MODE_CPU;
}

template <typename DTYPE>
void TensorQuantizationSim<DTYPE>::quantizeDequantizeTensor(const DTYPE* inputTensorData, size_t inputTensorCount,
                                                            DTYPE* outputTensorData, double encodingMin,
                                                            double encodingMax, uint8_t bw, RoundingMode roundingMode,
                                                            bool use_cuda)
{
    DlQuantization::ComputationMode cpuGpuMode;
    TfEncoding encoding;

    _fillQuantizeInfo(encoding, cpuGpuMode, bw, encodingMin, encodingMax, use_cuda);
    quantizeDequantize(inputTensorData, inputTensorCount, encoding, outputTensorData, cpuGpuMode, roundingMode);
}

template <typename DTYPE>
void TensorQuantizationSim<DTYPE>::quantizeTensor(const DTYPE* inputTensorData, size_t inputTensorCount,
                                                  DTYPE* outputTensorData, double encodingMin, double encodingMax,
                                                  uint8_t bw, RoundingMode roundingMode, bool use_cuda,
                                                  bool shiftToSigned)
{
    DlQuantization::ComputationMode cpuGpuMode;
    TfEncoding encoding;

    _fillQuantizeInfo(encoding, cpuGpuMode, bw, encodingMin, encodingMax, use_cuda);
    quantizeToFxp(inputTensorData, inputTensorCount, encoding, outputTensorData, cpuGpuMode, roundingMode,
                  shiftToSigned);
}

template class TensorQuantizationSim<float>;
template class TensorQuantizationSim<double>;

}