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

// This file contains code to analyze and calculate quantization encodings
// This code is specific for the TF Enhanced quantization scheme
// Support for other schemes is currently TBD

#ifndef QUANTIZATION_SIM_H
#define QUANTIZATION_SIM_H

#include <cstddef>
#include <cstdint>
#include "DlQuantization/ITensorQuantizationSim.h"
#include "DlQuantization/Quantization.hpp"

namespace DlQuantization
{
template <typename DTYPE>
class TensorQuantizationSim : public ITensorQuantizationSim<DTYPE>
{
private:
    void _fillQuantizeInfo(TfEncoding& encoding, DlQuantization::ComputationMode& cpuGpuMode, uint8_t bw,
                           double encodingMin, double encodingMax, bool use_cuda);
public:
    explicit TensorQuantizationSim();
    void gateMinMax(double& encodingMin, double& encodingMax);
    void quantizeDequantizeTensor(const DTYPE* inputTensorData, size_t inputTensorCount, DTYPE* outputTensorData,
                                  double encodingMin, double encodingMax, uint8_t bw, RoundingMode roundMode,
                                  bool use_cuda) override;
    void quantizeTensor(const DTYPE* inputTensorData, size_t inputTensorCount, int32_t* outputTensorData,
                        double encodingMin, double encodingMax, uint8_t bw, RoundingMode roundMode, bool use_cuda,
                        bool shiftToSigned)
                        override;
    void quantizeTensor(const DTYPE* inputTensorData, size_t inputTensorCount, int64_t* outputTensorData,
                        double encodingMin, double encodingMax, uint8_t bw, RoundingMode roundMode, bool use_cuda,
                        bool shiftToSigned)
                        override;
};

}   // namespace DlQuantization

#endif   // QUANTIZATION_SIM_H
