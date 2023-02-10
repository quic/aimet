//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2019 - 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

#ifndef I_QUANTIZATION_SIM_H
#define I_QUANTIZATION_SIM_H

#include "Quantization.hpp"

namespace DlQuantization
{
template <typename DTYPE>
class ITensorQuantizationSim
{
public:
    virtual void quantizeDequantizeTensor(const DTYPE* inputTensorData, size_t inputTensorCount,
                                          DTYPE* outputTensorData,
                                          double encodingMin, double encodingMax,
                                          uint8_t bw, RoundingMode roundMode,
                                          bool use_cuda) = 0;
    virtual void quantizeTensor(const DTYPE* inputTensorData, size_t inputTensorCount, DTYPE* outputTensorData,
                                double encodingMin, double encodingMax, uint8_t bw, RoundingMode roundMode,
                                bool use_cuda, bool shiftToSigned) = 0;
    /**
     * @brief Convert a tensor from DTYPE to quantized 8-bit packed format
     */
    virtual void quantizeTensorPacked(const DTYPE* inputTensorData, size_t inputTensorCount, std::vector<uint8_t>& outputTensorData,
                                      double encodingMin, double encodingMax, uint8_t bw, RoundingMode roundMode,
                                      bool useCuda, bool shiftToSigned) = 0;

    /**
     * @brief Convert a tensor from quantized 8-bit format into DTYPE
     */
    virtual void dequantizeTensor(const uint8_t* inputTensorData, size_t inputTensorCount, DTYPE* output,
                                  double encodingMin, double encodingMax, uint8_t bw, bool shiftToSigned) = 0;

    /**
     * @brief Performs per channel quantization for each split in splits, and concatenates the result into a quantized
     *        int output before de-quantizing back to float.
     * @relates quantizeDequantizeTensor
     */
    virtual void quantizeDequantizePerChannelTensor( std::vector <std::vector<DTYPE>>& splits,
                                                    std::vector <uint32_t> splitShape,
                                                    uint32_t axis, DTYPE* outputTensorData,
                                                    const std::vector <TfEncoding> &encodings,
                                                    uint8_t bw, RoundingMode roundMode,
                                                    bool useCuda) = 0;
    /**
     * @brief Performs per channel quantization for each split in splits, and concatenates the result into a quantized
     *         output before de-quantizing. Output is packed 8 bit quantized data.
     * @relates quantizeDequantizePerChannelTensor
     * @param[in/out] Unsigned 8 bit output tensor
     */
    virtual void quantizePerChannelTensorPacked(std::vector <std::vector<DTYPE>>& splits,
                                                std::vector <uint32_t> splitShape,
                                                uint32_t axis, std::vector<uint8_t>& outputTensorData,
                                                const std::vector <TfEncoding> &encodings,
                                                uint8_t bw, RoundingMode roundMode,
                                                bool useCuda, bool shiftToSigned) = 0;

    /**
     * @brief Convert a tensor from quantized 8-bit format into DTYPE, by splitting the data into channels and
     *        dequantizing independently, before concatenating the final result into the output tensor.
     */
    virtual void dequantizePerChannelTensor(const uint8_t* inputTensorData, const std::vector<uint32_t> &inputShape, uint32_t axis,
                                            DTYPE* outputTensorData, uint8_t bw, const std::vector<TfEncoding> &encodings,
                                            bool shiftToSigned) = 0;

    virtual void fillEncodingInfo(TfEncoding& encoding, uint8_t bw, double encodingMin, double encodingMax) = 0;

    virtual void generateScaleOffset(double &encodingMin, double &encodingMax, uint8_t bw,
                                     double &encodingScale, double &encodingOffset) = 0;
};

}   // namespace DlQuantization

#endif   // I_QUANTIZATION_SIM_H
