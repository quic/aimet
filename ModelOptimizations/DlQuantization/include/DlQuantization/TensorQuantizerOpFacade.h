//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2020-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

#ifndef AIMET_TENSOR_QUANTIZER_OP_FACADE_H
#define AIMET_TENSOR_QUANTIZER_OP_FACADE_H

#include <DlQuantization/Quantization.hpp>

namespace DlQuantization
{


enum class TensorQuantizerOpMode
{
    updateStats,
    oneShotQuantizeDequantize,
    quantizeDequantize,
    passThrough
};


/**
 * This is a facade interface for the TensorQuantizer class. This facade only exposes the interfaces that are needed
 * by a C++ custom op (for TensorFlow or PyTorch). Specifically methods that require numpy tensors are omitted
 * as these are only intended to be invoked from Python code which has easy access to numpy variants of torch and tf
 * tensors.
 */
class TensorQuantizerOpFacade
{
public:
    /**
     * Reset stats being collected to compute encoding
     */
    virtual void resetEncodingStats() = 0;

    /**
     * Update stats being collected to compute encoding
     * @param tensor Tensor to update the stats with
     * @param tensorSize Size of the tensor (number of tensor elements)
     * @param useCuda If true, the tensor is assumed to be in CUDA memory
     */
    virtual void updateStats(const float* tensor, std::size_t tensorSize, bool useCuda)                    = 0;
    virtual void updateStats(const float* tensor, std::size_t tensorSize, bool useCuda, IAllocator* alloc) = 0;

    /**
     * Convert a tensor from float to quantized int and back to float
     * @param input Input tensor
     * @param tensorSize Size of the input tensor (number of tensor elements)
     * @param output Output tensor
     * @param encodingMin minimum value of encoding range
     * @param encodingMax maximum value of encoding range
     * @param bitwidth to be used
     * @param useCuda If true, both the input and output tensors are assumed to be in CUDA memory
     */
    virtual void quantizeDequantize(const float* input, std::size_t tensorSize, float* output, double encodingMin,
                                    double encodingMax, unsigned int bitwidth, bool useCuda) = 0;

    virtual void quantizeDequantize(const float* input, std::size_t tensorSize, float* output, double encodingMin,
                                    double encodingMax, unsigned int bitwidth, bool useCuda, void* stream) = 0;
    /**
     * Compute the encoding for this tensor using stats collected so far
     */
    virtual TfEncoding computeEncoding(unsigned int bitwidth, bool useSymmetricEncoding) = 0;

    virtual bool getStrictSymmetric()   = 0;
    virtual bool getUnsignedSymmetric() = 0;
};


}   // namespace DlQuantization

#endif   // AIMET_TENSOR_QUANTIZER_OP_FACADE_H
