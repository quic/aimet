//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2016-2022, Qualcomm Innovation Center, Inc. All rights reserved.
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


#ifndef QUANTIZATION_HPP
#define QUANTIZATION_HPP
#include <cstddef>
#include <cstdint>
#include <vector>

namespace DlQuantization
{

/**
 * Run on the CPU or GPU.
 */
enum ComputationMode
{
    COMP_MODE_CPU,
    COMP_MODE_GPU
};


/**
 * @brief Device memory allocator interface.
 */
class IAllocator
{
public:
    /**
     * @brief Allocate memory to the associated device and return the pointer.
     * @param bytes Bytes to allocate
     */
    virtual void* allocateRaw(size_t bytes) = 0;

    /**
     * @brief Deallocate the memory occupied by the pointer.
     * @param ptr Pointer to deallocate.
     */
    virtual void deleteRaw(void* ptr) = 0;
};


/**
 * @brief The fixed point quantization mode defines what fixed point scheme
 * we use, and how we compute a suitable encoding.
 */
enum QuantizationMode
{
    // TensorFlow quantization with static encoding; zero can be represented.
    QUANTIZATION_TF,

    // An enhanced version of TensorFlow. The encoding is chosen such that the SQNR is maximal.
    QUANTIZATION_TF_ENHANCED,

    // Ranges (min, max) are learnt during training
    QUANTIZATION_RANGE_LEARNING,

    // Percentile calibration. Compute the encoding by adjusting the min/max range of the tensor
    // by clipping percentile of outliers.
    QUANTIZATION_PERCENTILE,

    // Compute the encoding by adjusting the min/max range of the tensor based on the mean square
    // error due to quantization of the tensor.
    QUANTIZATION_MSE,

    // Compute the optimal quantization range (thresholds) for a tensor based on minimizing the
    // Kullback-Leibler divergence.
    QUANTIZATION_ENTROPY,
};

/**
 * @brief TensorFlow-style fixed point format.
 *
 * We use this fixed point format for both TensorFlow quantization and Q-format
 * quantization.
 */
struct TfEncoding
{
    double min;
    double max;
    double delta;
    double offset;
    int bw;
};

/**
 * @brief The fixed point format of activations in a given network layer.
 */
struct TfEncodingLayer
{
    // The encoding of input tensors.
    std::vector<TfEncoding> in;
    // The encoding of output tensors.
    std::vector<TfEncoding> out;
};

/**
 * @brief Layer input or output activations.
 */
enum LayerInOut
{
    LAYER_INPUT,
    LAYER_OUTPUT
};

/**
 * @brief Rounding mode.
 */
enum RoundingMode
{
    ROUND_NEAREST,
    ROUND_STOCHASTIC
};

/**
 * @brief Performs quantize-dequantize using numpy-style broadcasting between input and encodings
 *
 * This function takes in input and encoding data along with padded stride information to indicate how the encodings
 * should be applied to the input. The following assumptions must be true about the strides and shapes of the input
 * and encodings:
 *
 *  1) The size of the input and output tensors must be equal to numElement
 *  2) The length of inputStrides and encodingStrides must both be equal to numDims
 *  3) The input data must be viewable as a tensor with rank numDims and strides inputStrides
 *  4) The encoding data must be viewable as a tensor with rank numDims which is broadcastable to
 *     the input tensor under numpy-style broadcasting
 *  5) The encodingStrides must indicate the strides with respect to the output tensor index.
 *     I.e., encodingStrides[i] = 0 for all broadcast dimensions i (examples below)
 *
 * Calculate encodingStride relative to the output tenosr:
 * 1) Pad the encoding shape with leading 1s such that len(encodingShape) == len(outputShape):
 *      outputShape = [x, y, z, w]
 *      encodingShape = [y, z, 1] --> [1, y, z, 1]
 * 2) Calculate the stride based on the padded shape:
 *      encodingStride = [yz, z, 1, 1]
 * 3) Set encodingStride[i] = 0 for all encodingShape[i] = 1 and outputShape[i] != 1:
 *      encodingStride = [0, z, 1, 0]
 *
 * Examples:
 *
 * Given: input tensor shape [x, y, z, w] and encoding shape [x, 1, z, 1]
 *  - Correct:
 *      inputStrides    = [yzw, zw, w, 1]
 *      encodingStrides = [z, 0, 1, 0] <- encodingStrides[i] = 0 where encoding is broadcast to input
 *  - Incorrect
 *      inputStrides    = [yzw, zw, w, 1]
 *      encodingStrides = [z, z, 1, 1] <- encodingStrides are w/r/t the encoding, not output
 *
 * Given: input tensor shape [x, y, z, w] and encoding shape [y, z, 1]
 *  - Correct:
 *      inputStrides = [yzw, zw, w, 1]
 *      encodingStrides = [0, z, 1, 0] <- len(encodingStrides) = numDims
 *  - Incorrect
 *      inputStrides = [yzw, zw, w, 1]
 *      encodingStrides = [z, 1, 0] <- len(encodingStrides) != numDims
 *
 *
 * @tparam DTYPE Floating point data type of input/output/encodings
 * @param in Pointer to input data
 * @param out Pointer to output data, must be on same device as input
 * @param numElement Number of elements in tensor
 * @param numDims Number of tensor dimensions
 * @param inputStrides Pointer to input stride data, must be on same device as input and length numDims
 * @param encodingStrides Pointer to padded encoding stride data, must be on same device as input and length numDims.
 * Should be 0 along broadcast dimensions
 * @param encodingMin Pointer to min encoding data (on same device as input)
 * @param encodingMax Pointer to max encoding data (on same device as input)
 * @param encodingDelta Pointer to delta encoding data (on same device as input)
 * @param encodingOffset Pointer to offset encoding data (on same device as input)
 * @param modeCpuGpu Indicates whether tensors exist on CPU or GPU
 * @param stream Cuda stream on which to launch kernel in GPU mode
 */
template <typename DTYPE>
void quantizeDequantizeBroadcast(
    const DTYPE* in,
    DTYPE* out,
    int64_t numElement,
    int64_t numDims,
    const int64_t* inputStrides,
    const int64_t* encodingStrides,
    const DTYPE* encodingMin,
    const DTYPE* encodingMax,
    const DTYPE* encodingDelta,
    const DTYPE* encodingOffset,
    ComputationMode modeCpuGpu,
    void* stream);

}   // End of namespace DlQuantization.

#endif   // QUANTIZATION_HPP
