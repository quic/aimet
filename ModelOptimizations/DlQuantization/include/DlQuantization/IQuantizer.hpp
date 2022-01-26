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


#ifndef I_QUANTIZER_HPP
#define I_QUANTIZER_HPP

#include <map>
#include <string>
#include <vector>

#include "DlQuantization/Quantization.hpp"

namespace DlQuantization
{
/**
 * @brief The IQuantizer object is templated and can be DTYPE=double or DTYPE=float.
 * All unquantized data which goes into and comes out of the library will have
 * this data type. As a case in point, if a user wants to quantize a CNN which
 * has double precision parameters, he should use DTYPE=double.
 */
template <typename DTYPE>
class IQuantizer
{
public:
    /**
     * @brief Gather statistics for layer activations.
     * @param layer The name of the network layer.
     * @param mode_in_out Gather statistics for layer inputs or layer outputs.
     * @param activations The pointers to the activation tensors.
     * @param count The number of elements in each input or output tensor.
     * @throw Throws a standard C++ exception if error encountered.
     *
     * This method doesn't perform any quantization, but just gathers statistical
     * data of the activations. The statistical data can be used later to find
     * a suitable fixed point encoding.
     * Since some layers have multiple input or output tensors, we gather
     * statistics for each tensor independently. The size of the vectors
     * 'activations' and 'count' should match the number of input or output
     * tensors in this layer.
     * Don't use this together with SetEncoding().
     */
    virtual void UpdateStats(const std::string& layer, LayerInOut mode_in_out,
                             const std::vector<const DTYPE*>& activations, const std::vector<size_t>& count) = 0;

    /**
     * @brief Quantize activations to fixed point and de-quantize result back to
     * floating point.
     * @param layer The name of the network layer.
     * @param mode_in_out Quantize layer inputs or layer outputs.
     * @param bw The activations will be quantized to this bit-width.
     * @param acts The pointers to the activation tensors.
     * @param count The number of elements in each activation tensor.
     * @param acts_quantized Return pointers to the quantized activation tensors.
     * @param encoding Return the fixed point format which was used for each
     * tensor.
     * @pre Before calling this method, use UpdateStats() or SetEncoding().
     * @throw Throws a standard C++ exception if error encountered.
     *
     * The quantization can be performed in-place.
     */
    virtual void QuantizeDequantizeActs(const std::string& layer, LayerInOut mode_in_out, int bw,
                                        std::vector<DTYPE*>& acts, const std::vector<size_t>& count,
                                        std::vector<DTYPE*>& acts_quantized, std::vector<TfEncoding>& encoding) = 0;

    /**
     * @brief Quantize network parameters to fixed point and de-quantize result
     * back to floating point.
     * @param bw The parameters will be quantized to this bit-width.
     * @param params The weight or bias tensor.
     * @param count The number of elements in the parameter tensor.
     * @param mode_rounding The rounding mode which should be used for
     * quantization.
     * @param params_quantized Return the fixed point parameters.
     * @param encoding Return the fixed point format which was used.
     * @pre There is no precondition requirement.
     * @throw Throws a standard C++ exception if error encountered.
     *
     * Parameters get quantized on-the-fly. No statistics need to be gathered
     * prior to this call. The library doesn't remember the fixed point format
     * it used.
     * The quantization can be performed in-place.
     */
    virtual void QuantizeDequantizeParams(int bw, DTYPE* params, size_t count, RoundingMode mode_rounding,
                                          DTYPE* params_quantized, TfEncoding& encoding) = 0;

    /**
     * @brief Set the fixed point format for layer activations of the whole
     * network.
     * @param encoding A map of layer names and their encoding.
     * @pre There is no precondition requirement.
     * @throw Throws a standard C++ exception if error encountered.
     *
     * This method should be used if the user knows what encoding should be used
     * for layer activations. The library will then use this encoding to quantize
     * activations.
     * Don't use this together with UpdateStats().
     */
    virtual void SetEncoding(const std::map<std::string, TfEncodingLayer>& encoding) = 0;

    /**
     * @brief Set the fixed point format for layer activations in one layer.
     * @param layer The name of the network layer.
     * @param encoding The fixed point format of layer inputs and outputs.
     * @pre There is no precondition requirement.
     * @throw Throws a standard C++ exception if error encountered.
     *
     * This method should be used if the user knows what encoding should be used
     * for layer activations. The library will then use this encoding to quantize
     * activations.
     * Don't use this together with UpdateStats().
     */
    virtual void SetEncoding(const std::string& layer, TfEncodingLayer& encoding) = 0;

    /**
     * @brief Get the fixed point format for layer activations.
     * @params bws For each layer, specify the fixed point bit-width.
     * @param encoding The method returns a map of layer names and their encoding.
     * @pre Before using this method, call UpdateStats() or SetEncoding().
     * @throw Throws a standard C++ exception if error encountered.
     *
     * This method can be used if the user wants to know the fixed point encoding
     * of all activation tensors in the network.
     */
    virtual void GetEncoding(const std::map<std::string, int>& bws,
                             std::map<std::string, TfEncodingLayer>& encoding) = 0;

    /**
     * @brief Get the fixed point format for one specific layer.
     * @param layer The layer name.
     * @param bw The bit-width of activations.
     * @param encoding Return the encoding of the input and output activations.
     */
    virtual void GetEncoding(const std::string& layer, unsigned int bw, TfEncodingLayer& encoding) = 0;

    /**
     * @brief Compute an accumulator fixed point format.
     * @param input_acts The fixed point format of the input activations.
     * @param weights The fixed point format of the weights.
     * @param accumulator Compute an accumulator fixed point format.
     * @pre There is no precondition requirement.
     * @throw Throws a standard C++ exception if error encountered.
     *
     * This method uses the fixed point format of activations and weights to find
     * a fixed point format for accumulators. This method assumes this layer
     * does MAC operations, and chooses the accumulator format as defined by CAL.
     */
    virtual void GetAccumulatorFormat(const TfEncoding& input_acts, const TfEncoding& weights,
                                      TfEncoding& accumulator) = 0;

    /**
     * @brief Compute the delta and offset, based on the bit-width and min and
     * max.
     * @param bw The bit-width of fixed point numbers.
     * @param min The lowest value that can be encoded.
     * @param max The largest value that can be encoded.
     * @param delta Compute the step size of the encoding.
     * @param offset Compute the offset of the encoding.
     * @throw Throws a standard C++ exception if error encountered.
     *
     * There are two uses cases for this call:
     *
     * 1) This is the common and simple use case. The user gets an encoding from
     * the library, only stores the min and max, and wants to recalculate the
     * delta and offset. In this use case, the user will get back the original
     * encoding. Specifically, the min and max values returned by the function
     * don't get changed.
     *
     * 2) In this use case, the user has a random min and max, which wasn't
     * computed by the library. In this case, the library will compute a number
     * encoding based on bw, min and max. Note that in this use case, the min and
     * max values might need to be adjusted to find a valid encoding.
     * When using Q-format encoding, the library only considers the bw and max
     * value, since the number format is well defined without the min value.
     * In the case of TF encoding, we calculate the offset based on bw, min, and
     * max. For TF encoding, the returned offset will be rounded to an integer.
     */
    virtual void ComputeDeltaAndOffset(int bw, double& min, double& max, double& delta, double& offset) = 0;

    virtual ~IQuantizer() {};
};

}   // End of namespace DlQuantization

#endif   // I_QUANTIZER_HPP
