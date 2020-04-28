//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2017-2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

#ifndef QC_QUANTIZER_HPP
#define QC_QUANTIZER_HPP

#include <memory>
#include "DlQuantization/IQuantizer.hpp"
#include "DlQuantization/QuantizerFactory.hpp"

namespace QcOp {

enum class Device {
  CPU = 0,
  GPU = 1
};

typedef enum {
    CONFIG_TYPE_MIN,
    CONFIG_TYPE_UPDATE_STATS,
    CONFIG_TYPE_GET_ENCODING,
    CONFIG_TYPE_SET_ENCODING,
    CONFIG_TYPE_Q_DQ_PARAMS,
    CONFIG_TYPE_Q_DQ_ACTIVATIONS,
    CONFIG_TYPE_MAX
} OP_CONFIG_TYPE;

template <typename T>
class QC_Quantizer {

public:

    void setup (
        std::string &op_name,
        DlQuantization::QuantizationMode quant_mode,
        DlQuantization::RoundingMode roundingMode,
        int bitwidth ) {
        op_name_ = op_name;
        quant_mode_ = quant_mode;
        // Use Op only in OUTPUT activation mode.
        activationMode_ = DlQuantization::LAYER_OUTPUT;
        roundingMode_ = roundingMode;
        bitwidth_ = bitwidth;
    }

    static void InitQuantizer (
        const std::vector<std::string>& layer_names,
        DlQuantization::ComputationMode mode_cpu_gpu,
        const std::vector<int>& bw_activations,
        DlQuantization::QuantizationMode quantization_mode) {
        if (nullptr == quantizer_) {
            quantizer_ = std::shared_ptr<DlQuantization::IQuantizer<float> >(
                DlQuantization::GetQuantizerInstance<T> (
                layer_names,mode_cpu_gpu,bw_activations,quantization_mode));
        }
    }

    static void ResetQuantizerInstance() { quantizer_.reset(); }

    // Forward() method with const pointers as inputs
    void Forward (
        OP_CONFIG_TYPE config,
        const std::vector<const T*>& in_tensors,
        const std::vector<size_t>& in_count,
        std::vector<T*>& out_tensors,
        DlQuantization::TfEncodingLayer &in_encoding,
        DlQuantization::TfEncodingLayer &out_encoding) {
        if (nullptr == quantizer_)
            throw std::runtime_error ("Quantizer not initialized. Aborting.");

        switch (config) {
            case CONFIG_TYPE_UPDATE_STATS: {
                quantizer_->UpdateStats (
                    op_name_, activationMode_, in_tensors, in_count);
            }
            break;
            case CONFIG_TYPE_GET_ENCODING: {
                quantizer_->GetEncoding (op_name_, bitwidth_, out_encoding);
            }
            break;
            case CONFIG_TYPE_SET_ENCODING: {
                // First compute delta, offset for the given input range
                for (int enc_count = 0; enc_count < in_encoding.out.size(); ++enc_count) {
                    quantizer_->ComputeDeltaAndOffset (bitwidth_,
                        in_encoding.out[enc_count].min, in_encoding.out[enc_count].max,
                        in_encoding.out[enc_count].delta, in_encoding.out[enc_count].offset);
                }
                // Now set encodings to the op
                quantizer_->SetEncoding(op_name_, in_encoding);
            }
            break;
            case CONFIG_TYPE_Q_DQ_ACTIVATIONS: {
                std::cerr << "Config type: " << config <<
                    " not handled with this API. Use the non-const version instead." << std::endl;
            }
            break;
            default:
            break;
        }
    }

    // Overloaded Forward() method with non-const pointers as inputs
    void Forward(OP_CONFIG_TYPE config,
                 std::vector<T*>& in_tensors,
                 const std::vector<size_t>& in_count,
                 const bool training_in_progress,
                 std::vector<T*>& out_tensors,
                 DlQuantization::TfEncodingLayer &in_encoding,
                 DlQuantization::TfEncodingLayer &out_encoding)
    {

        if (nullptr == quantizer_)
            throw std::runtime_error ("Quantizer not initialized. Aborting.");

        // If training is not in progress, we only support NEAREST rounding
        DlQuantization::RoundingMode roundingMode = roundingMode_;
        if (!training_in_progress)
        {
            roundingMode = DlQuantization::ROUND_NEAREST;
        }

        switch (config)
        {
            case CONFIG_TYPE_Q_DQ_ACTIVATIONS:
            {
                quantizer_->QuantizeDequantizeActs(op_name_, activationMode_, bitwidth_,
                                                   in_tensors, in_count, out_tensors, out_encoding.out);
            }
            break;
            case CONFIG_TYPE_Q_DQ_PARAMS:
            {
                DlQuantization::TfEncoding param_encodings;
                quantizer_->QuantizeDequantizeParams(bitwidth_, in_tensors[0], in_count[0],
                                                     roundingMode, out_tensors[0], param_encodings);
                out_encoding.out.push_back (param_encodings);
            }
            break;
            case CONFIG_TYPE_UPDATE_STATS:
            case CONFIG_TYPE_GET_ENCODING:
            case CONFIG_TYPE_SET_ENCODING:
            {
                std::cerr << "Config type: " << config <<
                          " not handled with this API. Use the const version instead." << std::endl;
                break;
            }
            default:
            break;
        }
    }

private:
    std::string op_name_;   // The name of the layer whose params/activations are to be quantized
    DlQuantization::QuantizationMode quant_mode_;
    DlQuantization::LayerInOut activationMode_;
    DlQuantization::RoundingMode roundingMode_;
    int bitwidth_;
    static std::shared_ptr<DlQuantization::IQuantizer<T> > quantizer_;
};

// Static object instantiation
template <typename T>
std::shared_ptr<DlQuantization::IQuantizer<T> > QC_Quantizer<T>::quantizer_;

}
#endif // QC_QUANTIZER_HPP
