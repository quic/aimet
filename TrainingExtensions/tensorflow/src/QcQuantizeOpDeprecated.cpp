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

#include "QcQuantizeOpDeprecated.hpp"

#define EIGEN_USE_THREADS
using namespace tensorflow;


REGISTER_OP("QcQuantizeDeprecated")
    .Input("in_tensors: L")   // list of input tensors (weights/activations)
    .Input("training_in_progress: bool")
    .Output("out_tensors: L")    // list of output tensors (weights/activations)
    .Output("enc_out_mins: T")   // tensor of O/P min encodings
    .Output("enc_out_maxs: T")   // tensor of O/P max encodings

    .Attr("op_name: string")
    .Attr("config: int")   //{'UPDATE_STATS', 'Q_DQ_PARAMS', 'Q_DQ_ACTS', 'GET_ENC','SET_ENC',...}")
    .Attr("T: {float} = DT_FLOAT")
    .Attr("L: list({float})")   // Accepts a list of tensors
    .Attr("bitwidth: int")
    .Attr("quant_mode: {'TF', 'TF_ENHANCED'} = 'TF_ENHANCED'")
    .Attr("round_mode: {'NEAREST', 'STOCHASTIC'} = 'NEAREST'")
    .Attr("fixed_enc_mins: list(float)")   // tensor of I/P min encodings
    .Attr("fixed_enc_maxs: list(float)")   // tensor of I/P max encodings
    .Attr("num_tensors: int = 0")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        for (int i = 0; i < c->num_inputs(); i++)
        {
            c->set_output(i, c->input(i));
        }
        return Status::OK();
    });

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computations.
template <typename T>
struct QcQuantizeDeprecatedFunctor<CPUDevice, T>
{
    /*Operator for const input tensors */
    void operator()(const CPUDevice& d, QcOp::OP_CONFIG_TYPE config, const std::vector<const T*>& in_tensors,
                    const std::vector<size_t>& in_tensor_counts, std::vector<T*> out_tensors,
                    DlQuantization::TfEncodingLayer& in_encoding, DlQuantization::TfEncodingLayer& out_encoding,
                    T* output_min_tensor, T* output_max_tensor, QcOp::QC_Quantizer<T>& quantizer)
    {
        quantizer.Forward(config, in_tensors, in_tensor_counts, out_tensors, in_encoding, out_encoding);

        // copy input_tensor to output_tensor
        // passthrough
        if (config == QcOp::CONFIG_TYPE_UPDATE_STATS)
        {
            for (int idx = 0; idx < in_tensors.size(); idx++)
            {
                std::copy(in_tensors[idx], in_tensors[idx] + in_tensor_counts[idx], out_tensors[idx]);
            }
        }
        long long int enc_size = static_cast<long long int>(out_encoding.out.size());
        // copy min and max output encodings
        for (size_t idx = 0; idx < enc_size; ++idx)
        {
            output_min_tensor[idx] = out_encoding.out[idx].min;
            output_max_tensor[idx] = out_encoding.out[idx].max;
        }
    }

    /*Operator for non-const input tensors */
    void operator()(const CPUDevice& d, QcOp::OP_CONFIG_TYPE config, std::vector<T*>& in_tensors,
                    const std::vector<size_t>& in_tensor_counts, const bool* training_in_progress,
                    std::vector<T*> out_tensors, DlQuantization::TfEncodingLayer& in_encoding,
                    DlQuantization::TfEncodingLayer& out_encoding, T* output_min_tensor, T* output_max_tensor,
                    QcOp::QC_Quantizer<T>& quantizer)
    {
        quantizer.Forward(config, in_tensors, in_tensor_counts, *training_in_progress, out_tensors, in_encoding,
                          out_encoding);

        // copy input_tensor to output_tensor
        // passthrough
        if (config == QcOp::CONFIG_TYPE_UPDATE_STATS)
        {
            for (int idx = 0; idx < in_tensors.size(); idx++)
            {
                std::copy(in_tensors[idx], in_tensors[idx] + in_tensor_counts[idx], out_tensors[idx]);
            }
        }
        long long int enc_size = static_cast<long long int>(out_encoding.out.size());
        // copy min and max output encodings
        for (size_t idx = 0; idx < enc_size; ++idx)
        {
            output_min_tensor[idx] = out_encoding.out[idx].min;
            output_max_tensor[idx] = out_encoding.out[idx].max;
        }
    }
};

// OpKernel definition.
// 'Device is templated on the type of device.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class QcQuantizeDeprecatedOp : public OpKernel
{
public:
    explicit QcQuantizeDeprecatedOp(OpKernelConstruction* context) : OpKernel(context)
    {
        std::string op_name;
        OP_REQUIRES_OK(context, context->GetAttr("op_name", &op_name));

        int32 config_data;
        OP_REQUIRES_OK(context, context->GetAttr("config", &config_data));
        config_ = static_cast<QcOp::OP_CONFIG_TYPE>(config_data);
        OP_REQUIRES(context, (config_ > QcOp::CONFIG_TYPE_MIN && config_ < QcOp::CONFIG_TYPE_MAX),
                    errors::InvalidArgument("Invalid config type: ", config_));

        // Get bitwidth
        int32 bitwidth = 0;
        OP_REQUIRES_OK(context, context->GetAttr("bitwidth", &bitwidth));

        // Get quantization mode format
        string quant_mode_string;
        DlQuantization::QuantizationMode quant_mode;
        OP_REQUIRES_OK(context, context->GetAttr("quant_mode", &quant_mode_string));
        OP_REQUIRES(context, (quant_mode_string == "TF" || quant_mode_string == "TF_ENHANCED"),
                    errors::InvalidArgument("Quantization mode must be one of "
                                            "TF|TF_ENHANCED; is " +
                                            quant_mode_string + "'"));

        if (quant_mode_string == "TF")
        {
            quant_mode = DlQuantization::QUANTIZATION_TF;
        }
        else
        {
            quant_mode = DlQuantization::QUANTIZATION_TF_ENHANCED;
        }

        // Set rounding mode
        string round_mode_string;
        DlQuantization::RoundingMode roundingMode;
        OP_REQUIRES_OK(context, context->GetAttr("round_mode", &round_mode_string));
        OP_REQUIRES(context, (round_mode_string == "NEAREST" || round_mode_string == "STOCHASTIC"),
                    errors::InvalidArgument("Round mode string must be NEAREST/STOCHASTIC"
                                            "; is '" +
                                            round_mode_string + "'"));
        if (round_mode_string == "NEAREST")
            roundingMode = DlQuantization::ROUND_NEAREST;
        else
            roundingMode = DlQuantization::ROUND_STOCHASTIC;

        // Set fixed encodings
        OP_REQUIRES_OK(context, context->GetAttr("fixed_enc_mins", &set_enc_min_));
        OP_REQUIRES_OK(context, context->GetAttr("fixed_enc_maxs", &set_enc_max_));

        OP_REQUIRES_OK(context, context->GetAttr("num_tensors", &num_tensors));

        // Setup QC Quantizer object with user configurations
        qc_quantizer_.setup(op_name, quant_mode, roundingMode, bitwidth);
    }

    void Compute(OpKernelContext* context) override
    {
        // Consume input tensors and allocate output tensors
        OpInputList in_tensors_list;
        OP_REQUIRES_OK(context, context->input_list("in_tensors", &in_tensors_list));

        // Read the is_training flag
        const Tensor* is_train_tensor;
        OP_REQUIRES_OK(context, context->input("training_in_progress", &is_train_tensor));

        const bool* training_in_progress = is_train_tensor->flat<bool>().data();

        OpOutputList out_tensors_list;
        OP_REQUIRES_OK(context, context->output_list("out_tensors", &out_tensors_list));
        for (int out_index = 0; out_index < in_tensors_list.size(); ++out_index)
        {
            TensorShape shape      = in_tensors_list[out_index].shape();
            Tensor* quantized_acts = NULL;
            OP_REQUIRES_OK(context, out_tensors_list.allocate(out_index, shape, &quantized_acts));
        }

        // Allocate placeholders for output encoding tensors
        Tensor* output_min_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output("enc_out_mins", {num_tensors}, &output_min_tensor));

        Tensor* output_max_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output("enc_out_maxs", {num_tensors}, &output_max_tensor));

        std::vector<const T*> in_tensors_const_vec;
        std::vector<T*> in_tensors_vec;
        std::vector<size_t> counts;
        std::vector<T*> out_tensors_vec;
        DlQuantization::TfEncodingLayer in_encoding;
        DlQuantization::TfEncodingLayer out_encoding;

        // Populate input, output tensors and their counts as std::vectors
        // to match DlQuantization API signatures.
        for (int index = 0; index < in_tensors_list.size(); ++index)
        {
            in_tensors_const_vec.push_back(in_tensors_list[index].flat<T>().data());
            counts.push_back(in_tensors_list[index].NumElements());
            out_tensors_vec.push_back(out_tensors_list[index]->flat<T>().data());
        }
        // Store set_encoding values into TfEncodingLayer structures
        OP_REQUIRES(context, (set_enc_min_.size() == set_enc_max_.size()),
                    errors::InvalidArgument("No. of elements in fixed_enc_min: ", set_enc_min_.size(),
                                            " doesn't match no. of elements in fixed_enc_max: ", set_enc_max_.size()));
        for (int index = 0; index < set_enc_min_.size(); ++index)
        {
            DlQuantization::TfEncoding encoding = {set_enc_min_.at(index), set_enc_max_.at(index), 0.0, 0.0, 0};
            // Assuming only output activations
            in_encoding.out.push_back(encoding);
        }

        switch (config_)
        {
        case QcOp::CONFIG_TYPE_UPDATE_STATS:
        case QcOp::CONFIG_TYPE_GET_ENCODING:
        case QcOp::CONFIG_TYPE_SET_ENCODING:
        {
            QcQuantizeDeprecatedFunctor<Device, T>()(
                context->eigen_device<Device>(), config_, in_tensors_const_vec, counts, out_tensors_vec, in_encoding,
                out_encoding, output_min_tensor->flat<T>().data(), output_max_tensor->flat<T>().data(), qc_quantizer_);
        }
        break;

        case QcOp::CONFIG_TYPE_Q_DQ_PARAMS:
        case QcOp::CONFIG_TYPE_Q_DQ_ACTIVATIONS:
        {
            // Store tensors in non-const pointer format for the API.
            for (int index = 0; index < in_tensors_list.size(); ++index)
            {
                in_tensors_vec.push_back(const_cast<T*>(in_tensors_list[index].flat<T>().data()));
            }
            QcQuantizeDeprecatedFunctor<Device, T>()(context->eigen_device<Device>(), config_, in_tensors_vec, counts,
                                                     training_in_progress, out_tensors_vec, in_encoding, out_encoding,
                                                     output_min_tensor->flat<T>().data(),
                                                     output_max_tensor->flat<T>().data(), qc_quantizer_);
        }
        break;
        default:
            break;
        }
    }

private:
    QcOp::OP_CONFIG_TYPE config_;
    std::vector<T> set_enc_min_;
    std::vector<T> set_enc_max_;
    QcOp::QC_Quantizer<T> qc_quantizer_;
    int32 num_tensors;
};

#define REGISTER_CPU(T)                                                                             \
    REGISTER_KERNEL_BUILDER(Name("QcQuantizeDeprecated").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
                            QcQuantizeDeprecatedOp<CPUDevice, T>);

REGISTER_CPU(float);

// Register the GPU kernels.

#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                                                             \
    REGISTER_KERNEL_BUILDER(Name("QcQuantizeDeprecated").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
                            QcQuantizeDeprecatedOp<GPUDevice, T>);
REGISTER_GPU(float);

#endif   // GOOGLE_CUDA
