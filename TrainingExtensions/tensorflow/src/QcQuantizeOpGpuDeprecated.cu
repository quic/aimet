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

#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "QcQuantizeOpDeprecated.hpp"

using namespace tensorflow;

#define EIGEN_USE_GPU
typedef Eigen::GpuDevice GPUDevice;

// GPU specialization of actual computations.
template <typename T>
struct QcQuantizeDeprecatedFunctor<GPUDevice, T>
{
    /*Operator for const input tensors */
    void operator()(const GPUDevice& d, QcOp::OP_CONFIG_TYPE config, const std::vector<const T*>& in_tensors,
                    const std::vector<size_t>& in_tensor_counts, std::vector<T*> out_tensors,
                    DlQuantization::TfEncodingLayer& in_encoding, DlQuantization::TfEncodingLayer& out_encoding,
                    T* output_min_tensor, T* output_max_tensor, QcOp::QC_Quantizer<T>& quantizer)
    {
        quantizer.Forward(config, in_tensors, in_tensor_counts, out_tensors, in_encoding, out_encoding);

        // copy input_tensors to output_tensors
        // passthrough for CONFIG_TYPE_UPDATE_STATS
        if (config == QcOp::CONFIG_TYPE_UPDATE_STATS)
        {
            for (int idx = 0; idx < in_tensors.size(); idx++)
            {
                cudaMemcpy(out_tensors[idx], in_tensors[idx], in_tensor_counts[idx] * sizeof(T),
                           cudaMemcpyDeviceToDevice);
            }
        }
        long long int enc_size = static_cast<long long int>(out_encoding.out.size());
        T output_min[enc_size], output_max[enc_size];

        // copy min and max out_encodings in local variable separetely
        for (int idx = 0; idx < enc_size; idx++)
        {
            output_min[idx] = out_encoding.out[idx].min;
            output_max[idx] = out_encoding.out[idx].max;
        }
        // transfer min and max encodings to GPU
        cudaMemcpy(output_min_tensor, &output_min, enc_size * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(output_max_tensor, &output_max, enc_size * sizeof(T), cudaMemcpyHostToDevice);
    }

    /*Operator for non-const input tensors */
    void operator()(const GPUDevice& d, QcOp::OP_CONFIG_TYPE config, std::vector<T*>& in_tensors,
                    const std::vector<size_t>& in_tensor_counts, const bool* training_in_progress,
                    std::vector<T*> out_tensors, DlQuantization::TfEncodingLayer& in_encoding,
                    DlQuantization::TfEncodingLayer& out_encoding, T* output_min_tensor, T* output_max_tensor,
                    QcOp::QC_Quantizer<T>& quantizer)
    {
        // Read the GPU memory to parse the training_in_progress flag
        bool is_train;
        cudaMemcpy(&is_train, training_in_progress, sizeof(is_train), cudaMemcpyDeviceToHost);

        quantizer.Forward(config, in_tensors, in_tensor_counts, is_train, out_tensors, in_encoding, out_encoding);

        // copy input_tensors to output_tensors
        // passthrough for CONFIG_TYPE_UPDATE_STATS
        if (config == QcOp::CONFIG_TYPE_UPDATE_STATS)
        {
            for (int idx = 0; idx < in_tensors.size(); idx++)
            {
                cudaMemcpy(out_tensors[idx], in_tensors[idx], in_tensor_counts[idx] * sizeof(T),
                           cudaMemcpyDeviceToDevice);
            }
        }
        long long int enc_size = static_cast<long long int>(out_encoding.out.size());
        T output_min[enc_size], output_max[enc_size];

        // copy min and max out_encodings in local variable separetely
        for (int idx = 0; idx < enc_size; idx++)
        {
            output_min[idx] = out_encoding.out[idx].min;
            output_max[idx] = out_encoding.out[idx].max;
        }
        // transfer min and max encodings to GPU
        cudaMemcpy(output_min_tensor, &output_min, enc_size * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(output_max_tensor, &output_max, enc_size * sizeof(T), cudaMemcpyHostToDevice);
    }
};

// Instantiate functors for the types of OpKernels registered.
template struct QcQuantizeDeprecatedFunctor<GPUDevice, float>;

#endif   // GOOGLE_CUDA
