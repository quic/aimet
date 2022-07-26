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
#ifndef QUANTIZER_FACTORY_HPP
#define QUANTIZER_FACTORY_HPP

#include <bits/unique_ptr.h>
#include <string>
#include <vector>


#include "DlQuantization/GraphQuantizer.h"
#include "DlQuantization/IQuantizationEncodingAnalyzer.hpp"
#include "DlQuantization/IQuantizer.hpp"
#include "DlQuantization/ITensorQuantizationSim.h"
#include "DlQuantization/Quantization.hpp"

namespace DlQuantization
{

/**
 * @brief Create an object for fixed point quantization.
 * @param layer_names The activations of these layers will get quantized to
 * fixed point.
 * @param mode_cpu_gpu The computation happens on the CPU or GPU.
 * @param bw_activations The library needs to know the fixed point bit-widths
 * in advance.
 * @param quantization_mode The fixed point mode.
 * @pre There is no precondition requirement.
 * @attention The new object is allocated on the heap and needs to be freed by
 * the caller.
 *
 * Before the library will be able to quantize activations, it needs to gather
 * statistical data to find a suitable fixed point format. For this to work, the
 * library needs to know the layer names as well as all the bit-widths that
 * will be used for quantization.
 * As a case in point, if the user will
 * quantize activations to 8-bit fixed point only, 'bw_activations' needs to
 * have one entry with value '8'.
 * The computation mode indicates whether the library will do the quantization
 * on the CPU or GPU. Some API calls contain pointers to tensors. In CPU mode,
 * those pointers will need to point to CPU memory, and vice versa in GPU mode.
 * The template parameter DTYPE can be float or double.
 */
template <typename DTYPE>
IQuantizer<DTYPE>* GetQuantizerInstance(const std::vector<std::string>& layer_names, ComputationMode mode_cpu_gpu,
                                        const std::vector<int>& bw_activations, QuantizationMode quantization_mode);

template <typename DTYPE>
std::unique_ptr<IQuantizationEncodingAnalyzer<DTYPE>> getEncodingAnalyzerInstance(QuantizationMode quantization_mode);

template <typename DTYPE>
std::unique_ptr<ITensorQuantizationSim<DTYPE>> getTensorQuantizationSim();

std::unique_ptr<GraphQuantizer> getGraphQuantizerInstance(const std::vector<std::string>& tensorNames,
                                                          ComputationMode modeCpuGpu, QuantizationMode quantMode);

}   // End of namespace DlQuantization

#endif   // QUANTIZER_FACTORY_HPP
