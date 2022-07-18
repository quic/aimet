//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
#ifndef AIMET_GRAPH_QUANTIZER_H
#define AIMET_GRAPH_QUANTIZER_H

#include <map>
#include <memory>
#include <vector>

#include "DlQuantization/Quantization.hpp"
#include "DlQuantization/TensorQuantizer.h"

namespace DlQuantization
{

class GraphQuantizer;

/**
 * @brief This class models a quantizer for all activation tensors in a graph.
 * It encapsulates updating stats and computing an encoding for all activation tensors
 */
class GraphQuantizer
{
public:
    GraphQuantizer() = delete;

    /**
     * @brief Constructor which creates a tensor quantizer instance for all tensors in the graph
     * @param tensorNames A vector of tensor names associated with the graph
     * @param modeCpuGpu  The computation mode for all tensors
     * @param quantMode   The quantization mode for all tensors
     */
    explicit GraphQuantizer(const std::vector<std::string>& tensorNames, ComputationMode modeCpuGpu,
                            QuantizationMode quantMode);

    /**
     * @brief Reset stats being collected to compute encoding
     * @param tensorName The name of the tensor
     */
    void resetEncodingStats(const std::string& tensorName);

    /**
     * @brief Update stats (min, max) using tensor data
     * @param tensorName The name of the tensor
     * @param tensor Tensor to update the stats with
     * @param tensorSize Size of the tensor (number of tensor elements)
     */
    void updateStats(const std::string& tensorName, const float* tensor, std::size_t tensorSize);

    /**
     * @brief Computes an encoding using tensor data
     * @param tensorName The name of the tensor
     * @param useSymmetricEncodings Flag to indicate if symmetric min/max ranges should be used
     * @param useStrictSymmetric Min/Max ranges are symmetric around zero
     * @param useUnsignedSymmetric Min/Max ranges are symmetric on an unsigned grid but not necessarily at 0
     * @param bitWidth The bitwidth used to compute the encoding
     * @return A valid encoding containing min, max, delta and offset values
     */
    TfEncoding computeEncoding(const std::string& tensorName, bool useSymmetricEncoding, int bitWidth = 0);

    /**
     * @brief Retrieves the encoding for all the tensors in graph. This function is expected
     *        to be called only after updateStats.
     * @param bws The bandwidth to be used to compute full encodings
     * @param[in/out] tensorEncodings A map of tensor name to encoding
     * @throws Exception if update stats has not been called
     */
    void getEncodings(const std::map<std::string, int>& bws, std::map<std::string, TfEncoding>& tensorEncodings,
                      bool useSymmetricEncodings);

    /**
     * @brief Returns a reference to a tensor quantizer reference associated with tensorName
     * @param tensorName The name of the tensor
     * @return A const shared pointer to a tensor quantizer
     */
    const std::shared_ptr<TensorQuantizer>& getTensorQuantizer(const std::string& tensorName) const;

    /**
     * @brief Checks if a tensor has accumulated min, max statistics
     * @param tensorName
     * @return True if tensorName has valid stats, false otherwise
     */

    bool hasValidStats(const std::string& tensorName) const;

    /**
     * @brief Checks if a tensor has accumulated min, max statistics
     * @param tensorName
     * @return True if tensorName has valid stats, false otherwise
     */

    bool isEncodingValid(const std::string& tensorName) const;


    /**
     * sets strict symmetric flag
     * @param bool, True if strict symmetric, False otherwise
     */
    void setStrictSymmetric(bool useStrictSymmetric, const std::string& tensorName = "");


    /**
     * sets unsigned symmetric flag
     * @param bool, True or False
     */
    void setUnsignedSymmetric(bool useUnsignedSymmetric, const std::string& tensorName = "");

    ~GraphQuantizer() = default;

private:
    ComputationMode _cpuGpuMode;
    std::vector<std::string> _tensorNames;
    std::map<std::string, std::shared_ptr<TensorQuantizer>> _tensorQuantizerActsMap;
    QuantizationMode _quantizationMode;   ///< Quantization scheme (e.g TF-Enhanced)
};


}   // namespace DlQuantization

#endif   // AIMET_GRAPH_QUANTIZER_H
