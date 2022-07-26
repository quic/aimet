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

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

#include "DlQuantization/GraphQuantizer.h"

namespace DlQuantization
{

using namespace std;


GraphQuantizer::GraphQuantizer(const std::vector<string>& tensorNames, ComputationMode modeCpuGpu,
                               QuantizationMode quantMode)
{
    _tensorNames      = tensorNames;
    _cpuGpuMode       = modeCpuGpu;
    _quantizationMode = quantMode;

    for (const auto& tensorName: tensorNames)
    {
        _tensorQuantizerActsMap[tensorName] =
            std::make_shared<TensorQuantizer>(_quantizationMode, RoundingMode::ROUND_NEAREST);
    }
}

void GraphQuantizer::updateStats(const string& tensorName, const float* tensor, std::size_t tensorSize)
{
    auto tQuant  = getTensorQuantizer(tensorName);
    bool useCuda = _cpuGpuMode == ComputationMode::COMP_MODE_GPU;
    tQuant->updateStats(tensor, tensorSize, useCuda);
}


void GraphQuantizer::resetEncodingStats(const std::string& tensorName)
{
    auto tQuant = getTensorQuantizer(tensorName);
    tQuant->resetEncodingStats();
}

const std::shared_ptr<TensorQuantizer>& GraphQuantizer::getTensorQuantizer(const std::string& tensorName) const
{
    if (_tensorQuantizerActsMap.find(tensorName) == _tensorQuantizerActsMap.end())
    {
        throw runtime_error("Unknown tensor name: " + tensorName);
    }
    return _tensorQuantizerActsMap.at(tensorName);
}

TfEncoding GraphQuantizer::computeEncoding(const std::string& tensorName, bool useSymmetricEncoding, int bitWidth)
{
    auto tQuant = getTensorQuantizer(tensorName);

    if (!tQuant->hasValidStats())
    {
        throw runtime_error("Tensor: " + tensorName + " has no valid statistics");
    }

    return tQuant->computeEncoding((uint8_t) bitWidth, useSymmetricEncoding);
}

bool GraphQuantizer::hasValidStats(const std::string& tensorName) const
{
    return this->getTensorQuantizer(tensorName)->hasValidStats();
}

bool GraphQuantizer::isEncodingValid(const std::string& tensorName) const
{
    return this->getTensorQuantizer(tensorName)->isEncodingValid;
}

void GraphQuantizer::getEncodings(const std::map<std::string, int>& bws,
                                  std::map<std::string, TfEncoding>& tensorEncodings, bool useSymmetricEncodings)
{
    for (auto& bw: bws)
    {
        std::string tensorName      = bw.first;
        auto tQuant                 = getTensorQuantizer(tensorName);
        tensorEncodings[tensorName] = this->computeEncoding(tensorName, useSymmetricEncodings, bw.second);
    }
}
void GraphQuantizer::setStrictSymmetric(bool useStrictSymmetric, const std::string& tensorName)
{
    std::vector<std::string> doForAllTensors {};
    if (tensorName.empty())
    {
        doForAllTensors = _tensorNames;
    }
    else
    {
        doForAllTensors.push_back(tensorName);
    }

    std::for_each(doForAllTensors.begin(), doForAllTensors.end(),
                  [&](const std::string& tName)
                  { this->getTensorQuantizer(tName)->setStrictSymmetric(useStrictSymmetric); });
}
void GraphQuantizer::setUnsignedSymmetric(bool useUnsignedSymmetric, const std::string& tensorName)
{
    std::vector<std::string> doForAllTensors {};
    if (tensorName.empty())
    {
        doForAllTensors = _tensorNames;
    }
    else
    {
        doForAllTensors.push_back(tensorName);
    }

    std::for_each(doForAllTensors.begin(), doForAllTensors.end(),
                  [&](const std::string& tName)
                  { this->getTensorQuantizer(tName)->setUnsignedSymmetric(useUnsignedSymmetric); });
}


}   // End of namespace DlQuantization
