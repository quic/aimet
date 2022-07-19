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


#include "quantization_utils.hpp"
#include "DlQuantization/Quantization.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "math_functions.hpp"
#ifdef GPU_QUANTIZATION_ENABLED
#include "cuda_util.hpp"
#endif


namespace DlQuantization
{
using namespace std;

TfEncoding getComputedEncodings(uint8_t bw, double min, double max, bool useSymmetricEncodings, bool useStrictSymmetric,
                                bool useUnsignedSymmetric)
{
    TfEncoding encoding;

    double numSteps = pow(2, bw) - 1;
    if (useSymmetricEncodings && useStrictSymmetric)
    {
        numSteps -= 1;
    }
    encoding.bw = bw;

    // Special case for symmetric encodings. If all values are positive or 0, we can treat the
    // symmetric encodings as unsigned, which essentially translates to asymmetric

    // This is a complex check: here is the explanation
    // If min < 0, then unsigned symmetric mode is immaterial
    // Also if user can explicitly requested to disable unsigned-symmetric mode, then we use regular symmetric
    if (useSymmetricEncodings && ((min < 0.0) || (!useUnsignedSymmetric)))
    {
        // If we desire symmetric encodings then we need to expand either the min or max to be mirrors of each other
        // centered around 0
        max                           = std::max(std::abs(max), std::abs(min));
        unsigned int numPositiveSteps = std::floor(numSteps / 2);
        encoding.delta                = max / numPositiveSteps;
        encoding.offset               = -std::ceil(numSteps / 2);
        encoding.min                  = encoding.offset * encoding.delta;
        encoding.max                  = encoding.delta * numPositiveSteps;
    }
    else
    {
        // Unsigned symmetric handling is the same as asymmetric from this point forward

        encoding.delta = (max - min) / numSteps;
        if (min < 0 && max > 0)
        {
            // Need to make sure 0-value is exactly quantizable
            // Quantization of q into b is given by:
            //     b = q / delta - offset, where
            //                             delta = (max - min)/#steps
            //                             offset = min / delta
            // For q = 0: b = -min / delta
            // Find the closest round b, and set q=0 for it
            double bZero    = round(-min / encoding.delta);
            bZero           = std::min(numSteps, std::max(0.0, bZero));   // just to be safe
            encoding.offset = -bZero;
        }
        else
        {
            // One of min or max is guaranteed to be zero, so 0 is exactly quantizable already
            encoding.offset = round(min / encoding.delta);
        }

        // Calculate 'min' and 'max' based on 'delta' and 'offset'.
        // Note this min and max can vary from the one in 'stats'. This min and max
        // can really be represented with the integer offset.
        encoding.min = encoding.delta * encoding.offset;
        // We want to calculate: max = delta * numSteps + min.
        // To avoid numerical accuracy issues on Linaro, we simplify the math.
        encoding.max = max - min + encoding.min;
    }
    return encoding;
}

void gateMinMax(double& encodingMin, double& encodingMax)
{
    // Additional handling to retain zero in range
    // encodingMin can be at maximum 0.0
    encodingMin = std::min(encodingMin, 0.0);

    // encodingMax can be at minimum 0.0
    encodingMax = std::max(encodingMax, 0.0);

    // handle case where encodingMin == encodingMax
    encodingMax = std::max(encodingMax, encodingMin + EPSILON);
}

void computeMinMaxRangeFromDeltaOffset(uint8_t bw, TfEncoding& encoding, bool useSymmetricEncodings,
                                       bool useUnsignedSymmetric, bool useStrictSymmetric)
{
    auto origEncoding = encoding;
    if (encoding.bw == 0)
    {
        throw std::invalid_argument("Encodings must have a valid non-zero bitwidth");
    }

    if (origEncoding.min != 0 && origEncoding.max != 0)
    {
        throw std::invalid_argument("Encoding min and max must be zero to use this function");
    }

    if (origEncoding.delta == 0 && origEncoding.offset > 0)
    {
        throw std::invalid_argument("Encoding must have a valid non-zero delta/offset if min and max are zero");
    }


    auto numSteps = pow(2, bw) - 1;

    if (useSymmetricEncodings && useStrictSymmetric)
    {
        numSteps -= 1;
    }

    // set up min and max values
    // Note delta and offset are assumed to allow for zero to be quantizable
    encoding.min = encoding.offset * encoding.delta;

    // factor in symmetry into max calculation
    if (useSymmetricEncodings && ((encoding.min < 0.0) || (!useUnsignedSymmetric)))
    {
        auto numPositiveSteps = std::floor(numSteps / 2);
        encoding.max          = encoding.delta * numPositiveSteps;
    }
    else
    {
        encoding.max = encoding.delta * numSteps + encoding.min;
    }

    // check that min, max are not too close
    // It's unlikely but we can still gate if needed
    if ((encoding.max - encoding.min < EPSILON))
    {
        gateMinMax(encoding.min, encoding.max);
    }
}

void computeDeltaAndOffsetFromMinMax(uint8_t bw, TfEncoding& encoding, bool useSymmetricEncodings,
                                     bool useUnsignedSymmetric, bool useStrictSymmetric)
{
    auto origEncoding = encoding;
    if (encoding.bw == 0)
    {
        throw std::invalid_argument("Encodings must have a valid non-zero bitwidth");
    }

    if (origEncoding.delta != 0 && origEncoding.offset != 0)
    {
        throw std::invalid_argument("Encoding delta and offset must be zero to use this function");
    }

    // Compute delta and offset, which may also adjust min and max
    // Note min, max is retained
    encoding     = getComputedEncodings(bw, encoding.min, encoding.max, useSymmetricEncodings, useStrictSymmetric,
                                        useUnsignedSymmetric);
    encoding.min = origEncoding.min;
    encoding.max = origEncoding.max;
}

// Function to slice a tensor along an axis. Output shape will be the same for each slice.
template <typename DTYPE>
void slice(const DTYPE* input, const std::vector<uint32_t>& inputDim, int32_t axis,
           std::vector<std::vector<DTYPE>>& outputs, std::vector<uint32_t>& outputDim)
{
    // Account for negative axis
    uint32_t realAxis   = (axis >= 0) ? axis : inputDim.size() + axis;
    outputDim           = inputDim;
    outputDim[realAxis] = 1;

    // If input slice axis dimension size == 1, then it's already "sliced". Copy input->output as there is nothing to
    // slice
    uint32_t outputCnt = std::accumulate(outputDim.begin(), outputDim.end(), 1, std::multiplies<uint32_t>());
    if (inputDim[realAxis] == 1)
    {
        outputs.emplace_back(input, input + outputCnt);
        return;
    }

    // Add all the slices
    std::vector<uint32_t> slices;
    for (uint32_t i = 1; i < inputDim[realAxis]; ++i)
    {
        slices.push_back(i);
    }

    uint32_t sliceCnt = slices.size() + 1;

    // std::cout << "Slice axis: " << realAxis << std::endl;
    // std::cout << "# slices: " << sliceCnt << std::endl;

    outputs.resize(sliceCnt);
    for (uint32_t i = 0; i < outputs.size(); ++i)
    {
        outputs[i].resize(outputCnt);
    }

    // Compute input dim strides
    // For e.g. input dim = {6, 3, 4}
    // strides = {12, 4, 1}
    std::vector<uint32_t> inputDimStrides(inputDim.size());
    for (uint32_t i = inputDim.size(); i > 0; --i)
    {
        inputDimStrides[i - 1] = std::accumulate(inputDim.begin() + i, inputDim.end(), 1, std::multiplies<uint32_t>());
        // std::cout << "inputDimStrides[" << (i-1) << "]=" << inputDimStrides[i-1] << std::endl;
    }

    // Compute num of slice runs
    uint32_t numSliceRuns =
        std::accumulate(inputDim.begin(), inputDim.begin() + realAxis, 1, std::multiplies<uint32_t>());
    // std::cout << "No. of slice runs : " << numSliceRuns << std::endl;

    // Compute the distane to move during each run for a given slice.
    // It is the same for each slice.
    uint32_t sliceRunStride = (realAxis == 0) ? 0 : inputDimStrides[realAxis - 1];
    // std::cout << "Slice run stride : " <<  sliceRunStride << std::endl;

    typedef struct
    {
        uint32_t inputStartOffset;
        uint32_t size;
    } SliceInfo;

    std::vector<SliceInfo> sliceInfos;

    // Compute slice info for each slice, using slice points, axis and input dim
    for (uint32_t sliceIdx = 0; sliceIdx < slices.size() + 1; ++sliceIdx)
    {
        SliceInfo sinfo;

        if (sliceIdx == 0)   // First slice
        {
            sinfo.inputStartOffset = 0;
            sinfo.size             = slices[sliceIdx] * inputDimStrides[realAxis];
        }
        else if (sliceIdx == slices.size())   // Last slice
        {
            sinfo.inputStartOffset = inputDimStrides[realAxis] * slices[sliceIdx - 1];
            sinfo.size             = (inputDim[realAxis] - slices[sliceIdx - 1]) * inputDimStrides[realAxis];
        }
        else   // Middle slices
        {
            sinfo.inputStartOffset = inputDimStrides[realAxis] * slices[sliceIdx - 1];
            sinfo.size             = (slices[sliceIdx] - slices[sliceIdx - 1]) * inputDimStrides[realAxis];
        }
        sliceInfos.push_back(sinfo);
        // std::cout << "SliceInfo Idx " << sliceIdx << ", inputStartOffset: " << sinfo.inputStartOffset << ", size: "
        // << sinfo.size << std::endl;
    }

    for (uint32_t runIdx = 0; runIdx < numSliceRuns; ++runIdx)
    {
        for (uint32_t sliceIdx = 0; sliceIdx < sliceCnt; ++sliceIdx)
        {
            auto inPtr = input + sliceInfos[sliceIdx].inputStartOffset + runIdx * sliceRunStride;
            std::copy(inPtr, inPtr + sliceInfos[sliceIdx].size,
                      outputs[sliceIdx].data() + runIdx * sliceInfos[sliceIdx].size);
        }
    }
}

// Function to concatenate from slice along an axis. Output shape should be the same shape as the original input shape
// to slice.
template <typename DTYPE>
void concat(const std::vector<std::vector<DTYPE>>& inputs, const std::vector<uint32_t>& inputDim, int32_t axis,
            DTYPE* output, std::vector<uint32_t>& outputDim)
{
    uint32_t realAxis   = (axis >= 0) ? axis : inputDim.size() + axis;
    outputDim           = inputDim;
    outputDim[realAxis] = inputs.size();

    uint32_t a        = 0;
    uint32_t numUnits = 1;
    for (; a < realAxis; ++a)
    {
        numUnits = numUnits * inputDim[a];
    }

    uint32_t unitSize = 1;
    for (; a < inputDim.size(); ++a)
    {
        unitSize = unitSize * inputDim[a];
    }
    for (uint32_t i = 0; i < numUnits; ++i)
    {
        for (uint32_t u = 0; u < (uint32_t) inputs.size(); ++u)
        {
            const DTYPE* src = inputs[u].data() + unitSize * i;
            std::copy(src, src + unitSize, output);
            output += unitSize;
        }
    }
}

template void slice(const float* input, const std::vector<uint32_t>& inputDim, int32_t axis,
                    std::vector<std::vector<float>>& outputs, std::vector<uint32_t>& outputDim);
template void slice(const double* input, const std::vector<uint32_t>& inputDim, int32_t axis,
                    std::vector<std::vector<double>>& outputs, std::vector<uint32_t>& outputDim);
template void slice(const uint8_t* input, const std::vector<uint32_t>& inputDim, int32_t axis,
                    std::vector<std::vector<uint8_t>>& outputs, std::vector<uint32_t>& outputDim);

template void concat(const std::vector<std::vector<float>>& inputs, const std::vector<uint32_t>& inputDim, int32_t axis,
                     float* output, std::vector<uint32_t>& outputDim);
template void concat(const std::vector<std::vector<double>>& inputs, const std::vector<uint32_t>& inputDim,
                     int32_t axis, double* output, std::vector<uint32_t>& outputDim);
template void concat(const std::vector<std::vector<unsigned char>>& inputs, const std::vector<uint32_t>& inputDim,
                     int32_t axis, unsigned char* output, std::vector<uint32_t>& outputDim);

}   // End of namespace DlQuantization