//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2016-2021, Qualcomm Innovation Center, Inc. All rights reserved.
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


#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <stdlib.h>

#include "DlQuantization/Quantization.hpp"
#include "math_functions.hpp"

#ifdef GPU_QUANTIZATION_ENABLED

#include "cuda_util.hpp"

#endif

namespace DlQuantization
{
using namespace std;

template <typename DTYPE>
DTYPE GetMax(const DTYPE* data, int cnt, ComputationMode cpuGpuMode)
{
    switch (cpuGpuMode)
    {
    case COMP_MODE_CPU:
        return GetMax_cpu(data, cnt);
        break;
    case COMP_MODE_GPU:
#ifdef GPU_QUANTIZATION_ENABLED
        return GetMax_gpu(data, cnt);
#else
        throw runtime_error("Not compiled for GPU mode.");
#endif
        break;
    default:
        throw runtime_error("Unknown computation mode.");
        return 0;
        break;
    }
}

template <typename DTYPE>
DTYPE GetMin(const DTYPE* data, int cnt, ComputationMode cpuGpuMode)
{
    switch (cpuGpuMode)
    {
    case COMP_MODE_CPU:
        return GetMin_cpu(data, cnt);
    case COMP_MODE_GPU:
#ifdef GPU_QUANTIZATION_ENABLED
        return GetMin_gpu(data, cnt);
#else
        throw runtime_error("Not compiled for GPU mode.");
#endif
    default:
        throw runtime_error("Unknown computation mode.");
        return 0;
    }
}

double logBase2(double d)
{
    return log(d) / log(2);
}

void* MemoryAllocation(ComputationMode modeCpuGpu, size_t bytes)
{
    switch (modeCpuGpu)
    {
    case COMP_MODE_CPU:
        return MemoryAllocation_cpu(bytes);
        break;
    case COMP_MODE_GPU:
    {
#ifdef GPU_QUANTIZATION_ENABLED
        return MemoryAllocation_gpu(bytes);
#else
        throw runtime_error("Not compiled for GPU mode.");
#endif
    }
    break;
    default:
        throw runtime_error("Unknown computation mode.");
        break;
    }
}

void MemoryFree(ComputationMode modeCpuGpu, void* data)
{
    switch (modeCpuGpu)
    {
    case COMP_MODE_CPU:
        MemoryFree_cpu(data);
        break;
    case COMP_MODE_GPU:
    {
#ifdef GPU_QUANTIZATION_ENABLED
        MemoryFree_gpu(data);
#else
        throw runtime_error("Not compiled for GPU mode.");
#endif
    }
    break;
    default:
        throw runtime_error("Unknown computation mode.");
        break;
    }
}

void ElementwiseMult(ComputationMode modeCpuGpu, const float* in, size_t cnt, float factor, float* out)
{
    switch (modeCpuGpu)
    {
    case COMP_MODE_CPU:
        ElementwiseMult_cpu(in, cnt, factor, out);
        break;
    case COMP_MODE_GPU:
    {
#ifdef GPU_QUANTIZATION_ENABLED
        ElementwiseMult_gpu(in, cnt, factor, out);
#else
        throw runtime_error("Not compiled for GPU mode.");
#endif
    }
    break;
    default:
        throw runtime_error("Unknown computation mode.");
        break;
    }
}

void GemmFloat(ComputationMode modeCpuGpu, bool transposeB, size_t m, size_t n, size_t k, const float* A,
               const float* B, float* C)
{
    switch (modeCpuGpu)
    {
    case COMP_MODE_CPU:
        (void) modeCpuGpu;
        (void) transposeB;
        (void) m;
        (void) n;
        (void) k;
        (void) A;
        (void) B;
        (void) C;
        throw runtime_error("CPU mode not implemented yet.");
    case COMP_MODE_GPU:
    {
#ifdef GPU_QUANTIZATION_ENABLED
        bool result = GemmFloat_gpu(m, n, k, A, B, C, transposeB);
        if (!result)
        {
            throw runtime_error("CUDA GEMM failed.");
        }
#else
        throw runtime_error("Not compiled for GPU mode.");
#endif
    }
    break;
    default:
        throw runtime_error("Unknown computation mode.");
        break;
    }
}

template <typename DTYPE>
void UpdatePdf(const DTYPE* data, int cnt, ComputationMode mode_cpu_gpu, bool signed_vals, PDF& pdf)
{
    switch (mode_cpu_gpu)
    {
    case COMP_MODE_CPU:
        if (signed_vals)
        {
            UpdatePdfSigned_cpu(data, cnt, pdf);
        }
        else
        {
            UpdatePdfUnsigned_cpu(data, cnt, pdf);
        }
        break;
    case COMP_MODE_GPU:
    {
#ifdef GPU_QUANTIZATION_ENABLED
        // Fall back to CPU mode.
        DTYPE* data_h = (DTYPE*) malloc(sizeof(DTYPE) * cnt);
        CudaMemCpy(data_h, data, cnt * sizeof(DTYPE), CudaMemcpyDirection::DEVICE_TO_HOST);
        if (signed_vals)
        {
            UpdatePdfSigned_cpu(data_h, cnt, pdf);
        }
        else
        {
            UpdatePdfUnsigned_cpu(data_h, cnt, pdf);
        }
        free(data_h);
#else
        throw runtime_error("Not compiled for GPU mode.");
#endif
    }
    break;
    default:
        throw runtime_error("Unknown computation mode.");
        break;
    }
}

// CPU mode implementations

template <typename DTYPE>
DTYPE GetMax_cpu(const DTYPE* data, int cnt)
{
    DTYPE val = -numeric_limits<double>::max();
    for (int i = 0; i < cnt; ++i)
    {
        val = max(val, data[i]);
    }
    return val;
}

template <typename DTYPE>
DTYPE GetMin_cpu(const DTYPE* data, int cnt)
{
    DTYPE val = numeric_limits<double>::max();
    for (int i = 0; i < cnt; ++i)
    {
        val = min(val, data[i]);
    }
    return val;
}

void ElementwiseMult_cpu(const float* in, size_t cnt, float factor, float* out)
{
    for (unsigned int i = 0; i < cnt; ++i)
    {
        out[i] = in[i] * factor;
    }
}

void* MemoryAllocation_cpu(size_t bytes)
{
    return malloc(bytes);
}

void MemoryFree_cpu(void* data)
{
    free(data);
}

template <typename DTYPE>
void UpdatePdfSigned_cpu(const DTYPE* data, int cnt, PDF& pdf)
{
    // Check if we need to initialize the PDF
    if (0 == pdf.xLeft.size())
    {
        // Define the range over which we want to calculate the PDF.
        DTYPE min_val = GetMin(data, cnt, COMP_MODE_CPU);
        DTYPE max_val = GetMax(data, cnt, COMP_MODE_CPU);

        if ((min_val == 0) && (max_val == 0))
        {
            // Special case, we don't have a histogram initialized, but we have a zero tensor here
            // No point in trying to initialize the histogram using this
            return;
        }

        // Make sure we have a non-zero range.
        if (min_val == max_val)
        {
            max_val = std::max(max_val, min_val + (DTYPE) 0.01);
        }
        // Enlarge the range by factor 3, to be on the safe side.
        DTYPE center = (max_val + min_val) / 2;
        min_val      = center - 3 * (center - min_val);
        max_val      = center + 3 * (max_val - center);
        // Initialize the PDF's buckets.
        DTYPE bucket_size = (max_val - min_val) / PDF_SIZE;
        pdf.xLeft.resize(PDF_SIZE);
        for (int i = 0; i < PDF_SIZE; ++i)
        {
            pdf.xLeft[i] = min_val + i * bucket_size;
        }
        // Initialize the rest of the PDF structure.
        pdf.pdf.resize(PDF_SIZE);
        pdf.hist.resize(PDF_SIZE);
        pdf.iterations = 0;
    }

    // Create the histogram of this number distribution.
    // The histogram's range is min_val to max_val.
    DTYPE min_val     = pdf.xLeft[0];
    DTYPE bucket_size = pdf.xLeft[1] - pdf.xLeft[0];
    vector<DTYPE> pdf_this_iter(PDF_SIZE, 0);
    vector<DTYPE> hist_this_iter(PDF_SIZE, 0);
    // This offset is used to help map numbers to histogram buckets.
    DTYPE pdf_offset = min_val / bucket_size;
    // Go through all data points and add them to the histogram.
    for (int i = 0; i < cnt; ++i)
    {
        // Map a floating point number to the appropriate bucket.
        int index = round(data[i] / bucket_size - pdf_offset);
        // Add to histogram, if inside the histogram range.
        if (index >= 0 && index < PDF_SIZE)
        {
            pdf_this_iter[index] += 1;
            hist_this_iter[index] += 1;
        }
    }

    // Average this histogram into the average of all batches.
    for (int i = 0; i < PDF_SIZE; ++i)
    {
        // Convert histogram to probability density function.
        pdf_this_iter[i] /= cnt;
        // Average this PDF into the running average.
        pdf.pdf[i] = (pdf.pdf[i] * pdf.iterations + pdf_this_iter[i]) / (pdf.iterations + 1);
        pdf.hist[i] += hist_this_iter[i];
    }
    pdf.iterations++;
}

template <typename DTYPE>
void UpdatePdfUnsigned_cpu(const DTYPE* data, int cnt, PDF& pdf)
{
    // Check if we need to initialize the PDF.
    if (0 == pdf.xLeft.size())
    {
        // Define the range over which we want to calculate the PDF.
        DTYPE min_val = GetMin(data, cnt, COMP_MODE_CPU);
        DTYPE max_val = GetMax(data, cnt, COMP_MODE_CPU);

        if ((min_val == 0) && (max_val == 0))
        {
            // Special case, we don't have a histogram initialized, but we have a zero tensor here
            // No point in trying to initialize the histogram using this
            return;
        }

        // Make sure we have a non-zero range.
        if (min_val == max_val)
        {
            max_val = std::max(max_val, min_val + (DTYPE) 0.01);
        }
        // Enlarge the range by factor 3, to be on the safe side.
        DTYPE center = (max_val + min_val) / 2;
        min_val      = center - 3 * (center - min_val);
        max_val      = center + 3 * (max_val - center);
        // Initialize the PDF's buckets.
        DTYPE max_abs_val = std::max(std::abs(max_val), std::abs(min_val));
        DTYPE bucket_size = max_abs_val / PDF_SIZE;
        pdf.xLeft.resize(PDF_SIZE);
        for (int i = 0; i < PDF_SIZE; ++i)
        {
            pdf.xLeft[i] = i * bucket_size;
        }
        // Initialize the rest of the PDF structure.
        pdf.pdf.resize(PDF_SIZE);
        pdf.hist.resize(PDF_SIZE);
        pdf.iterations = 0;
    }

    // Create the histogram of this number distribution.
    DTYPE bucket_size = pdf.xLeft[1] - pdf.xLeft[0];
    vector<DTYPE> pdf_this_iter(PDF_SIZE, 0);
    vector<DTYPE> hist_this_iter(PDF_SIZE, 0);
    // Go through all data points and add them to the histogram.
    for (int i = 0; i < cnt; ++i)
    {
        // Map a floating point number to the appropriate bucket.
        int index = round(std::abs(data[i]) / bucket_size);
        // Add to histogram, if inside the histogram range.
        if (index >= 0 && index < PDF_SIZE)
        {
            pdf_this_iter[index] += 1;
            hist_this_iter[index] += 1;
        }
    }

    // Average this histogram into the average of all batches.
    for (int i = 0; i < PDF_SIZE; ++i)
    {
        // Convert histogram to probability density function.
        pdf_this_iter[i] /= cnt;
        // Average this PDF into the running average.
        pdf.pdf[i] = (pdf.pdf[i] * pdf.iterations + pdf_this_iter[i]) / (pdf.iterations + 1);
        pdf.hist[i] = (pdf.hist[i] * pdf.iterations + hist_this_iter[i]) / (pdf.iterations + 1);
    }
    pdf.iterations++;
}

// Explicit instantiations
template double GetMax(const double* data, int cnt, ComputationMode mode_cpu_gpu);

template float GetMax(const float* data, int cnt, ComputationMode mode_cpu_gpu);

template double GetMin(const double* data, int cnt, ComputationMode mode_cpu_gpu);

template float GetMin(const float* data, int cnt, ComputationMode mode_cpu_gpu);

template void UpdatePdf(const double* data, int cnt, ComputationMode mode_cpu_gpu, bool signed_vals, PDF& pdf);

template void UpdatePdf(const float* data, int cnt, ComputationMode mode_cpu_gpu, bool signed_vals, PDF& pdf);

}   // End of namespace DlQuantization
