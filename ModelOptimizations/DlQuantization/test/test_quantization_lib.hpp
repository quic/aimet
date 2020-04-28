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

#ifndef TEST_QUANTIZATION_LIB_HPP
#define TEST_QUANTIZATION_LIB_HPP

#include "gtest/gtest.h"

#include "DlQuantization/Quantization.hpp"
#include "math_functions.hpp"

#ifdef GPU_QUANTIZATION_ENABLED

#include "cuda_util.hpp"

#endif   // GPU_QUANTIZATION_ENABLED

// Definition of test template
typedef ::testing::Types<float, double> TestDtypes;

template <typename DataType>
struct CpuDevice
{
    typedef DataType dataType;
    static const DlQuantization::ComputationMode modeCpuGpu = DlQuantization::COMP_MODE_CPU;
};

#ifndef GPU_QUANTIZATION_ENABLED

typedef ::testing::Types<CpuDevice<float>, CpuDevice<double> > TestDataTypesAndDevices;

#else

template <typename DataType>
struct GpuDevice
{
    typedef DataType dataType;
    static const DlQuantization::ComputationMode modeCpuGpu = DlQuantization::COMP_MODE_GPU;
};

typedef ::testing::Types<CpuDevice<float>, CpuDevice<double>, GpuDevice<float>, GpuDevice<double> >
    TestDataTypesAndDevices;

#endif

/**
 * @brief This method should be called at runtime by gtests that require a GPU
 * with CUDA support.
 */
template <typename DataType>
bool CheckRunTest()
{
#ifndef GPU_QUANTIZATION_ENABLED
    return true;
#else
    if (DlQuantization::COMP_MODE_CPU == DataType::modeCpuGpu)
    {
        return true;
    }
    else
    {
        return DlQuantization::CudaSupportedHelper();
    }
#endif
}

// Move data between CPU and GPU memory
template <typename TestParams>
class Blob
{
    typedef typename TestParams::dataType DataType;

public:
    // Blob is initialized with data from CPU
    Blob(DataType* cpuDataPtr, int cnt)
    {
        _cpuDataPtr = (DataType*) malloc(cnt * sizeof(DataType));
        memcpy(_cpuDataPtr, cpuDataPtr, cnt * sizeof(DataType));
        _count       = cnt;
        _gpuDataPtr  = nullptr;
        _isDataOnGpu = false;
#ifdef GPU_QUANTIZATION_ENABLED
        if (DlQuantization::COMP_MODE_GPU == TestParams::modeCpuGpu)
        {
            _gpuDataPtr =
                (DataType*) DlQuantization::MemoryAllocation(DlQuantization::COMP_MODE_GPU, _count * sizeof(DataType));
        }
#endif
    }

    // Free all memory
    ~Blob()
    {
        free(_cpuDataPtr);
#ifdef GPU_QUANTIZATION_ENABLED
        if (DlQuantization::COMP_MODE_GPU == TestParams::modeCpuGpu)
        {
            DlQuantization::MemoryFree(DlQuantization::COMP_MODE_GPU, _gpuDataPtr);
        }
#endif
    }

    // Move data to CPU memory, is necessary, and return data pointer.
    DataType* getDataPtrOnCpu()
    {
#ifdef GPU_QUANTIZATION_ENABLED
        if (_isDataOnGpu)
        {
            DlQuantization::CudaMemCpy(_cpuDataPtr, _gpuDataPtr, _count * sizeof(DataType),
                                       DlQuantization::CudaMemcpyDirection::DEVICE_TO_HOST);
            _isDataOnGpu = false;
        }
#endif
        return _cpuDataPtr;
    }

    // Move data to test device, and return data pointer.
    // In CPU mode, the CPU is test device. In GPU mode, the GPU is test device.
    DataType* getDataPtrOnDevice()
    {
#ifdef GPU_QUANTIZATION_ENABLED
        // Case device == CPU
        if (DlQuantization::COMP_MODE_CPU == TestParams::modeCpuGpu)
        {
            return _cpuDataPtr;
            // Case device == GPU
        }
        else
        {
            if (!_isDataOnGpu)
            {
                DlQuantization::CudaMemCpy(_gpuDataPtr, _cpuDataPtr, _count * sizeof(DataType),
                                           DlQuantization::CudaMemcpyDirection::HOST_TO_DEVICE);
                memset(_cpuDataPtr, 0, _count * sizeof(DataType));
                _isDataOnGpu = true;
            }
            return _gpuDataPtr;
        }
#else
        return _cpuDataPtr;
#endif
    }

private:
    DataType* _cpuDataPtr;
    DataType* _gpuDataPtr;
    // The number of data points to hold in memory.
    int _count;
    bool _isDataOnGpu;
};

#endif   // TEST_QUANTIZATION_LIB_HPP
