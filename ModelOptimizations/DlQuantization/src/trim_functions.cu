// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "cuda_fp16.h"
#include "cuda_util.hpp"
#include "trim_functions.cuh"
#include "trim_functions.hpp"
#include <Eigen/Core>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace DlQuantization
{
namespace
{
void throwIfCudaError(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class ScopedCudaBuffer
{
public:
    explicit ScopedCudaBuffer(size_t count)
    {
        throwIfCudaError(cudaMalloc(reinterpret_cast<void**>(&_data), count * sizeof(T)), "cudaMalloc failed");
    }

    ~ScopedCudaBuffer()
    {
        if (_data)
        {
            cudaFree(_data);
        }
    }

    ScopedCudaBuffer(const ScopedCudaBuffer&)            = delete;
    ScopedCudaBuffer& operator=(const ScopedCudaBuffer&) = delete;

    T* data()
    {
        return _data;
    }

private:
    T* _data = nullptr;
};

float checkedFp8Scale(double scale)
{
    const float fp8Scale = static_cast<float>(scale);
    if (!(fp8Scale > 0.0f) || !std::isfinite(fp8Scale))
    {
        throw std::invalid_argument("FP8 quantize-dequantize scale must be positive and finite");
    }
    return fp8Scale;
}
}   // namespace

template <typename DTYPE, typename EncType>
__global__ void quantizeDequantizeKernel(const DTYPE* in, uint64_t cnt, DTYPE* out, EncType encoding_min,
                                         EncType encoding_max, EncType encoding_delta, EncType encoding_offset,
                                         RoundingMode rounding_mode)
{
    CUDA_KERNEL_LOOP(i, cnt)
    {
        EncType val = in[i];
        quantizeToFxpDevice<EncType>(&val, &val, encoding_min, encoding_max, encoding_delta, encoding_offset,
                                     rounding_mode, i);
        dequantizeFromFxpDevice<EncType>(&val, encoding_delta, encoding_offset);
        out[i] = static_cast<DTYPE>(val);
    }
}

template <typename DTYPE>
__global__ void quantizeToFxpKernel(const DTYPE* in, uint64_t cnt, DTYPE* out,
                                    DTYPE encoding_min, DTYPE encoding_max,
                                    DTYPE encoding_delta, DTYPE encoding_offset,
                                    RoundingMode rounding_mode, unsigned int shift)
{
    CUDA_KERNEL_LOOP(i, cnt)
    {
        quantizeToFxpDevice<DTYPE>(in + i, out + i,
                                   encoding_min, encoding_max,
                                   encoding_delta, encoding_offset,
                                   rounding_mode, i);
        *(out + i) -= shift;
    }
}

template <typename DTYPE>
__global__ void quantizeDequantizePerChannelKernel(const DTYPE* in, int numChannel, int numElement, int numElementPerChannel,
                                                   DTYPE* out, DTYPE* encodingMin, DTYPE* encodingMax, DTYPE* encodingDelta,
                                                   DTYPE* encodingOffset, RoundingMode roundingMode)
{
    CUDA_KERNEL_LOOP(i, numElement)
    {
        int channelIdx = (i / numElementPerChannel) % numChannel;
        quantizeToFxpDevice<DTYPE>(in + i, out + i,
                                   *(encodingMin + channelIdx), *(encodingMax + channelIdx),
                                   *(encodingDelta + channelIdx), *(encodingOffset + channelIdx),
                                   roundingMode, i);
        dequantizeFromFxpDevice<DTYPE>(out + i, *(encodingDelta + channelIdx), *(encodingOffset + channelIdx));
    }
}



template <typename DTYPE, typename EncType>
__global__ void quantizeDequantizeBroadcastKernel(const DTYPE* in, DTYPE* out, int64_t numElements, int64_t numDims,
                                                  const TensorDim* inputStrides, const TensorDim* encodingStrides,
                                                  const EncType* encodingMin, const EncType* encodingMax,
                                                  const EncType* encodingDelta, const EncType* encodingOffset)
{
    CUDA_KERNEL_LOOP(i, numElements)
    {
        int encodingIdx = 0;
        int remainder   = i;
        for (auto dim = 0; dim < numDims; dim++)
        {
            int dimIdx = remainder / inputStrides[dim];
            remainder = remainder - dimIdx * inputStrides[dim];
            // encodingStrides will be 0 along broadcast dimensions
            encodingIdx += encodingStrides[dim] * dimIdx;
        }

        auto delta  = *(encodingDelta + encodingIdx);
        auto offset = *(encodingOffset + encodingIdx);
        auto min    = *(encodingMin + encodingIdx);
        auto max    = *(encodingMax + encodingIdx);

        EncType val = in[i];
        quantizeToFxpDevice<EncType>(&val, &val, min, max, delta, offset, ROUND_NEAREST, i);
        dequantizeFromFxpDevice<EncType>(&val, delta, offset);
        out[i] = static_cast<DTYPE>(val);
    }
}


template <typename DTYPE>
void quantizeDequantizeGpu(const DTYPE* in, uint64_t cnt, const TfEncoding& encoding, DTYPE* out, RoundingMode rounding_mode,
                           void* stream)
{
    using EncType = QdqEncType<DTYPE>;
    quantizeDequantizeKernel<DTYPE, EncType>
        <<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            in, cnt, out, encoding.min, encoding.max, encoding.delta, encoding.offset, rounding_mode);
}


__global__ void quantizeDequantizeFp16Kernel(const float* in, uint64_t cnt, float* out)
{
    CUDA_KERNEL_LOOP(i, cnt)
    {
        *(out + i) = __half2float(__float2half(*(in + i)));
    }
}


void quantizeDequantizeFp16ForGPU(const float* in, uint64_t cnt, float* out, void* stream)
{
    quantizeDequantizeFp16Kernel<<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        in, cnt, out);
}

__global__ void convertFloatToFp16Kernel(const float* in, uint64_t cnt, __half* out)
{
    CUDA_KERNEL_LOOP(i, cnt)
    {
        *(out + i) = __float2half(*(in + i));
    }
}


void convertFloatToFp16KernelForGPU(const float* in, uint64_t cnt, __half* out, void* stream)
{
    convertFloatToFp16Kernel<<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        in, cnt, out);
}


template <typename DTYPE>
void quantizeToFxpGpu(const DTYPE* in, uint64_t cnt, const TfEncoding& encoding,
                      DTYPE* out, RoundingMode rounding_mode, bool shiftToSigned)
{
    unsigned int shift = 0;
    if (shiftToSigned) {
        shift = pow(2, encoding.bw - 1);
    }
    quantizeToFxpKernel<DTYPE><<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS>>>(
            in, cnt, out, encoding.min, encoding.max, encoding.delta,
            encoding.offset, rounding_mode, shift);
}

template <typename DTYPE>
void quantizeDequantizePerChannelGpu(const DTYPE* in, int numChannel, int numElement, int numElementPerChannel,
                                     DTYPE* out, DTYPE* encodingMin, DTYPE* encodingMax, DTYPE* encodingDelta,
                                     DTYPE* encodingOffset, RoundingMode roundingMode, void* stream)
{
    quantizeDequantizePerChannelKernel<DTYPE>
        <<<CUDA_NUM_BLOCKS(numElement), CUDA_NUM_THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            in, numChannel, numElement, numElementPerChannel, out, encodingMin, encodingMax, encodingDelta,
            encodingOffset, roundingMode);
}

template <typename DTYPE>
void quantizeDequantizeBroadcastGpu(const DTYPE* in, DTYPE* out, const Encodings& encodings, int64_t numElements,
                                    const TensorDims& inputStrides, const TensorDims& encodingStrides, void* stream)
{
    using EncType = QdqEncType<DTYPE>;

    int64_t numEncodings = encodings.size();
    int64_t numDims      = inputStrides.size();
    std::vector<EncType> encVec(4 * numEncodings);

    for (int i = 0; i < numEncodings; i++)
    {
        encVec[i]                    = encodings[i].min;
        encVec[numEncodings + i]     = encodings[i].max;
        encVec[2 * numEncodings + i] = encodings[i].delta;
        encVec[3 * numEncodings + i] = encodings[i].offset;
    }

    // Allocate device memory for strides and encodings
    TensorDim* stridesDevice;
    EncType* encodingVectorDevice;
    cudaMalloc((void**) &stridesDevice, 2 * numDims * sizeof(TensorDim));
    cudaMalloc((void**) &encodingVectorDevice, 4 * numEncodings * sizeof(EncType));

    // Send encoding information to device
    cudaMemcpyAsync(encodingVectorDevice, encVec.data(), 4 * numEncodings * sizeof(EncType), cudaMemcpyHostToDevice,
                    static_cast<cudaStream_t>(stream));

    // Send stride information to device
    TensorDim* strideBuffer[2 * numDims];
    memcpy(strideBuffer, inputStrides.data(), numDims * sizeof(TensorDim));
    memcpy(strideBuffer + numDims, encodingStrides.data(), numDims * sizeof(TensorDim));
    cudaMemcpyAsync(stridesDevice, strideBuffer, 2 * numDims * sizeof(TensorDim), cudaMemcpyHostToDevice,
                    static_cast<cudaStream_t>(stream));

    EncType* encodingMin    = encodingVectorDevice;
    EncType* encodingMax    = encodingVectorDevice + numEncodings;
    EncType* encodingDelta  = encodingVectorDevice + 2 * numEncodings;
    EncType* encodingOffset = encodingVectorDevice + 3 * numEncodings;

    TensorDim* dTensorStrides   = stridesDevice;
    TensorDim* dEncodingStrides = stridesDevice + numDims;

    quantizeDequantizeBroadcastKernel<DTYPE, EncType>
        <<<CUDA_NUM_BLOCKS(numElements), CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
            in, out, numElements, numDims, dTensorStrides, dEncodingStrides, encodingMin, encodingMax, encodingDelta,
            encodingOffset);

    // Free device memory for strides and encodings
    cudaFree(stridesDevice);
    cudaFree(encodingVectorDevice);
}


// Keep the device fake cast aligned with the CPU implementation.
__device__ inline float fakeCastToFp8Device(float value, const FloatQuantizationSpec fp8Spec)
{
    const float maxValue = static_cast<float>(fp8Spec.maxValue);

    // Preserve NaN to match the CPU std::clamp path.
    if (isnan(value))
    {
        return value;
    }
    if (value < -maxValue)
    {
        value = -maxValue;
    }
    else if (value > maxValue)
    {
        value = maxValue;
    }

    // frexpf yields value = mantissa * 2**exp with |mantissa| in [0.5, 1), so
    // floor(log2(|value|)) == exp - 1.
    int exponent;
    frexpf(value, &exponent);
    exponent = max(exponent - 1, fp8Spec.exponentMin);

    const float step = ldexpf(1.0f, exponent - fp8Spec.mantissaBits);
    return rintf(__fdiv_rn(value, step)) * step;
}

__global__ void quantizeDequantizeFp8Kernel(const float* in, uint64_t cnt, float* out, float scale,
                                            const FloatQuantizationSpec fp8Spec)
{
    CUDA_KERNEL_LOOP(i, cnt)
    {
        out[i] = fakeCastToFp8Device(__fdiv_rn(in[i], scale), fp8Spec) * scale;
    }
}

__global__ void quantizeDequantizeFp8BroadcastKernel(const float* in, float* out, int64_t numElements, size_t numDims,
                                                     const TensorDim* inputStrides, const TensorDim* encodingStrides,
                                                     const float* encodingDelta, const FloatQuantizationSpec fp8Spec)
{
    CUDA_KERNEL_LOOP(i, numElements)
    {
        size_t encodingIdx = 0;
        size_t remainder   = i;
        for (size_t dim = 0; dim < numDims; dim++)
        {
            const size_t inputStride = inputStrides[dim];
            const size_t dimIdx      = remainder / inputStride;
            remainder -= dimIdx * inputStride;
            // encodingStrides will be 0 along broadcast dimensions
            encodingIdx += encodingStrides[dim] * dimIdx;
        }

        const float scale = encodingDelta[encodingIdx];
        out[i]            = fakeCastToFp8Device(__fdiv_rn(in[i], scale), fp8Spec) * scale;
    }
}

void quantizeDequantizeFp8Gpu(const float* in, uint64_t cnt, const TfEncoding& encoding, float* out,
                              const FloatQuantizationSpec& fp8Spec, void* stream)
{
    const float scale = checkedFp8Scale(encoding.delta);

    quantizeDequantizeFp8Kernel<<<CUDA_NUM_BLOCKS(cnt), CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        in, cnt, out, scale, fp8Spec);
    throwIfCudaError(cudaGetLastError(), "FP8 quantize-dequantize kernel launch failed");
}

void quantizeDequantizeFp8BroadcastGpu(const float* in, float* out, const Encodings& encodings,
                                       const FloatQuantizationSpec& fp8Spec, int64_t numElements,
                                       const TensorDims& inputStrides, const TensorDims& encodingStrides, void* stream)
{
    const size_t numEncodings = encodings.size();
    const size_t numDims      = inputStrides.size();

    // FP8 needs only one scale per encoding.
    std::vector<float> deltas(numEncodings);
    for (size_t i = 0; i < numEncodings; i++)
    {
        deltas[i] = checkedFp8Scale(encodings[i].delta);
    }

    ScopedCudaBuffer<TensorDim> stridesDevice(2 * numDims);
    ScopedCudaBuffer<float>     deltasDevice(numEncodings);
    const auto                  cudaStream = static_cast<cudaStream_t>(stream);

    throwIfCudaError(cudaMemcpyAsync(deltasDevice.data(), deltas.data(), numEncodings * sizeof(float),
                                     cudaMemcpyHostToDevice, cudaStream),
                     "Copying FP8 scales to CUDA device failed");

    std::vector<TensorDim> strideBuffer(2 * numDims);
    memcpy(strideBuffer.data(), inputStrides.data(), numDims * sizeof(TensorDim));
    memcpy(strideBuffer.data() + numDims, encodingStrides.data(), numDims * sizeof(TensorDim));
    throwIfCudaError(cudaMemcpyAsync(stridesDevice.data(), strideBuffer.data(), 2 * numDims * sizeof(TensorDim),
                                     cudaMemcpyHostToDevice, cudaStream),
                     "Copying FP8 broadcast strides to CUDA device failed");

    quantizeDequantizeFp8BroadcastKernel<<<CUDA_NUM_BLOCKS(numElements), CUDA_NUM_THREADS, 0, cudaStream>>>(
        in, out, numElements, numDims, stridesDevice.data(), stridesDevice.data() + numDims, deltasDevice.data(),
        fp8Spec);
    throwIfCudaError(cudaGetLastError(), "FP8 broadcast quantize-dequantize kernel launch failed");

    // Temporary device metadata must remain alive until the kernel finishes.
    throwIfCudaError(cudaStreamSynchronize(cudaStream), "FP8 broadcast quantize-dequantize kernel failed");
}


template void quantizeDequantizeGpu(const float* in, uint64_t cnt, const TfEncoding& encoding, float* out,
                                    RoundingMode rounding_mode, void* stream);

template void quantizeToFxpGpu(const float* in, uint64_t cnt, const TfEncoding& encoding, float* out,
                               RoundingMode rounding_mode, bool shiftToSigned);

template void quantizeDequantizePerChannelGpu(const float* in, int numChannel, int numElement, int numElementPerChannel,
                                              float* out, float* encodingMin, float* encodingMax, float* encodingDelta,
                                              float* encodingOffset, RoundingMode roundingMode, void* stream);

template void quantizeDequantizeBroadcastGpu(const float* in, float* out, const Encodings& encodings,
                                             int64_t numElements, const TensorDims& inputStrides,
                                             const TensorDims& encodingStrides, void* stream);

template void quantizeDequantizeGpu(const Eigen::half* in, uint64_t cnt, const TfEncoding& encoding, Eigen::half* out,
                                    RoundingMode rounding_mode, void* stream);

template void quantizeDequantizeBroadcastGpu(const Eigen::half* in, Eigen::half* out, const Encodings& encodings,
                                             int64_t numElements, const TensorDims& inputStrides,
                                             const TensorDims& encodingStrides, void* stream);

template void quantizeDequantizeGpu(const Eigen::bfloat16* in, uint64_t cnt, const TfEncoding& encoding,
                                    Eigen::bfloat16* out, RoundingMode rounding_mode, void* stream);

template void quantizeDequantizeBroadcastGpu(const Eigen::bfloat16* in, Eigen::bfloat16* out,
                                             const Encodings& encodings, int64_t numElements,
                                             const TensorDims& inputStrides, const TensorDims& encodingStrides,
                                             void* stream);

}   // End of namespace DlQuantization
