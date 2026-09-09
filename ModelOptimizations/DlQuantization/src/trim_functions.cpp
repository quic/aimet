// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "trim_functions.hpp"
#include "DlQuantization/Quantization.hpp"
#include "tensor_utils.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <cfenv>
#include <climits>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace DlQuantization
{
using namespace std;

// Minimum number of elements per parallel chunk. Tensors smaller than this
// threshold are processed sequentially to avoid thread-dispatch overhead.
static constexpr size_t MIN_ELEMENTS_PER_CHUNK = 1024;

namespace
{

// std::nearbyint follows the active floating-point rounding mode; set nearest-even for FP8 fake-cast.
class RoundToNearestEvenGuard
{
public:
    RoundToNearestEvenGuard() : _roundingMode(std::fegetround())
    {
        std::fesetround(FE_TONEAREST);
    }

    ~RoundToNearestEvenGuard()
    {
        std::fesetround(_roundingMode);
    }

private:
    int _roundingMode;
};

template <typename DTYPE>
DTYPE fakeCastToFp8(DTYPE value, const FloatQuantizationSpec& fp8Spec)
{
    const DTYPE maxValue = static_cast<DTYPE>(fp8Spec.maxValue);

    value = std::clamp(value, -maxValue, maxValue);

    // frexp yields value = mantissa * 2**exp with |mantissa| in [0.5, 1), so
    // floor(log2(|value|)) == exp - 1. Equivalent to log2/exp2 but roughly 1.7x faster,
    // since frexp/ldexp only manipulate the exponent field.
    int exponent;
    std::frexp(value, &exponent);
    exponent = std::max(exponent - 1, fp8Spec.exponentMin);

    // maxValue is exactly representable on the FP8 grid, so rounding a value already
    // clamped to [-maxValue, maxValue] can never step outside it: no post-round clamp needed.
    const DTYPE step = std::ldexp(DTYPE(1), exponent - fp8Spec.mantissaBits);
    return std::nearbyint(value / step) * step;
}

}   // namespace

inline double randUniformCpu()
{
    return rand() / (RAND_MAX + static_cast<double>(1.0));
}

double computeDelta(double encodingMin, double encodingMax, double numSteps)
{
    double delta = (encodingMax - encodingMin) / numSteps;
    return delta;
}


double computeOffset(double encodingMin, double delta)
{
    double offset = round(encodingMin / delta);

    return offset;
}


template <class Lambda>
Lambda parallelize(const uint32_t number_of_threads, Lambda lambda)
{
    std::vector<std::thread> threads(number_of_threads);
    for (uint32_t i = 0; i < number_of_threads; ++i)
    {
        threads[i] = std::thread(lambda, i);
    }
    for (uint32_t i = 0; i < number_of_threads; ++i)
    {
        threads[i].join();
    }
    return lambda;
};


// encoding: TF: rounded
template <typename DTYPE>
void quantizeDequantize(const DTYPE* in, uint64_t cnt, const TfEncoding& encoding, DTYPE* out, ComputationMode mode_cpu_gpu,
                        RoundingMode rounding_mode, void* stream, IForLoopRunner* runner)
{
    switch (mode_cpu_gpu)
    {
    case COMP_MODE_CPU:
        quantizeDequantizeCpu(in, cnt, encoding, out, rounding_mode, runner);
        break;
    case COMP_MODE_GPU:
#ifdef GPU_QUANTIZATION_ENABLED
        quantizeDequantizeGpu(in, cnt, encoding, out, rounding_mode, stream);
#else
        throw runtime_error("Not compiled for GPU mode.");
#endif
        break;
    default:
        throw runtime_error("Unknown computation mode.");
        break;
    }
}

void quantizeDequantizeFp8(const float* in, uint64_t cnt, const TfEncoding& encoding, float* out,
                           const FloatQuantizationSpec& fp8Spec, ComputationMode modeCpuGpu, void* stream,
                           IForLoopRunner* runner)
{
    switch (modeCpuGpu)
    {
    case COMP_MODE_CPU:
        quantizeDequantizeFp8Cpu(in, cnt, encoding, out, fp8Spec, runner);
        break;
    case COMP_MODE_GPU:
#ifdef GPU_QUANTIZATION_ENABLED
        quantizeDequantizeFp8Gpu(in, cnt, encoding, out, fp8Spec, stream);
#else
        (void) stream;
        throw runtime_error("Not compiled for GPU mode.");
#endif
        break;
    default:
        throw runtime_error("Unknown computation mode.");
    }
}

// encoding: TF: rounded
template <typename DTYPE>
void quantizeToFxp(const DTYPE* in, uint64_t cnt, const TfEncoding& encoding, DTYPE* out, ComputationMode mode_cpu_gpu,
                   RoundingMode rounding_mode, bool shiftToSigned)
{
    switch (mode_cpu_gpu)
    {
    case COMP_MODE_CPU:
        quantizeToFxpCpu(in, cnt, encoding, out, rounding_mode, shiftToSigned);
        break;
    case COMP_MODE_GPU:
#ifdef GPU_QUANTIZATION_ENABLED
        quantizeToFxpGpu(in, cnt, encoding, out, rounding_mode, shiftToSigned);
#else
        throw runtime_error("Not compiled for GPU mode.");
#endif
        break;
    default:
        throw runtime_error("Unknown computation mode.");
        break;
    }
}

// CPU implementations

template <typename DTYPE>
inline void quantizeValueCpu(const DTYPE* in, DTYPE* out, DTYPE encoding_min, DTYPE encoding_max, DTYPE encoding_delta,
                             DTYPE encoding_offset, RoundingMode rounding_mode)
{
    *out = std::isnan(*in) ? encoding_min : *in;
    *out = fmax(fmin(*out, encoding_max), encoding_min);
    // Scale and add offset to get something in the range [0,2^bw-1]
    *out = *out / encoding_delta - encoding_offset;

    switch (rounding_mode)
    {
    case ROUND_NEAREST:
    {
        *out = round(*out);
        break;
    }
    case ROUND_STOCHASTIC:
    {
        *out = floor(*out + randUniformCpu());
        break;
    }
    default:
    {
        throw runtime_error("Unknown rounding mode.");
    }
    }
}

template <typename DTYPE>
inline void dequantizeValueCpu(DTYPE* out, DTYPE encoding_delta, DTYPE encoding_offset)
{
    *out = encoding_delta * (*out + encoding_offset);
}

template <typename DTYPE>
void quantizeDequantizeCpu(const DTYPE* in, uint64_t cnt, const TfEncoding& encoding, DTYPE* out,
                           RoundingMode rounding_mode, IForLoopRunner* runner)
{
    using EncType = QdqEncType<DTYPE>;

    auto qdqLoop = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i)
        {
            EncType val = in[i];
            quantizeValueCpu<EncType>(&val, &val, encoding.min, encoding.max, encoding.delta, encoding.offset,
                                      rounding_mode);
            dequantizeValueCpu<EncType>(&val, encoding.delta, encoding.offset);
            out[i] = static_cast<DTYPE>(val);
        }
    };

    if (runner && cnt > MIN_ELEMENTS_PER_CHUNK)
    {
        size_t numChunks = std::max<size_t>(1, cnt / MIN_ELEMENTS_PER_CHUNK);
        size_t chunkSize = (cnt + numChunks - 1) / numChunks;
        runner->run([&, chunkSize](size_t chunkId) {
            size_t start = chunkId * chunkSize;
            size_t end   = std::min(start + chunkSize, static_cast<size_t>(cnt));
            qdqLoop(start, end);
        }, numChunks);
        return;
    }

    qdqLoop(0, static_cast<size_t>(cnt));
}

void quantizeDequantizeFp8Cpu(const float* in, uint64_t cnt, const TfEncoding& encoding, float* out,
                              const FloatQuantizationSpec& fp8Spec, IForLoopRunner* runner)
{
    const float scale = static_cast<float>(encoding.delta);

    // Note: the rounding-mode guard is constructed inside the loop body rather than once
    // up front, because the floating-point rounding mode is thread-local and the body may
    // run on a worker thread.
    auto qdqLoop = [&](size_t start, size_t end) {
        RoundToNearestEvenGuard guard;
        for (size_t i = start; i < end; ++i)
        {
            const float scaled = in[i] / scale;
            out[i]             = fakeCastToFp8(scaled, fp8Spec) * scale;
        }
    };

    if (runner && cnt > MIN_ELEMENTS_PER_CHUNK)
    {
        size_t numChunks = std::max<size_t>(1, cnt / MIN_ELEMENTS_PER_CHUNK);
        size_t chunkSize = (cnt + numChunks - 1) / numChunks;
        runner->run([&, chunkSize](size_t chunkId) {
            size_t start = chunkId * chunkSize;
            size_t end   = std::min(start + chunkSize, static_cast<size_t>(cnt));
            qdqLoop(start, end);
        }, numChunks);
        return;
    }

    qdqLoop(0, static_cast<size_t>(cnt));
}


template <typename DTYPE>
void quantizeToFxpPacked(const DTYPE* in, uint64_t cnt, const TfEncoding& encoding, uint8_t* out, size_t out_size,
                         ComputationMode mode_cpu_gpu, RoundingMode rounding_mode, bool shiftToSigned)
{
    switch (mode_cpu_gpu)
    {
    case COMP_MODE_CPU:
        quantizeToFxpPackedCpu(in, cnt, encoding, out, out_size, rounding_mode, shiftToSigned);
        break;
    case COMP_MODE_GPU:
        throw runtime_error("GPU packed quantization not supported.");
        break;
    default:
        throw runtime_error("Unknown computation mode.");
        break;
    }
}
template <typename DTYPE>
void quantizeToFxpCpu(const DTYPE* in, uint64_t cnt, const TfEncoding& encoding, DTYPE* out, RoundingMode rounding_mode,
                      bool shiftToSigned)
{
    // Using unsigned int to account for case of signed symmetric 32 bit, when shift will be 2^31
    unsigned int shift = 0;
    if (shiftToSigned)
    {
        shift = pow(2, encoding.bw - 1);
    }
    for (uint64_t i = 0; i < cnt; ++i)
    {
        quantizeValueCpu<DTYPE>(&in[i], &out[i], encoding.min, encoding.max, encoding.delta, encoding.offset,
                                rounding_mode);
        out[i] -= shift;
    }
}

template <typename DTYPE>
void quantizeToFxpPackedCpu(const DTYPE* in, uint64_t cnt, const TfEncoding& encoding, uint8_t* out, size_t out_size,
                            RoundingMode rounding_mode, bool shiftToSigned)
{
    size_t min_out_size = ceil(max(encoding.bw, 8) * cnt / 8.0);
    if (out_size < min_out_size)
    {
        throw runtime_error("Out buffer is too small");
    }

    int number_of_threads = 4;   // determined by testing

#if 0
  if(encoding.bw < 8) {
    // Multi-threading not supported due to dependence between loop iterations
    number_of_threads = 1;
  }
#endif
    uint64_t iteration_per_threads = (uint64_t) ceil((double) cnt / (double) number_of_threads);
    auto quantize_job         = [&](int thread_id)
    {
        uint64_t start = thread_id * iteration_per_threads;
        uint64_t end   = std::min(start + iteration_per_threads, cnt);

        double data_quantized;
        for (uint64_t i = start; i < end; ++i)
        {
            // Saturate
            data_quantized = max(min((double) in[i], encoding.max), encoding.min);
            // Scale and add offset to get something in the range [0,2^bw-1]
            data_quantized = data_quantized / encoding.delta - encoding.offset;

            // Round
            switch (rounding_mode)
            {
            case ROUND_NEAREST:
            {
                data_quantized = round(data_quantized);
                break;
            }
            case ROUND_STOCHASTIC:
            {
                data_quantized = floor(data_quantized + randUniformCpu());
                break;
            }
            default:
            {
                throw runtime_error("Unknown rounding mode.");
                break;
            }
            }

            // Pack the data according to the target bit-width and symmetry
            if (!shiftToSigned)
            {
                switch (encoding.bw)
                {
                case 1:
                case 2:
                case 4:
                {
                    // Note: this case should not be parallelized because the OR operation introduces dependency
                    // between iterations
                    uint8_t* ptr          = &out[0];
                    uint8_t data_shrinked = (uint8_t) data_quantized;
// Currently unsupported packed case
#if 0
                      int bit_offset = encoding.bw * i;
                      if (bit_offset % 8 == 0) {
                        // zero-out buffer on first write to byte
                        ptr[bit_offset / 8] = 0;
                      }
                      // OR-in data_shrinked
                      ptr[bit_offset / 8] |= (data_shrinked << (bit_offset % 8));
            // Supported one value per byte
#else
                    ptr[i] = (uint8_t) max(min((double) data_shrinked, double(pow(2, encoding.bw) - 1)), 0.0);

#endif
                    break;
                }
                case 8:
                {
                    uint8_t* ptr = &out[0];
                    ptr[i]       = (uint8_t) max(min(data_quantized, double(UCHAR_MAX)), 0.0);
                    break;
                }
                case 16:
                {
                    uint16_t* ptr = (uint16_t*) &out[0];
                    ptr[i]        = (uint16_t) max(min(data_quantized, double(USHRT_MAX)), 0.0);
                    break;
                }
                case 32:
                {
                    uint32_t* ptr = (uint32_t*) &out[0];
                    ptr[i]        = (uint32_t) max(min(data_quantized, double(UINT_MAX)), 0.0);
                    break;
                }
                default:
                {
                    throw runtime_error("Bit-width needs to be power of two and "
                                        "between 1 and 32.");
                }
                }   // end of switch encoding.bw
            }       // end of if (shiftToSigned)
            else
            {
                // Using unsigned int to account for case of signed symmetric i.e in the case of bw = 8, it will be -127
                // to 127
                double shift = 0;
                if (shiftToSigned)
                {
                    shift = pow(2, encoding.bw - 1) - 1;
                }
                data_quantized -= shift;
                // Pack the data according to the target bit-width ...
                switch (encoding.bw)
                {
                case 1:
                case 2:
                case 4:
                {
                    int8_t* ptr          = (int8_t*) &out[0];
                    int8_t data_shrinked = (int8_t) data_quantized;
// Currently unsupported packed case
#if 0
              int bit_offset = encoding.bw * i;
              if (bit_offset % 8 == 0) ptr[bit_offset / 8] = 0; // zero-out buffer on first write to byte
              // OR-in data_shrinked
              ptr[bit_offset / 8] |= (data_shrinked * (int8_t)pow(2, (bit_offset % 8)));
#else
                    // Mask off the lower bw bits as a single byte
                    ptr[i] = data_shrinked & (int8_t) (pow(2, encoding.bw) - 1);
#endif
                    break;
                }
                case 8:
                {
                    int8_t* ptr = (int8_t*) &out[0];
                    ptr[i]      = (int8_t) max(min(data_quantized, double(SCHAR_MAX)), double(SCHAR_MIN));
                    break;
                }
                case 16:
                {
                    int16_t* ptr = (int16_t*) &out[0];
                    ptr[i]       = (int16_t) max(min(data_quantized, double(SHRT_MAX)), double(SHRT_MIN));
                    break;
                }
                case 32:
                {
                    int32_t* ptr = (int32_t*) &out[0];
                    ptr[i]       = (int32_t) max(min(data_quantized, double(INT_MAX)), double(INT_MIN));
                    break;
                }
                default:
                {
                    throw runtime_error("Bit-width needs to be power of two and "
                                        "between 1 and 32.");
                }
                }   // End of switch(encoding.bw).
            }
        }   // end of for loop
    };
    parallelize(number_of_threads, quantize_job);
}


template <typename DTYPE>
void dequantizeFromPackedFxp(const uint8_t* input, uint64_t cnt, const TfEncoding& encoding, DTYPE* output,
                             ComputationMode mode_cpu_gpu, bool shiftToSigned)
{
    switch (mode_cpu_gpu)
    {
    case COMP_MODE_CPU:
        dequantizeFromPackedFxpCpuMt(input, cnt, encoding, output, shiftToSigned);
        break;
    case COMP_MODE_GPU:
        throw runtime_error("GPU de-quantization not supported.");
        break;
    default:
        throw runtime_error("Unknown computation mode.");
        break;
    }
}

template <typename DTYPE>
void dequantizeFromPackedFxpCpuMt(const uint8_t* input, uint64_t cnt, const TfEncoding& encoding, DTYPE* output,
                                  bool shiftToSigned)
{
    int32_t num_threads = std::max(1, std::min((int32_t)(cnt / 120000), 4));
    uint64_t chunkSize   = cnt / num_threads;
    int32_t bw_adj      = encoding.bw / 8;

    if (cnt % num_threads)
    {
        // add one to distribute remainder size evenly
        chunkSize++;
    }
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i)
    {
        uint64_t chunkStart = chunkSize * i;
        uint64_t chunkEnd   = std::min(chunkStart + chunkSize, cnt);
        threads.push_back(std::thread(dequantizeFromPackedFxpCpu<DTYPE>, input + (chunkStart * bw_adj),
                                      chunkEnd - chunkStart, encoding, output + chunkStart, shiftToSigned));
    }
    std::for_each(threads.begin(), threads.end(), std::mem_fn(&thread::join));
}


template <typename DTYPE>
void dequantizeFromPackedFxpTfBitsCpu(const uint8_t* input, uint64_t cnt, const TfEncoding& encoding, DTYPE* output)
{
    double data_quantized;
    for (uint64_t i = 0; i < cnt; ++i)
    {
        // Extract next value from packed data stream
        // The packed data is unsigned in TF-style quantization
        uint64_t bit_offset = encoding.bw * i;
        uint32_t tmp   = input[bit_offset / 8] >> (bit_offset % 8);
        data_quantized = (double) (tmp & (uint32_t) ((1 << encoding.bw) - 1));

        // De-quantize the data and write it to output vector.
        output[i] = (encoding.delta * (data_quantized + encoding.offset));
    }
}

template <typename DTYPE>
void dequantizeFromPackedFxpTf8Cpu(const uint8_t* input, uint64_t cnt, const TfEncoding& encoding, DTYPE* output)
{
    for (uint64_t i = 0; i < cnt; ++i)
    {
        // De-quantize the data and write it to output vector.
        output[i] = (encoding.delta * ((double) input[i] + encoding.offset));
    }
}

template <typename DTYPE>
void dequantizeFromPackedFxpTf16Cpu(const uint16_t* input, uint64_t cnt, const TfEncoding& encoding, DTYPE* output)
{
    for (uint64_t i = 0; i < cnt; ++i)
    {
        // De-quantize the data and write it to output vector.
        output[i] = (encoding.delta * ((double) input[i] + encoding.offset));
    }
}

template <typename DTYPE>
void dequantizeFromPackedFxpTf32Cpu(const uint32_t* input, uint64_t cnt, const TfEncoding& encoding, DTYPE* output)
{
    for (uint64_t i = 0; i < cnt; ++i)
    {
        // De-quantize the data and write it to output vector.
        output[i] = (encoding.delta * ((double) input[i] + encoding.offset));
    }
}

template <typename DTYPE>
void dequantizeFromPackedFxpSymmetricBitsCpu(const uint8_t* input, uint64_t cnt, const TfEncoding& encoding, DTYPE* output)
{
    double data_quantized;
    for (uint64_t i = 0; i < cnt; ++i)
    {
// Removed packed support, no current use case
#if 0
    // Extract next value from packed data stream
    // The packed data is signed for Qmn quantization
    int bit_offset = encoding.bw * i;
    int8_t* ptr = (int8_t*)input;
    // We need to extract a signed number from this byte. Take the byte,
    // shift it left until the MSB reaches the byte boundary, and shift it
    // down so the LSB reaches the byte boundary.
    int8_t tmp = ptr[bit_offset / 8] <<
                 (8 - bit_offset % 8 - encoding.bw);
    data_quantized = (double)(tmp >> (8 - encoding.bw));
#else
        int8_t* ptr = (int8_t*) input;
        // Mask the sign bit 2^(bw-1) and f negative apply to the upper MSB bits while retaining the
        // LSB for 2^(bw-1)-1. Eg 4bit # 0b00001011 is negative, and should become 0b11111011
        if (ptr[i] & (int8_t) pow(2, encoding.bw - 1))
        {
            data_quantized = ~((int8_t) pow(2, encoding.bw) - 1) | ptr[i];
        }
        else
        {
            data_quantized = ptr[i];
        }

#endif

        // De-quantize the data and write it to output vector.
        output[i] = (encoding.delta * (data_quantized + encoding.offset));
    }
}

template <typename DTYPE>
void dequantizeFromPackedFxpSymmetric8Cpu(const uint8_t* input, uint64_t cnt, const TfEncoding& encoding, DTYPE* output)
{
    for (uint64_t i = 0; i < cnt; ++i)
    {
        int8_t* ptr = (int8_t*) input;
        // De-quantize the data and write it to output vector.
        output[i] = (encoding.delta * ((double) ptr[i] + encoding.offset));
    }
}

template <typename DTYPE>
void dequantizeFromPackedFxpSymmetric16Cpu(const int16_t* input, uint64_t cnt, const TfEncoding& encoding, DTYPE* output)
{
    for (uint64_t i = 0; i < cnt; ++i)
    {
        // De-quantize the data and write it to output vector.
        output[i] = (encoding.delta * ((double) input[i] + encoding.offset));
    }
}

template <typename DTYPE>
void dequantizeFromPackedFxpSymmetric32Cpu(const int32_t* input, uint64_t cnt, const TfEncoding& encoding, DTYPE* output)
{
    for (uint64_t i = 0; i < cnt; ++i)
    {
        // De-quantize the data and write it to output vector.
        output[i] = (encoding.delta * ((double) input[i] + encoding.offset));
    }
}

template <typename DTYPE>
void dequantizeFromPackedFxpCpu(const uint8_t* input, uint64_t cnt, const TfEncoding& encoding, DTYPE* output,
                                bool shiftToSigned)
{
    if (!shiftToSigned)
    {
        // Unpacking the data is bit-width specific
        switch (encoding.bw)
        {
        case 1:
        case 2:
        case 4:
            // Removing packed support since there's no current use case
            // Fall through to standard unsigned tf8 dequant since there's no difference

            // DeQuantizeFromPackedFxpTfBitsCpu(input, cnt, encoding, output);
        case 8:
            dequantizeFromPackedFxpTf8Cpu(input, cnt, encoding, output);
            break;
        case 16:
            dequantizeFromPackedFxpTf16Cpu((const uint16_t*) input, cnt, encoding, output);
            break;
        case 32:
            dequantizeFromPackedFxpTf32Cpu((const uint32_t*) input, cnt, encoding, output);
            break;
        default:
        {
            throw runtime_error("Bit-width needs to be power of two and "
                                "between 1 and 32.");
        }
        }
    }
    else
    {
        // Unpacking the data is bit-width specific
        switch (encoding.bw)
        {
        case 1:
        case 2:
        case 4:
            dequantizeFromPackedFxpSymmetricBitsCpu(input, cnt, encoding, output);
            break;
        case 8:
            dequantizeFromPackedFxpSymmetric8Cpu(input, cnt, encoding, output);
            break;
        case 16:
            dequantizeFromPackedFxpSymmetric16Cpu((const int16_t*) input, cnt, encoding, output);
            break;
        case 32:
            dequantizeFromPackedFxpSymmetric32Cpu((const int32_t*) input, cnt, encoding, output);
            break;
        default:
        {
            throw runtime_error("Bit-width needs to be power of two and "
                                "between 1 and 32.");
        }
        }
    }
}

template <typename DTYPE>
void quantizeDequantizePerChannel(const DTYPE* in, int numChannel, int numElement, int numElementPerChannel, DTYPE* out,
                                  DTYPE* encodingMin, DTYPE* encodingMax, DTYPE* encodingDelta, DTYPE* encodingOffset,
                                  ComputationMode modeCpuGpu, RoundingMode roundingMode, void* stream)
{
    switch (modeCpuGpu)
    {
    case COMP_MODE_CPU:
        quantizeDequantizePerChannelCpu(in, numChannel, numElement, numElementPerChannel, out, encodingMin, encodingMax,
                                        encodingDelta, encodingOffset, roundingMode);
        break;
    case COMP_MODE_GPU:
#ifdef GPU_QUANTIZATION_ENABLED
        quantizeDequantizePerChannelGpu(in, numChannel, numElement, numElementPerChannel, out, encodingMin, encodingMax,
                                        encodingDelta, encodingOffset, roundingMode, stream);
#else
        throw runtime_error("Not compiled for GPU mode.");
#endif
        break;
    default:
        throw runtime_error("Unknown computation mode.");
        break;
    }
}


template <typename DTYPE>
void quantizeDequantizeBroadcastCpu(const DTYPE* in, DTYPE* out, const Encodings& encodings,
                                    int64_t numElement, const TensorDims& inputStrides,
                                    const TensorDims& encodingStrides, const TensorDims& inputShape,
                                    IForLoopRunner* runner)
{
    auto ndim = inputStrides.size();

    using EncType = QdqEncType<DTYPE>;

    auto qdqLoop = [&](size_t start, size_t end) {
        int64_t encodingIdx = 0;
        TensorDims coords(ndim);
        size_t remainder = start;
        for (size_t d = 0; d < ndim; d++)
        {
            coords[d] = remainder / inputStrides[d];
            remainder -= coords[d] * inputStrides[d];
            encodingIdx += encodingStrides[d] * coords[d];
        }

        for (size_t i = start; i < end; i++)
        {
            auto delta  = encodings[encodingIdx].delta;
            auto offset = encodings[encodingIdx].offset;
            auto min    = encodings[encodingIdx].min;
            auto max    = encodings[encodingIdx].max;

            EncType val = in[i];
            quantizeValueCpu<EncType>(&val, &val, min, max, delta, offset, ROUND_NEAREST);
            dequantizeValueCpu<EncType>(&val, delta, offset);
            out[i] = static_cast<DTYPE>(val);

            // Increment coords by 1 and find new encodingIdx
            for (int d = ndim - 1; d >= 0; d--)
            {
                coords[d]++;
                if (coords[d] < inputShape[d])
                {
                    encodingIdx += encodingStrides[d];
                    break;
                }
                else
                {
                    coords[d] = 0;
                    encodingIdx -= encodingStrides[d] * (inputShape[d] - 1);
                }

            }
        }
    };

    if (runner && numElement > static_cast<int64_t>(MIN_ELEMENTS_PER_CHUNK))
    {
        size_t numChunks = std::max<size_t>(1, static_cast<size_t>(numElement) / MIN_ELEMENTS_PER_CHUNK);
        size_t chunkSize = (static_cast<size_t>(numElement) + numChunks - 1) / numChunks;
        runner->run([&, chunkSize](size_t chunkId) {
            size_t start = chunkId * chunkSize;
            size_t end   = std::min(start + chunkSize, static_cast<size_t>(numElement));
            qdqLoop(start, end);
        }, numChunks);
        return;
    }

    qdqLoop(0, static_cast<size_t>(numElement));
}

void quantizeDequantizeFp8BroadcastCpu(const float* in, float* out, const Encodings& encodings,
                                       const FloatQuantizationSpec& fp8Spec, int64_t numElement,
                                       const TensorDims& inputStrides, const TensorDims& encodingStrides,
                                       const TensorDims& inputShape, IForLoopRunner* runner)
{
    auto ndim = inputStrides.size();

    // Note: the rounding-mode guard is constructed inside the loop body rather than once
    // up front, because the floating-point rounding mode is thread-local and the body may
    // run on a worker thread.
    auto qdqLoop = [&](size_t start, size_t end) {
        RoundToNearestEvenGuard guard;

        // Recover the encoding index for an arbitrary starting offset, so each chunk can
        // be processed independently.
        int64_t    encodingIdx = 0;
        TensorDims coords(ndim);
        size_t     remainder = start;
        for (size_t d = 0; d < ndim; d++)
        {
            coords[d] = remainder / inputStrides[d];
            remainder -= coords[d] * inputStrides[d];
            encodingIdx += encodingStrides[d] * coords[d];
        }

        for (size_t i = start; i < end; ++i)
        {
            const float scale  = static_cast<float>(encodings[encodingIdx].delta);
            const float scaled = in[i] / scale;
            out[i]             = fakeCastToFp8(scaled, fp8Spec) * scale;

            for (int d = ndim - 1; d >= 0; d--)
            {
                coords[d]++;
                if (coords[d] < inputShape[d])
                {
                    encodingIdx += encodingStrides[d];
                    break;
                }

                coords[d] = 0;
                encodingIdx -= encodingStrides[d] * (inputShape[d] - 1);
            }
        }
    };

    if (runner && numElement > static_cast<int64_t>(MIN_ELEMENTS_PER_CHUNK))
    {
        const size_t total     = static_cast<size_t>(numElement);
        const size_t numChunks = std::max<size_t>(1, total / MIN_ELEMENTS_PER_CHUNK);
        const size_t chunkSize = (total + numChunks - 1) / numChunks;
        runner->run([&, chunkSize](size_t chunkId) {
            const size_t start = chunkId * chunkSize;
            const size_t end   = std::min(start + chunkSize, total);
            qdqLoop(start, end);
        }, numChunks);
        return;
    }

    qdqLoop(0, static_cast<size_t>(numElement));
}

void quantizeDequantizeFp8Broadcast(const float* inTensor, float* outTensor, const Encodings& encodings,
                                    const FloatQuantizationSpec& fp8Spec, const TensorDims& inputShape,
                                    const TensorDims& encodingShape, ComputationMode mode, void* stream,
                                    IForLoopRunner* runner)
{
    auto numElements = getNumel(inputShape);

    auto bcShapes        = getBroadcastableShapes(inputShape, encodingShape);
    auto bcTensorShape   = std::get<0>(bcShapes);
    auto bcEncShape      = std::get<1>(bcShapes);
    auto inputStrides    = shapeToStrides(bcTensorShape);
    auto encodingStrides = shapeToStrides(bcEncShape);

    for (size_t idx = 0; idx < inputStrides.size(); idx++)
    {
        if (bcEncShape[idx] == 1 and bcTensorShape[idx] != 1)
        {
            encodingStrides[idx] = 0;
        }
    }

    switch (mode)
    {
    case COMP_MODE_CPU:
        quantizeDequantizeFp8BroadcastCpu(inTensor, outTensor, encodings, fp8Spec, numElements, inputStrides,
                                          encodingStrides, bcTensorShape, runner);
        break;
    case COMP_MODE_GPU:
#ifdef GPU_QUANTIZATION_ENABLED
        quantizeDequantizeFp8BroadcastGpu(inTensor, outTensor, encodings, fp8Spec, numElements, inputStrides,
                                          encodingStrides, stream);
#else
        (void) stream;
        throw std::runtime_error("Not compiled for GPU mode.");
#endif
        break;
    default:
        throw std::runtime_error("Unknown computation mode.");
    }
}


template <typename DTYPE>
void quantizeDequantizePerChannelCpu(const DTYPE* in, int numChannel, int numElement, int numElementPerChannel,
                                     DTYPE* out, DTYPE* encodingMin, DTYPE* encodingMax, DTYPE* encodingDelta,
                                     DTYPE* encodingOffset, RoundingMode roundingMode)
{
    for (uint64_t i = 0; i < numElement; ++i)
    {
        int channelIdx = (i / numElementPerChannel) % numChannel;
        quantizeValueCpu<DTYPE>(&in[i], &out[i], encodingMin[channelIdx], encodingMax[channelIdx],
                                encodingDelta[channelIdx], encodingOffset[channelIdx], roundingMode);
        dequantizeValueCpu<DTYPE>(&out[i], encodingDelta[channelIdx], encodingOffset[channelIdx]);
    }
}

template <typename T>
void quantizeDequantizeBroadcast(const T* inTensor, T* outTensor, const Encodings& encodings,
                                 const TensorDims& inputShape, const TensorDims& encodingShape, ComputationMode mode,
                                 void* stream, IForLoopRunner* runner)
{
    auto numElements = getNumel(inputShape);

    auto bcShapes        = getBroadcastableShapes(inputShape, encodingShape);
    auto bcTensorShape   = std::get<0>(bcShapes);
    auto bcEncShape      = std::get<1>(bcShapes);
    auto inputStrides    = shapeToStrides(bcTensorShape);
    auto encodingStrides = shapeToStrides(bcEncShape);

    // For broadcasting, encodingStrides = 0 over broadcast dimensions
    for (size_t idx = 0; idx < inputStrides.size(); idx++)
    {
        if (bcEncShape[idx] == 1 and bcTensorShape[idx] != 1)
        {
            encodingStrides[idx] = 0;
        }
    }

    switch (mode)
    {
    case COMP_MODE_GPU:
#ifdef GPU_QUANTIZATION_ENABLED
        quantizeDequantizeBroadcastGpu(inTensor, outTensor, encodings, numElements, inputStrides, encodingStrides,
                                       stream);
#else
        throw std::runtime_error("Not compiled for GPU mode.");
#endif
        break;
    case COMP_MODE_CPU:
        quantizeDequantizeBroadcastCpu(inTensor, outTensor, encodings, numElements, inputStrides, encodingStrides,
                                       bcTensorShape, runner);
        break;
    default:
        throw std::runtime_error("Unknown computation mode.");
    }
}


template void quantizeDequantize(const float* in, uint64_t cnt, const TfEncoding& encoding, float* out,
                                 ComputationMode mode_cpu_gpu, RoundingMode rounding_mode, void* stream,
                                 IForLoopRunner* runner);

template void quantizeToFxp(const float* in, uint64_t cnt, const TfEncoding& encoding, float* out,
                            ComputationMode mode_cpu_gpu, RoundingMode rounding_mode, bool shiftToSigned);

template void quantizeToFxpPacked(const float* in, uint64_t cnt, const TfEncoding& encoding, uint8_t* out, size_t out_size,
                                  ComputationMode mode_cpu_gpu, RoundingMode rounding_mode, bool shiftToSigned);
template void dequantizeFromPackedFxp(const uint8_t* input, uint64_t cnt, const TfEncoding& encoding, float* output,
                                      ComputationMode mode_cpu_gpu, bool shiftToSigned);

template void quantizeDequantizePerChannel(const float* in, int numChannel, int numElement, int numElementPerChannel,
                                           float* out, float* encodingMin, float* encodingMax, float* encodingDelta,
                                           float* encodingOffset, ComputationMode modeCpuGpu, RoundingMode roundingMode,
                                           void* stream);

template void quantizeDequantizeBroadcast(const float* inTensor, float* outTensor,
                                          const Encodings& encodings, const TensorDims& inputShape,
                                          const TensorDims& encodingShape, ComputationMode mode, void* stream,
                                          IForLoopRunner* runner);

template void quantizeDequantizeBroadcastCpu(const float* in, float* out, const Encodings& encodings,
                                             int64_t numElement, const TensorDims& inputStrides,
                                             const TensorDims& encodingStrides, const TensorDims& inputShape,
                                             IForLoopRunner* runner);

template void quantizeDequantize(const Eigen::half* in, uint64_t cnt, const TfEncoding& encoding, Eigen::half* out,
                                 ComputationMode mode_cpu_gpu, RoundingMode rounding_mode, void* stream,
                                 IForLoopRunner* runner);

template void quantizeDequantizeBroadcast(const Eigen::half* inTensor, Eigen::half* outTensor,
                                          const Encodings& encodings, const TensorDims& inputShape,
                                          const TensorDims& encodingShape, ComputationMode mode, void* stream,
                                          IForLoopRunner* runner);

template void quantizeDequantizeBroadcastCpu(const Eigen::half* in, Eigen::half* out, const Encodings& encodings,
                                             int64_t numElement, const TensorDims& inputStrides,
                                             const TensorDims& encodingStrides, const TensorDims& inputShape,
                                             IForLoopRunner* runner);

template void quantizeDequantize(const Eigen::bfloat16* in, uint64_t cnt, const TfEncoding& encoding,
                                 Eigen::bfloat16* out, ComputationMode mode_cpu_gpu, RoundingMode rounding_mode,
                                 void* stream, IForLoopRunner* runner);

template void quantizeDequantizeBroadcast(const Eigen::bfloat16* inTensor, Eigen::bfloat16* outTensor,
                                          const Encodings& encodings, const TensorDims& inputShape,
                                          const TensorDims& encodingShape, ComputationMode mode, void* stream,
                                          IForLoopRunner* runner);

template void quantizeDequantizeBroadcastCpu(const Eigen::bfloat16* in, Eigen::bfloat16* out,
                                             const Encodings& encodings, int64_t numElement,
                                             const TensorDims& inputStrides, const TensorDims& encodingStrides,
                                             const TensorDims& inputShape, IForLoopRunner* runner);

}   // End of namespace DlQuantization
