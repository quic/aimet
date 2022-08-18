# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

import unittest
import numpy as np
import aimet_common.libpymo as libpymo


class TestTensorQuantizer(unittest.TestCase):
    """
    Temporarily skipping this test since we removed the numpy interfaces to TensorQuantizer
    Need to revive when we create a Numpy-specific facade to TensorQuantizer
    """

    def get_delta(self, n_bits, x_min, x_max):
        """
        helper to calculate delta
        :return:
        """
        n_levels = 2 ** n_bits
        delta = float(x_max - x_min) / (n_levels - 1)  #scaling

        if delta < 1e-8:
            print('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
            delta = 1e-8

        return delta

    def get_offset_qat(self, x_min, delta):
        """
        helper to calculate offset
        :param x_min:
        :param delta:
        :return:
        """

        offset = np.round(-x_min / delta)
        return offset

    def qat_python_asymmetric_quantizer(self, x, n_bits, x_max, x_min):
        """
        Python implementation for asymmetric quantization for range learning.
        :param x:
        :param n_bits:
        :param x_max:
        :param x_min:
        :return:
        """
        scaling = self.get_delta(n_bits, x_min, x_max)
        offset = self.get_offset_qat(x_min, scaling) #zero_point

        x = np.where(x <= x_max, x, x_max)
        x = np.where(x >= x_min, x, x_min)
        x_int = x/scaling + offset
        x_int = np.round(x_int)
        x_float_q = (x_int - offset) * scaling

        return x_float_q

    def set_quantizer_values(self, quantizer, x_min, x_max):
        """
        helper to update quantizer encoding values
        :param quantizer:
        :param x_min:
        :param x_max:
        :return:
        """

        quantizer.isEncodingValid = True

    @unittest.skip
    def test_sanity(self):
        quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                                            libpymo.RoundingMode.ROUND_NEAREST)

        np.random.seed(10)
        random_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

        self.assertFalse(quantizer.isEncodingValid)
        quantizer.updateStats(random_input, False)
        self.assertFalse(quantizer.isEncodingValid)

        encoding = quantizer.computeEncoding(8, False, False, False)
        print(quantizer.encoding.min, quantizer.encoding.max, quantizer.encoding.delta, quantizer.encoding.offset)
        self.assertTrue(quantizer.isEncodingValid)

        self.assertEqual(quantizer.quantScheme, libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED)
        self.assertEqual(quantizer.roundingMode, libpymo.RoundingMode.ROUND_NEAREST)

        input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)
        output_tensor = np.zeros((1, 3, 224, 224)).astype(np.float32)

        quantizer.quantizeDequantize(input_tensor, output_tensor, encoding.min, encoding.max, 8, False)

        # Check that the output tensor did get updated
        self.assertFalse(np.all(output_tensor == 0))

        # Check that the quantized tensor is close to the input tensor but not the same
        self.assertTrue(np.allclose(output_tensor, input_tensor, atol=0.2))
        self.assertFalse(np.allclose(output_tensor, input_tensor, atol=0.1))

    def test_compare_qat_qc_quantize(self):
        """
        compare qat asymmetric quantization with  qc quantize implementation
        :return:
        """

        quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF,
                                            libpymo.RoundingMode.ROUND_NEAREST)

        np.random.seed(11)
        # random_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        random_input = 5 * (np.random.normal(size=[1,
                                              3,
                                              224, 224])) + 20

        # Full range min, max  (no scaling input)
        x_min = min(0., float(random_input.min()))
        x_max = max(0., float(random_input.max()))

        x_min = min(x_min, 0)
        x_max = max(x_max, 0)

        #  qc quantize
        self.set_quantizer_values(quantizer, x_min, x_max)
        # print(quantizer.encoding.min, quantizer.encoding.max, quantizer.encoding.delta, quantizer.encoding.offset)

        # aimet quantizer
        output_tensor = np.zeros((1, 3, 224, 224)).astype(np.float32)
        quantizer.quantizeDequantize(random_input, output_tensor, x_min, x_max, 8, False)

        # qat asymmenteic quantizer output as float32
        x_quant = self.qat_python_asymmetric_quantizer(random_input, 8, x_max, x_min).astype(np.float32)

        # compare qc quantize output and qat asymmetric quantizer output
        self.assertTrue(np.allclose(x_quant, output_tensor))

    def test_compare_qat_qc_quantize_two_dims(self):
        """
        compare qat asymmetric quantization with  qc quantize implementation for a 2D tensor
        :return:
        """

        quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF,
                                            libpymo.RoundingMode.ROUND_NEAREST)

        np.random.seed(11)
        random_input = 5 * (np.random.normal(size=[1, 3 * 224 * 224])) + 20

        # Full range min, max  (no scaling input)
        x_min = min(0., float(random_input.min()))
        x_max = max(0., float(random_input.max()))

        x_min = min(x_min, 0)
        x_max = max(x_max, 0)

        #  qc quantize
        self.set_quantizer_values(quantizer, x_min, x_max)
        # print(quantizer.encoding.min, quantizer.encoding.max, quantizer.encoding.delta, quantizer.encoding.offset)

        # aimet quantizer
        output_tensor = np.zeros((1, 3 * 224 * 224)).astype(np.float32)
        quantizer.quantizeDequantize(random_input, output_tensor, x_min, x_max, 8, False)

        # qat asymmetric quantizer output as float32
        x_quant = self.qat_python_asymmetric_quantizer(random_input, 8, x_max, x_min).astype(np.float32)

        # compare qc quantize output and qat asymmetric quantizer output
        self.assertTrue(np.allclose(x_quant, output_tensor))

    def test_compare_qat_qc_quantize_half_range_scaled_input(self):
        """
        compare qat asymmetric quantization with  qc quantize implementation
        :return:
        """

        quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF,
                                            libpymo.RoundingMode.ROUND_NEAREST)

        np.random.seed(10)

        # Half range min, max  (no scaling input)
        random_input = 10 * np.random.normal(size=[1,
                                              3,
                                              224, 224]) + 20

        x_min = min(0., 0.25*float(random_input.min()))
        x_max = max(0., 0.25*float(random_input.max()))

        x_min = min(x_min, 0)
        x_max = max(x_max, 0)

        #  qc quantize
        self.set_quantizer_values(quantizer, x_min, x_max)
        # print(quantizer.encoding.min, quantizer.encoding.max, quantizer.encoding.delta, quantizer.encoding.offset)

        # aimet quantizer
        output_tensor = np.zeros((1, 3, 224, 224)).astype(np.float32)
        quantizer.quantizeDequantize(random_input, output_tensor, x_min, x_max, 8, False)

        # qat asymmenteic quantizer output as float32
        x_quant = self.qat_python_asymmetric_quantizer(random_input, 8, x_max, x_min).astype(np.float32)

        # compare qc quantize output and qat asymmetric quantizer output
        self.assertTrue(np.allclose(x_quant, output_tensor))

    def test_compare_qat_qc_quantize_quarter_range_scaled_input(self):
        """
        compare qat asymmetric quantization with  qc quantize implementation
        :return:
        """

        quantizer = libpymo.TensorQuantizer(libpymo.QuantizationMode.QUANTIZATION_TF,
                                            libpymo.RoundingMode.ROUND_NEAREST)

        np.random.seed(1)
        random_input = 10 * np.random.normal(size=[1,
                                              3,
                                              224, 224]) - 20

        # 1/4 range min, max  (no scaling input)
        x_min = min(0., 0.5*float(random_input.min()))
        x_max = max(0., 0.5*float(random_input.max()))

        x_min = min(x_min, 0)
        x_max = max(x_max, 0)

        #  qc quantize
        self.set_quantizer_values(quantizer, x_min, x_max)
        # print(quantizer.encoding.min, quantizer.encoding.max, quantizer.encoding.delta, quantizer.encoding.offset)

        # aimet quantizer
        output_tensor = np.zeros((1, 3, 224, 224)).astype(np.float32)
        quantizer.quantizeDequantize(random_input, output_tensor, x_min, x_max, 8, False)

        # qat asymmenteic quantizer output as float32
        x_quant = self.qat_python_asymmetric_quantizer(random_input, 8, x_max, x_min).astype(np.float32)

        # compare qc quantize output and qat asymmetric quantizer output
        self.assertTrue(np.allclose(x_quant, output_tensor))

    def test_encoding_analyzer_with_numpy_interface(self):
        """
        compare qat asymmetric quantization with  qc quantize implementation
        :return:
        """

        np.random.seed(10)
        random_input = 5 * (np.random.normal(size=[1, 3, 224, 224])) + 2

        # Full range min, max  (no scaling input)
        x_min = np.min([0., random_input.min()])
        x_max = np.max([0., random_input.max()])
        delta = (x_max - x_min) / 255
        offset = np.round(x_min/delta)
        x_min = offset * delta
        x_max = x_min + 255 * delta


        enc_analyzer = libpymo.EncodingAnalyzerForPython(libpymo.QuantizationMode.QUANTIZATION_TF)
        enc_analyzer.updateStats(random_input, False)
        encoding, is_valid = enc_analyzer.computeEncoding(8, False, False, False)

        print("Encoding.min=", encoding.min)
        print("Encoding.max=", encoding.max)

        self.assertTrue(is_valid)
        self.assertAlmostEqual(x_min, encoding.min, places=5)
        self.assertAlmostEqual(x_max, encoding.max, places=5)


    def test_encoding_analyzer_with_numpy_interface_other_dimensions(self):
        """
        compare qat asymmetric quantization with  qc quantize implementation
        :return:
        """

        np.random.seed(10)
        random_input = 5 * (np.random.normal(size=[1, 3, 224, 224])) + 2

        # Full range min, max  (no scaling input)
        x_min = np.min([0., random_input.min()])
        x_max = np.max([0., random_input.max()])
        delta = (x_max - x_min) / 255
        offset = np.round(x_min/delta)
        x_min = offset * delta
        x_max = x_min + 255 * delta

        # 2-dimensional tensor
        random_input = random_input.reshape(3, -1)

        enc_analyzer = libpymo.EncodingAnalyzerForPython(libpymo.QuantizationMode.QUANTIZATION_TF)
        enc_analyzer.updateStats(random_input, False)
        encoding, is_valid = enc_analyzer.computeEncoding(8, False, False, False)

        print("Encoding.min=", encoding.min)
        print("Encoding.max=", encoding.max)

        self.assertTrue(is_valid)
        self.assertAlmostEqual(x_min, encoding.min, places=5)
        self.assertAlmostEqual(x_max, encoding.max, places=5)

        # 3-dimensional tensor
        random_input = random_input.reshape(3, 2, -1)

        enc_analyzer = libpymo.EncodingAnalyzerForPython(libpymo.QuantizationMode.QUANTIZATION_TF)
        enc_analyzer.updateStats(random_input, False)
        encoding, is_valid = enc_analyzer.computeEncoding(8, False, False, False)

        print("Encoding.min=", encoding.min)
        print("Encoding.max=", encoding.max)

        self.assertTrue(is_valid)
        self.assertAlmostEqual(x_min, encoding.min, places=5)
        self.assertAlmostEqual(x_max, encoding.max, places=5)

        # 1-dimensional tensor
        random_input = random_input.flatten()

        enc_analyzer = libpymo.EncodingAnalyzerForPython(libpymo.QuantizationMode.QUANTIZATION_TF)
        enc_analyzer.updateStats(random_input, False)
        encoding, is_valid = enc_analyzer.computeEncoding(8, False, False, False)

        print("Encoding.min=", encoding.min)
        print("Encoding.max=", encoding.max)

        self.assertTrue(is_valid)
        self.assertAlmostEqual(x_min, encoding.min, places=5)
        self.assertAlmostEqual(x_max, encoding.max, places=5)

        # 1-dimensional tensor
        random_input = random_input.reshape(2, 2, 2, 2, -1)

        enc_analyzer = libpymo.EncodingAnalyzerForPython(libpymo.QuantizationMode.QUANTIZATION_TF)
        enc_analyzer.updateStats(random_input, False)
        encoding, is_valid = enc_analyzer.computeEncoding(8, False, False, False)

        print("Encoding.min=", encoding.min)
        print("Encoding.max=", encoding.max)

        self.assertTrue(is_valid)
        self.assertAlmostEqual(x_min, encoding.min, places=5)
        self.assertAlmostEqual(x_max, encoding.max, places=5)
