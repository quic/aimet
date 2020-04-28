# /usr/bin/env python2.7
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

import os
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
# Import Quantizer class to initialize the DlQuantization static object
import libpytrext
# Import the python bindings for the DlQuantization library
import libpymo
from aimet_common.utils import AimetLogger

_log = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

# import inspect  # For inspecting function signature for python wrappers


class QCQuantizeTest(tf.test.TestCase):

  qc_quantize_module = tf.load_op_library('libaimet_tf_ops.so')

  def _compute_encodings (self, min, max, bw):
    steps = (2**bw) - 1
    delta = (max - min) / steps
    offset = np.round(min / delta)
    new_min = delta * offset
    new_max = delta * steps + new_min
    tf_encoding = np.array(list([new_min, new_max, delta, offset]))
    return tf_encoding

  def testQCQuantize_Params(self):
    _log.info('running testQCQuantize_Params')
    for use_gpu in [False, True]:
      _log.info('GPU mode is selected') if use_gpu else _log.info('CPU mode is selected')
      with self.test_session(use_gpu=use_gpu):
        bw = 8
        PARAM_MIN = -50.0
        PARAM_MAX = 80.0
        comp_mode = libpymo.ComputationMode.COMP_MODE_GPU if use_gpu else libpymo.ComputationMode.COMP_MODE_CPU
        # Instantiate DlQuantization object
        libpytrext.InitQuantizer(["conv1"], comp_mode, [], libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED)

        weights = constant_op.constant([-40.0, -1.0, 0.0, 1.0, 2.0, -50.0, 80.0])

        # Quantize and de-quantize params
        test_output = self.qc_quantize_module.qc_quantize_deprecated(op_name='conv1',
                                                                     training_in_progress=False,
                                                                     config=int(libpytrext.config_type.CONFIG_TYPE_Q_DQ_PARAMS),
                                                                     bitwidth=bw, in_tensors=[weights],
                                                                     fixed_enc_mins=[], fixed_enc_maxs=[],
                                                                     num_tensors=1)
        quantized_weights = ops.convert_to_tensor(test_output[0]).eval()
        self.assertAllClose(quantized_weights[0], weights.eval(), 1.0)

        # Examine encodings of quantized params
        out_enc_min = ops.convert_to_tensor(test_output[1]).eval()
        out_enc_max = ops.convert_to_tensor(test_output[2]).eval()
        true_encodings = self._compute_encodings(out_enc_min[0], out_enc_max[0], bw)
        expected_encodings = self._compute_encodings(PARAM_MIN, PARAM_MAX, bw)
        error_margin = 10  # Use better heuristics; ideally there should be 0 error margin
        self.assertArrayNear(true_encodings, expected_encodings, error_margin)

        # Repeat test with training_in_progress == true
        test_output = self.qc_quantize_module.qc_quantize_deprecated(op_name='conv1',
                                                                     training_in_progress=True,
                                                                     config=int(libpytrext.config_type.CONFIG_TYPE_Q_DQ_PARAMS),
                                                                     bitwidth=bw, in_tensors=[weights],
                                                                     fixed_enc_mins=[], fixed_enc_maxs=[],
                                                                     num_tensors=1)
        quantized_weights = ops.convert_to_tensor(test_output[0]).eval()
        self.assertAllClose(quantized_weights[0], weights.eval(), 1.0)

        # Examine encodings of quantized params
        out_enc_min = ops.convert_to_tensor(test_output[1]).eval()
        out_enc_max = ops.convert_to_tensor(test_output[2]).eval()
        true_encodings = self._compute_encodings(out_enc_min[0], out_enc_max[0], bw)
        expected_encodings = self._compute_encodings(PARAM_MIN, PARAM_MAX, bw)
        error_margin = 10  # Use better heuristics; ideally there should be 0 error margin
        self.assertArrayNear(true_encodings, expected_encodings, error_margin)

        libpytrext.ResetQuantizer()

  def testQCQuantize_SingleActivation(self):
    _log.info('running testQCQuantize_SingleActivation')
    for use_gpu in [False, True]:
      _log.info('GPU mode is selected') if use_gpu else _log.info('CPU mode is selected')
      with self.test_session(use_gpu=use_gpu):
        bw = 8
        # Instantiate DlQuantization object
        comp_mode = libpymo.ComputationMode.COMP_MODE_GPU if use_gpu else libpymo.ComputationMode.COMP_MODE_CPU
        libpytrext.InitQuantizer(["conv1"], comp_mode, [], libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED)

        actvn_0 = constant_op.constant(np.arange(0, 20).astype(np.float32))
        actvn_1 = constant_op.constant(np.arange(0, 50).astype(np.float32))
        actvn_2 = constant_op.constant(np.arange(0, 100).astype(np.float32))
        # Update quantization stats
        output_0 = self.qc_quantize_module.qc_quantize_deprecated(op_name='conv1',
                                                                  training_in_progress=False,
                                                                  config=
                                                                  int(libpytrext.config_type.CONFIG_TYPE_UPDATE_STATS),
                                                                  bitwidth=bw, in_tensors=[actvn_0],
                                                                  fixed_enc_mins=[], fixed_enc_maxs=[])
        ops.convert_to_tensor(output_0[0]).eval()
        output_1 = self.qc_quantize_module.qc_quantize_deprecated(op_name='conv1',
                                                                  training_in_progress=False,
                                                                  config=
                                                                  int(libpytrext.config_type.CONFIG_TYPE_UPDATE_STATS),
                                                                  bitwidth=bw, in_tensors=[actvn_1],
                                                                  fixed_enc_mins=[], fixed_enc_maxs=[])
        ops.convert_to_tensor(output_1[0]).eval()
        output_2 = self.qc_quantize_module.qc_quantize_deprecated(op_name='conv1',
                                                                  training_in_progress=False,
                                                                  config=
                                                                  int(libpytrext.config_type.CONFIG_TYPE_UPDATE_STATS),
                                                                  bitwidth=bw, in_tensors=[actvn_2],
                                                                  fixed_enc_mins=[], fixed_enc_maxs=[])
        ops.convert_to_tensor(output_2[0]).eval()

        ACT_MIN = 0.0
        ACT_MAX = 16.0
        test_actvn = constant_op.constant([ACT_MAX]) # Single input activation
        # Quantize and de-quantize activations
        test_output = self.qc_quantize_module.qc_quantize_deprecated(op_name='conv1',
                                                                     training_in_progress=False,
                                                                     config=int(libpytrext.config_type.CONFIG_TYPE_Q_DQ_ACTIVATIONS),
                                                                     bitwidth=bw, in_tensors=[test_actvn],
                                                                     fixed_enc_mins=[], fixed_enc_maxs=[],
                                                                     num_tensors=1)
        quantized_acts = ops.convert_to_tensor(test_output[0]).eval()
        # Test output activations
        self.assertAllClose (quantized_acts[0], test_actvn.eval(), 1.0)

        # Test output encodings from quantizing activations.
        enc_min = ops.convert_to_tensor(test_output[1]).eval()
        enc_max = ops.convert_to_tensor(test_output[2]).eval()

        true_encodings = self._compute_encodings(enc_min[0], enc_max[0], bw)
        # Compare against encodings obtained from get_encoding()
        get_enc_tensor = self.qc_quantize_module.qc_quantize_deprecated(op_name='conv1',
                                                                        training_in_progress=False,
                                                                        config=
                                                                        int(libpytrext.config_type.CONFIG_TYPE_GET_ENCODING),
                                                                        bitwidth=bw, in_tensors=[[]],
                                                                        fixed_enc_mins=[], fixed_enc_maxs=[],
                                                                        num_tensors=1)

        exp_enc_min = ops.convert_to_tensor(get_enc_tensor[1]).eval()
        exp_enc_max = ops.convert_to_tensor(get_enc_tensor[2]).eval()
        expected_encodings = self._compute_encodings(exp_enc_min[0], exp_enc_max[0], bw)
        self.assertAllEqual (true_encodings, expected_encodings)

        libpytrext.ResetQuantizer()

  def testQCQuantize_MultipleActivations(self):

    _log.info('running testQCQuantize_MultipleActivations')
    for use_gpu in [False, True]:
      _log.info('GPU mode is selected') if use_gpu else _log.info('CPU mode is selected')
      with self.test_session(use_gpu=use_gpu):
        bw = 8
        actvn_stats_0 = actvn_stats_1 = actvn_stats_2 = actvn_stats_3 = constant_op.constant(
          np.arange(0, 100).astype(np.float32))

        # Instantiate DlQuantization object
        comp_mode = libpymo.ComputationMode.COMP_MODE_GPU if use_gpu else libpymo.ComputationMode.COMP_MODE_CPU
        libpytrext.InitQuantizer(["conv1"], comp_mode, [], libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED)

        # Update quantization stats
        output_0 = self.qc_quantize_module.qc_quantize_deprecated(op_name='conv1',
                                                                  training_in_progress=False,
                                                                  config=
                                                                  int(libpytrext.config_type.CONFIG_TYPE_UPDATE_STATS),
                                                                  bitwidth=bw,
                                                                  in_tensors=[actvn_stats_0, actvn_stats_1,
                                                                              actvn_stats_2, actvn_stats_3],
                                                                  fixed_enc_mins=[], fixed_enc_maxs=[])

        ops.convert_to_tensor(output_0[0]).eval()

        actvn_0 = constant_op.constant(np.arange(0, 10).astype(np.float32))
        actvn_1 = constant_op.constant(np.arange(10, 20).astype(np.float32))
        actvn_2 = constant_op.constant(np.arange(20, 30).astype(np.float32))
        actvn_3 = constant_op.constant(np.arange(30, 40).astype(np.float32))
        test_actvn = [actvn_0, actvn_1, actvn_2, actvn_3]
        # Quantize and de-quantize activations

        output_1 = self.qc_quantize_module.qc_quantize_deprecated(op_name='conv1',
                                                                  training_in_progress=False,
                                                                  config=
                                                                  int(libpytrext.config_type.CONFIG_TYPE_Q_DQ_ACTIVATIONS),
                                                                  bitwidth=bw,
                                                                  in_tensors=[actvn_0, actvn_1, actvn_2, actvn_3],
                                                                  fixed_enc_mins=[], fixed_enc_maxs=[],
                                                                  num_tensors=4)
        quantized_acts = ops.convert_to_tensor(output_1[0]).eval()

        quantization_error_margin = 1.0
        for index in np.arange(0, len(quantized_acts)):
          self.assertArrayNear(ops.convert_to_tensor(test_actvn[index]).eval(), quantized_acts[index],
                               quantization_error_margin)

        # Test output encodings
        enc_min = ops.convert_to_tensor(output_1[1]).eval()
        enc_max = ops.convert_to_tensor(output_1[2]).eval()

        # Compare against encodings obtained from get_encoding()
        get_enc_tensor = self.qc_quantize_module.qc_quantize_deprecated(op_name='conv1',
                                                                        training_in_progress=False,
                                                                        config=
                                                                        int(libpytrext.config_type.CONFIG_TYPE_GET_ENCODING),
                                                                        bitwidth=bw, in_tensors=[[]],
                                                                        fixed_enc_mins=[], fixed_enc_maxs=[],
                                                                        num_tensors=4)

        exp_enc_min = ops.convert_to_tensor(get_enc_tensor[1]).eval()
        exp_enc_max = ops.convert_to_tensor(get_enc_tensor[2]).eval()

        for index in np.arange(0, len(quantized_acts)):
          true_encodings = self._compute_encodings(enc_min[index], enc_max[index], bw)
          expected_encodings = self._compute_encodings(exp_enc_min[index], exp_enc_max[index], bw)
          error_margin = 1.0  # Not a fair test to compare TF with TF_ENHANCED, but works for now
          self.assertAllEqual(true_encodings, expected_encodings)

        libpytrext.ResetQuantizer()

  def testQCQuantize_GetEncodings(self):
    _log.info('running testQCQuantize_GetEncodings')
    for use_gpu in [False, True]:
      _log.info('GPU mode is selected') if use_gpu else _log.info('CPU mode is selected')
      with self.test_session(use_gpu=use_gpu):
        bw = 8
        # Prepare activation tensors
        ACT_MIN = -20.0
        ACT_MAX = 25.0
        actvn_1 = constant_op.constant([-10.0, -20.0, 25.0])
        actvn_2 = constant_op.constant([8.0, -19.0, 30.0])
        actvn_3 = constant_op.constant([12.0, -31.0, 35.4])

        # Instantiate DlQuantization object
        comp_mode = libpymo.ComputationMode.COMP_MODE_GPU if use_gpu else libpymo.ComputationMode.COMP_MODE_CPU
        libpytrext.InitQuantizer(["conv1"], comp_mode, [], libpymo.QuantizationMode.QUANTIZATION_TF)

        # Update stats
        output_0 = self.qc_quantize_module.qc_quantize_deprecated(op_name='conv1',
                                                                  training_in_progress=False,
                                                                  config=
                                                                  int(libpytrext.config_type.CONFIG_TYPE_UPDATE_STATS),
                                                                  bitwidth=bw, in_tensors=[actvn_1, actvn_2, actvn_3],
                                                                  fixed_enc_mins=[], fixed_enc_maxs=[])
        ops.convert_to_tensor(output_0[0]).eval()

        # Get encodings
        output_1 = self.qc_quantize_module.qc_quantize_deprecated(op_name='conv1',
                                                                  training_in_progress=False,
                                                                  config=
                                                                  int(libpytrext.config_type.CONFIG_TYPE_GET_ENCODING),
                                                                  bitwidth=bw, in_tensors=[[]],
                                                                  fixed_enc_mins=[], fixed_enc_maxs=[], num_tensors=3)

        enc_min = ops.convert_to_tensor(output_1[1]).eval()
        enc_max = ops.convert_to_tensor(output_1[2]).eval()

        true_encodings = self._compute_encodings(enc_min[0], enc_max[0], bw)
        expected_encodings = self._compute_encodings(ACT_MIN, ACT_MAX, bw)
        error_margin = 1e-5  # Use better heuristics
        self.assertArrayNear (true_encodings, expected_encodings, error_margin)

        libpytrext.ResetQuantizer()

  def testQCQuantize_SetEncodings(self):
    _log.info('running testQCQuantize_SetEncodings')
    for use_gpu in [False, True]:
      _log.info('GPU mode is selected') if use_gpu else _log.info('CPU mode is selected')
      with self.test_session(use_gpu=use_gpu):
        bw = 8
        # Instantiate DlQuantization object
        comp_mode = libpymo.ComputationMode.COMP_MODE_CPU if use_gpu else libpymo.ComputationMode.COMP_MODE_GPU
        libpytrext.InitQuantizer(["conv1"], comp_mode, [], libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED)

        # Set encodings
        # For the purpose of exact matches with expected encodings
        # we choose to avoid ranges excluding 0 as the TF algorithm
        # forces the encoding to include 0, thus differing from expected results.
        enc_min_list = [-10.0, -0.5, 0]
        enc_max_list = [100.0, 200.0, 160.0]

        output_0 = self.qc_quantize_module.qc_quantize_deprecated(op_name='conv1',
                                                                  training_in_progress=False,
                                                                  config=
                                                                  int(libpytrext.config_type.CONFIG_TYPE_SET_ENCODING),
                                                                  bitwidth=bw, in_tensors=[[]],
                                                                  fixed_enc_mins=enc_min_list,
                                                                  fixed_enc_maxs=enc_max_list)
        ops.convert_to_tensor(output_0[0]).eval()

        # Retrieve encodings from op and validate
        output_1 = self.qc_quantize_module.qc_quantize_deprecated(op_name='conv1',
                                                                  training_in_progress=False,
                                                                  config=int(libpytrext.config_type.CONFIG_TYPE_GET_ENCODING),
                                                                  bitwidth=bw, in_tensors=[[]],
                                                                  fixed_enc_mins=[], fixed_enc_maxs=[], num_tensors=3)

        get_enc_min = ops.convert_to_tensor(output_1[1]).eval()
        get_enc_max = ops.convert_to_tensor(output_1[2]).eval()
        for index in np.arange(0, len(enc_min_list)):
          actual_encodings = self._compute_encodings(get_enc_min[index], get_enc_max[index], bw)
          expected_encodings = self._compute_encodings(enc_min_list[index], enc_max_list[index], bw)
          self.assertAllEqual(actual_encodings, expected_encodings)

        libpytrext.ResetQuantizer()

# @testQCQuantize_CheckZeroRepresentation:
# Test examines if the TF quantization algorithm ensures the representation
# of 0 in the quantized space even when the original sample space is
# entirely positive or entirely negative.

  def testQCQuantize_CheckZeroRepresentation(self):
    _log.info('running testQCQuantize_CheckZeroRepresentation')
    for use_gpu in [False, True]:
      _log.info('GPU mode is selected') if use_gpu else _log.info('CPU mode is selected')
      with self.test_session(use_gpu=use_gpu):
        bw = 8
        # Test all negative ranges
        act_min = -8.0
        act_max = -5.0

        # Instantiate DlQuantization object
        comp_mode = libpymo.ComputationMode.COMP_MODE_GPU if use_gpu else libpymo.ComputationMode.COMP_MODE_CPU
        libpytrext.InitQuantizer(["conv1"], comp_mode, [], libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED)

        # Set encodings
        output_0 = self.qc_quantize_module.qc_quantize_deprecated(op_name='conv1',
                                                                  training_in_progress=False,
                                                                  config=int(libpytrext.config_type.CONFIG_TYPE_SET_ENCODING),
                                                                  bitwidth=bw, in_tensors=[[]],
                                                                  fixed_enc_mins=[act_min], fixed_enc_maxs=[act_max])
        ops.convert_to_tensor(output_0[0]).eval()

        # Get encodings from op and validate
        output_1 = self.qc_quantize_module.qc_quantize_deprecated(op_name='conv1',
                                                                  training_in_progress=False,
                                                                  config=
                                                                  int(libpytrext.config_type.CONFIG_TYPE_GET_ENCODING),
                                                                  bitwidth=bw, in_tensors=[[]],
                                                                  fixed_enc_mins=[], fixed_enc_maxs=[], num_tensors=1)

        enc_max = ops.convert_to_tensor(output_1[2]).eval()
        self.assertEqual(enc_max[0], 0.0)

        # Test all positive ranges
        act_min = 20.0
        act_max = 100.0

        # Set encodings
        output_0 = self.qc_quantize_module.qc_quantize_deprecated(op_name='conv1',
                                                                  training_in_progress=False,
                                                                  config=
                                                                  int(libpytrext.config_type.CONFIG_TYPE_SET_ENCODING),
                                                                  bitwidth=bw, in_tensors=[[]],
                                                                  fixed_enc_mins=[act_min], fixed_enc_maxs=[act_max])
        ops.convert_to_tensor(output_0[0]).eval()

        # Get encodings from op and validate
        output_1 = self.qc_quantize_module.qc_quantize_deprecated(op_name='conv1',
                                                                  training_in_progress=False,
                                                                  config=
                                                                  int(libpytrext.config_type.CONFIG_TYPE_GET_ENCODING),
                                                                  bitwidth=bw, in_tensors=[[]],
                                                                  fixed_enc_mins=[], fixed_enc_maxs=[], num_tensors=1)

        enc_min = ops.convert_to_tensor(output_1[1]).eval()
        self.assertEqual(enc_min[0], 0.0)

        libpytrext.ResetQuantizer()

  def test_QCQuantize_CheckInvalidConfig(self):
    _log.info('running test_QCQuantize_CheckInvalidConfig')
    for use_gpu in [False, True]:
      _log.info('GPU mode is selected') if use_gpu else _log.info('CPU mode is selected')
      with self.test_session(use_gpu = use_gpu):
        bw = 8

        # Instantiate DlQuantization object
        comp_mode = libpymo.ComputationMode.COMP_MODE_GPU if use_gpu else libpymo.ComputationMode.COMP_MODE_CPU
        libpytrext.InitQuantizer(["conv1"], comp_mode, [], libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED)

        # Feed an invalid configuration number
        output_0 = self.qc_quantize_module.qc_quantize_deprecated(op_name='conv1',
                                                                  training_in_progress=False,
                                                                  config=10,
                                                                  bitwidth=bw, in_tensors=[[]],
                                                                  fixed_enc_mins=[], fixed_enc_maxs=[], num_tensors=1)
        with self.assertRaises(errors_impl.InvalidArgumentError):
          ops.convert_to_tensor(output_0[0]).eval()

        libpytrext.ResetQuantizer()

  def testQCQuantize_CheckInvalidEncodings(self):
    _log.info('running testQCQuantize_CheckInvalidEncodings')
    for use_gpu in [False, True]:
      _log.info('GPU mode is selected') if use_gpu else _log.info('CPU mode is selected')
      with self.test_session(use_gpu=use_gpu):
        bw = 8
        # Instantiate DlQuantization object
        comp_mode = libpymo.ComputationMode.COMP_MODE_GPU if use_gpu else libpymo.ComputationMode.COMP_MODE_CPU
        libpytrext.InitQuantizer(["conv1"], comp_mode, [], libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED)

        # Set encodings
        enc_min_list = [-10.0, 0.5, 20]
        enc_max_list = [100.0, 150.0, 200.0, 255.0]

        output = self.qc_quantize_module.qc_quantize_deprecated(op_name='conv1',
                                                                training_in_progress=False,
                                                                config=
                                                                int(libpytrext.config_type.CONFIG_TYPE_SET_ENCODING),
                                                                bitwidth=bw, in_tensors=[[]],
                                                                fixed_enc_mins=enc_min_list,
                                                                fixed_enc_maxs=enc_max_list)
        with self.assertRaises(errors_impl.InvalidArgumentError):
          ops.convert_to_tensor(output[0]).eval()

        libpytrext.ResetQuantizer()

  # Handle std::runtime_error from C++ to enable this test
  # def testQCQuantize_CheckNoEncodings(self):

    # with self.test_session():
      # bw = 8

      #Instantiate DlQuantization object
      # libpytrext.InitQuantizer(["conv1"], libpymo.ComputationMode.COMP_MODE_CPU,
        # [], libpymo.QuantizationMode.QUANTIZATION_TF)

      #Do not Update stats. Directly attempt to get encodings
      # output = self.qc_quantize_module.qc_quantize(op_name='conv1', config=int(libpytrext.config_type.CONFIG_TYPE_GET_ENCODING),
          # bitwidth=bw, in_tensors=[[]],
          # fixed_enc_mins=[], fixed_enc_maxs=[])

      # try:
        # enc_min = ops.convert_to_tensor(output[1]).eval()
      # except RuntimeError:
        # raise

      #libpytrext.ResetQuantizer()

if __name__ == "__main__":
  tf.test.main()
