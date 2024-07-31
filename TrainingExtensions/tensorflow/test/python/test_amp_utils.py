# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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


""" Unit tests for AMP utils """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import pytest
import tensorflow as tf

from aimet_common.utils import AimetLogger
from aimet_tensorflow.keras.amp.utils import (
    create_mac_dict,
    calculate_running_bit_ops,
    find_bit_ops_reduction,
    find_read_var_op_parent_op_dict
)
from aimet_common.amp.utils import calculate_starting_bit_ops, QuantizationDataType
from aimet_tensorflow.keras.amp.quantizer_groups import QuantizerGroup
from models.test_models import keras_model


class TestAMPUtils:
    """ AMP utils Unit Test Cases """
    @pytest.mark.cuda
    def test_create_mac_dict(self):
        """ Test create_mac_dict """
        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        tf.keras.backend.clear_session()

        model = keras_model()

        mac_dict = create_mac_dict(model)
        print(mac_dict)
        # conv2d/Conv2D weight shape [8, 3, 2, 2] and output shape [1, 8, 15, 15]
        assert mac_dict['conv2d'] == 8 * 3 * 2 * 2 * 15 * 15



    def test_calculate_running_bit_ops(self):
        """ Test calculate running bit ops """
        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        mac_dict = {'conv2d': 21600, 'conv2d_1': 512, 'keras_model': 32}

        # 1) activation + weight
        quantizer_group = QuantizerGroup(
            parameter_quantizers=('conv2d/kernel',),
            input_quantizers=('conv2d_input_quantizer',),
        )
        read_var_op_parent_op_dict = {'conv2d_input_quantizer': 'conv2d', 'conv2d/kernel': 'conv2d'}

        op_bitwidth_dict = {}
        max_bw = ((16, QuantizationDataType.int), (16, QuantizationDataType.int))
        new_bw = ((8, QuantizationDataType.int), (8, QuantizationDataType.int))
        starting_bit_ops = 5668864
        running_bit_ops = calculate_running_bit_ops(mac_dict, quantizer_group, read_var_op_parent_op_dict,
                                                    op_bitwidth_dict, max_bw, new_bw, starting_bit_ops)

        print(running_bit_ops)
        assert running_bit_ops == mac_dict['conv2d'] * new_bw[0][0] * new_bw[1][0] +\
               mac_dict['conv2d_1'] * max_bw[0][0] * max_bw[1][0] +\
               mac_dict['keras_model'] * max_bw[0][0] * max_bw[1][0]

        # 2) only weight
        quantizer_group = QuantizerGroup(parameter_quantizers=('conv2d/kernel',))

        read_var_op_parent_op_dict = {'conv2d_input_quantizer': 'conv2d', 'conv2d/kernel': 'conv2d'}
        op_bitwidth_dict = {}
        max_bw = ((16, QuantizationDataType.int), (16, QuantizationDataType.int))
        new_bw = ((8, QuantizationDataType.int), (8, QuantizationDataType.int))
        starting_bit_ops = 5668864
        running_bit_ops = calculate_running_bit_ops(mac_dict, quantizer_group, read_var_op_parent_op_dict,
                                                    op_bitwidth_dict, max_bw, new_bw, starting_bit_ops)

        print(running_bit_ops)
        assert running_bit_ops == mac_dict['conv2d'] * max_bw[0][0] * new_bw[1][0] + \
               mac_dict['conv2d_1'] * max_bw[0][0] * max_bw[1][0] + \
               mac_dict['keras_model'] * max_bw[0][0] * max_bw[1][0]


    def test_find_bit_ops_reduction(self):
        """ Test find bit ops reduction """
        AimetLogger.set_level_for_all_areas(logging.DEBUG)

        mac_dict = {'conv2d': 21600, 'conv2d_1': 512, 'keras_model': 32}

        # 1) activation + weight
        quantizer_group = QuantizerGroup(
            parameter_quantizers=('conv2d/kernel',),
            input_quantizers=('conv2d_input_quantizer',),
        )
        read_var_op_parent_op_dict = {'conv2d_input_quantizer': 'conv2d', 'conv2d/kernel': 'conv2d'}
        bit_ops_reduction = find_bit_ops_reduction(quantizer_group, mac_dict, read_var_op_parent_op_dict,
                                                   ((16, QuantizationDataType.int), (16, QuantizationDataType.int)),
                                                   ((8, QuantizationDataType.int), (8, QuantizationDataType.int)))

        # 21600 * 16 * 16 - 21600 * 8 * 8
        #FIXME
        #assert bit_ops_reduction == 4147200

        # 2) only weight
        quantizer_group = QuantizerGroup(parameter_quantizers=('conv2d/kernel',))

        read_var_op_parent_op_dict = {'conv2d_input_quantizer': 'conv2d', 'conv2d/kernel': 'conv2d'}
        bit_ops_reduction = find_bit_ops_reduction(quantizer_group, mac_dict, read_var_op_parent_op_dict,
                                                   ((16, QuantizationDataType.int), (16, QuantizationDataType.int)),
                                                   ((8, QuantizationDataType.int), (8, QuantizationDataType.int)))

        # 21600 * 16 * 16 - 21600 * 16 * 8
        assert bit_ops_reduction == 2764800


    def test_find_op_read_var_op_dict(self):
        """ Test find op and it's read variable op dictionary """
        AimetLogger.set_level_for_all_areas(logging.DEBUG)
        tf.keras.backend.clear_session()

        model = keras_model()

        read_var_op_parent_op_dict = find_read_var_op_parent_op_dict(model)
        print(read_var_op_parent_op_dict)

        assert len(read_var_op_parent_op_dict) == 14
        assert read_var_op_parent_op_dict['conv2d/kernel'] == 'conv2d'
        assert read_var_op_parent_op_dict['conv2d/bias'] == 'conv2d'
        assert read_var_op_parent_op_dict['conv2d_1/kernel'] == 'conv2d_1'
        assert read_var_op_parent_op_dict['keras_model/kernel'] == 'keras_model'