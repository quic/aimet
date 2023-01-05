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

import unittest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil
import tensorflow as tf
from aimet_tensorflow import svd as s
from aimet_tensorflow.common import tfrecord_generator as tf_gen
from aimet_tensorflow.common.tfrecord_generator import MnistParser
from aimet_common.utils import AimetLogger
logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)
tf.compat.v1.disable_eager_execution()

mnist_model_path = os.path.join(os.environ.get('DEPENDENCY_DATA_PATH'), 'mnist/models/')
mnist_tfrecords_path = os.path.join(os.environ.get('DEPENDENCY_DATA_PATH'), 'mnist/data/')


class SVD(unittest.TestCase):

    def test_svd_automatic_rank_selection_mac_top_percent(self):

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # Allocate the generator you wish to use to provide the network with data
        generator = tf_gen.TfRecordGenerator(tfrecords=[os.path.join(mnist_tfrecords_path, 'validation.tfrecords')],
                                             parser=MnistParser(data_inputs=['reshape_input']))

        # error_margin 85 % : forces to go through all 20 rank indices
        error_margin = 85

        # Allocate the SVD instance and compress the network
        meta_path = os.path.join(mnist_model_path, 'mnist_save.meta')
        checkpoint_path = os.path.join(mnist_model_path, 'mnist_save')
        svd = s.Svd(graph=meta_path,
                    checkpoint=checkpoint_path, output_file=os.path.join('svd', 'svd_graph'),
                    layers=[], num_ranks=5, layer_selection_threshold=0.90, metric=s.CostMetric.mac)

        stats = svd.compress_net(generator=generator, iterations=10, error_margin=error_margin)
        stats.pretty_print(logger=logger)

        shutil.rmtree(str('./svd'))

    @unittest.skip
    def test_svd_automatic_rank_selection_mem_top_percent(self):

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # Allocate the generator you wish to use to provide the network with data
        generator = tf_gen.TfRecordGenerator(tfrecords=[os.path.join(mnist_tfrecords_path, 'validation.tfrecords')],
                                             parser=MnistParser(data_inputs=['reshape_input']))

        # Allocate the SVD instance and compress the network
        meta_path = os.path.join(mnist_model_path, 'mnist_save.meta')
        checkpoint_path = os.path.join(mnist_model_path, 'mnist_save')
        svd = s.Svd(graph=meta_path, checkpoint=checkpoint_path,
                    output_file=os.path.join('svd', 'svd_graph'), layers=[], num_ranks=5,
                    layer_selection_threshold=0.90, metric=s.CostMetric.mac)

        stats = svd.compress_net(generator=generator, iterations=10, error_margin=2)

        stats.pretty_print(logger=logger)

        self.assertTrue(1.0 > stats.compressed_model_accuracy)
        self.assertTrue(stats.baseline_model_accuracy >= stats.compressed_model_accuracy)
        self.assertEqual(2, len(stats.per_rank_index[stats.best_rank_index].per_selected_layer))

        shutil.rmtree(str('./svd'))

    def test_svd_manual_rank_selection(self):

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # Allocate the generator you wish to use to provide the network with data
        generator = tf_gen.TfRecordGenerator(tfrecords=[os.path.join(mnist_tfrecords_path, 'validation.tfrecords')],
                                             parser=MnistParser(data_inputs=['reshape_input']))

        # Only Compress Conv2d_1 and MatMul_1 with ranks 31 and 9 respectively
        # no_evaluation should be True in Manual mode

        layers = ['conv2/Conv2D', 'dense_1/MatMul']
        layer_ranks = [('conv2/Conv2D', 31), ('dense_1/MatMul', 9)]

        meta_path = os.path.join(mnist_model_path, 'mnist_save.meta')
        checkpoint_path = os.path.join(mnist_model_path, 'mnist_save')
        svd = s.Svd(graph=meta_path, checkpoint=checkpoint_path,
                    output_file=os.path.join('svd', 'svd_graph'), layers=layers, layer_ranks=layer_ranks, num_ranks=20,
                    no_evaluation=True, metric=s.CostMetric.memory)

        stats = svd.compress_net(generator=generator, iterations=10, error_margin=2)
        stats.pretty_print(logger=logger)

        self.assertTrue(1.0 >= stats.compressed_model_accuracy)
        self.assertTrue(stats.baseline_model_accuracy >= stats.compressed_model_accuracy)

        shutil.rmtree(str('./svd'))
