# /usr/bin/env python2.7
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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
from aimet_tensorflow import svd as s
from aimet_tensorflow.common import tfrecord_generator as tf_gen
from aimet_tensorflow.common.tfrecord_generator import MnistParser


def weight_svd_auto_mode(self):

    # Allocate the generator you wish to use to provide the network with data
    generator = tf_gen.TfRecordGenerator(tfrecords=[os.path.join('data', 'mnist', 'validation.tfrecords')],
                                         parser=MnistParser())

    # Allocate the SVD instance and compress the network
    svd = s.Svd(graph=os.path.join('models', 'mnist_save.meta'), checkpoint=os.path.join('models', 'mnist_save'),
                output_file=os.path.join('svd', 'svd_graph'), layers=[], num_ranks=20,
                layer_selection_threshold=0.95, metric=s.CostMetric.memory)

    stats = svd.compress_net(generator=generator, iterations=10)

    stats.pretty_print() # Print the stats for Weight SVD compression


def weight_svd_manual_mode(self):

    # Allocate the generator you wish to use to provide the network with data
    generator = tf_gen.TfRecordGenerator(tfrecords=[os.path.join('data', 'mnist', 'validation.tfrecords')],
                                         parser=MnistParser())

    # Only Compress Conv2d_1 and MatMul_1 with ranks 31 and 9 respectively
    # no_evaluation should be True in Manual mode

    layers = ['Conv2D_1', 'MatMul_1']
    layer_ranks = [('Conv2D_1', 31), ('MatMul_1', 9)]

    svd = s.Svd(graph=os.path.join('models', 'mnist_save.meta'), checkpoint=os.path.join('models', 'mnist_save'),
                output_file=os.path.join('svd', 'svd_graph'), layers=layers, layer_ranks=layer_ranks, num_ranks=20,
                no_evaluation=True, metric=s.CostMetric.memory)

    stats = svd.compress_net(generator=generator, iterations=10)

    stats.pretty_print() # Print the stats for Weight SVD compression
