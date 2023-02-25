# /usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Keras code example for bn_reestimation """
# pylint: skip-file
import json
import os
import tensorflow as tf

from aimet_common.defs import QuantScheme
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.bn_reestimation import reestimate_bn_stats
from aimet_tensorflow.keras.batch_norm_fold import fold_all_batch_norms_to_scale


# Load FP32 model

def load_fp32_model():

    from tensorflow.compat.v1.keras.applications.resnet import ResNet50

    tf.keras.backend.clear_session()
    model = ResNet50(weights='imagenet', input_shape=(224, 224, 3))
    sess = tf.keras.backend.get_session()

    # Following lines are additional steps to make keras model work with AIMET.
    from Examples.tensorflow.utils.add_computational_nodes_in_graph import add_image_net_computational_nodes_in_graph
    add_image_net_computational_nodes_in_graph(sess, model.output.name, image_net_config.dataset['images_classes'])

    input_op_names = [model.input.op.name]
    output_op_names = [model.output.op.name]

    return sess, input_op_names, output_op_names

    # End of Load FP32 model


def rewrite_batch_norms(t_session, input_op_names, output_op_names):

    # Rewrite BatchNorm Layers

    from aimet_tensorflow.utils.op.bn_mutable import modify_sess_bn_mutable
    modify_sess_bn_mutable(sess, input_op_names, output_op_names, training_tf_placeholder=False)

    # End of Rewrite BatchNorm Layers

def create_quant_sim(tf_session, input_op_names, output_op_names):

    # Create QuantSim

    from aimet_common.defs import QuantScheme
    from aimet_tensorflow.quantsim import QuantizationSimModel

    sim = QuantizationSimModel(tf_session, input_op_names, output_op_names, use_cuda=True,
                               quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                               config_file=config_file_path)

    sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                      forward_pass_callback_args=None)

    return sim

    # End of Create QuantSim


def perform_qat(quant_sim):

    # Perform QAT

    update_ops_name = [op.name for op in model.updates] # Used for finetuning

    # User action required
    # The following line of code is an example of how to use an example ImageNetPipeline's train function.
    # Replace the following line with your own pipeline's  train function.
    ImageNetDataPipeline.finetune(quant_sim.session, update_ops_name=update_ops_name, epochs=1, learning_rate=5e-7, decay_steps=5)

    # End of Perform QAT


def call_bn_reestimation_apis(quant_sim, input_op_names, output_op_names):

    # Call reestimate_bn_stats

    from aimet_tensorflow.bn_reestimation import reestimate_bn_stats

    reestimate_bn_stats(quant_sim, start_op_names=input_op_names, output_op_names=output_op_names,
                        bn_re_estimation_dataset=bn_re_restimation_dataset, bn_num_batches=100)

    # End of Call reestimate_bn_stats

    # Call fold_all_batch_norms_to_scale

    from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms_to_scale

    fold_all_batch_norms_to_scale(quant_sim, input_op_names, output_op_names)

    # End of Call fold_all_batch_norms_to_scale


def export_the_model(quant_sim):

    os.makedirs('./output/', exist_ok=True)
    quant_sim.export(path='./output/', filename_prefix='resnet50_after_qat_and_bn_reestimation')


def bn_reestimation_example():

    tf_session, input_op_names, output_op_names = load_fp32_model()

    rewrite_batch_norms(tf_session, input_op_names, output_op_names)

    quant_sim = create_quant_sim(tf_session, input_op_names, output_op_names)

    perform_qat(quant_sim)

    call_bn_reestimation_apis(quant_sim, input_op_names, output_op_names)

    export_the_model(quant_sim)






