# /usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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

"""
This file demonstrates the use of compression using AIMET TensorFlow Channel Pruning
technique followed by fine tuning.
"""

import os
import argparse
from decimal import Decimal
from datetime import datetime
import logging
from typing import List, Tuple
import tensorflow as tf
from tensorflow.python.keras.applications.resnet import ResNet50

# imports for AIMET
import aimet_common.defs as aimet_common_defs
from aimet_tensorflow.compress import ModelCompressor
import aimet_tensorflow.defs as aimet_tensorflow_defs
from aimet_tensorflow.utils.graph_saver import save_model_to_meta

# imports for data pipelines
from Examples.common import image_net_config
from Examples.tensorflow.utils.image_net_data_loader import ImageNetDataLoader
from Examples.tensorflow.utils.image_net_evaluator import ImageNetEvaluator
from Examples.tensorflow.utils.image_net_trainer import ImageNetTrainer
from Examples.tensorflow.utils.add_computational_nodes_in_graph import add_image_net_computational_nodes_in_graph

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logger = logging.getLogger('TensorFlowChannelPruning')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(format=formatter)


###
# This script utilizes AIMET to perform channel pruning compression (50% ratio) on a resnet50
# pretrained model with the ImageNet data set. It should re-create the same performance numbers
# as published in the AIMET release for the particular scenario as described below.

# Scenario parameters:
#    - AIMET Channel Pruning compression using auto mode
#    - Ignored conv1_conv/Conv2D (this is the first layer of the session graph)
#    - Target compression ratio: 0.5 (or 50%)
#    - Number of compression ration candidates: 10
#    - Input shape: [1, 3, 224, 224]
#    - Learning rate: 0.001
#    - Decay Steps: 5
###

class ImageNetDataPipeline:
    """
    Provides APIs for model evaluation and finetuning using ImageNet Dataset.
    """

    def __init__(self, _config: argparse.Namespace):
        """
        :param _config:
        """
        self._config = _config

    def data_loader(self):
        """
        Return ImageNet Data-loader.
        """

        data_loader = ImageNetDataLoader(self._config.tfrecord_dir,
                                         image_size=image_net_config.dataset['image_size'],
                                         batch_size=image_net_config.evaluation['batch_size'],
                                         format_bgr=True)

        return data_loader

    # pylint: disable=unused-argument
    def evaluate(self, sess: tf.Session, iterations: int = None, use_cuda: bool = False) -> float:
        """
        Evaluate the specified session using the specified number of samples from the validation set.
        AIMET's compress_model() expects the function with this signature to its eval_callback
        parameter.

        :param sess: The sess graph to be evaluated.
        :param iterations: The number of batches of the dataset.
        :param use_cuda: If True then use a GPU for inference. (Note: This variable is not used in this function)
        :return: The accuracy for the sample with the maximum accuracy.
        """

        # your code goes here instead of the example from below

        evaluator = ImageNetEvaluator(self._config.tfrecord_dir, training_inputs=['keras_learning_phase:0'],
                                      data_inputs=['input_1:0'], validation_inputs=['labels:0'],
                                      image_size=image_net_config.dataset['image_size'],
                                      batch_size=image_net_config.evaluation['batch_size'],
                                      format_bgr=True)

        return evaluator.evaluate(sess, iterations)
    # pylint: enable=unused-argument

    def finetune(self, sess: tf.Session, update_ops_name: List[str] = None):
        """
        Fine-tunes the session graph. The implementation provided here is just an example,
        provide your own implementation if needed.

        :param sess: The sess graph to fine-tune.
        :param update_ops_name: list of name of update ops (mostly BatchNorms' moving averages).
                                tf.GraphKeys.UPDATE_OPS collections is always used
                                in addition to this list
        """

        # Your code goes here instead of the example from below

        trainer = ImageNetTrainer(self._config.tfrecord_dir, training_inputs=['keras_learning_phase:0'],
                                  data_inputs=['input_1:0'], validation_inputs=['labels:0'],
                                  image_size=image_net_config.dataset['image_size'],
                                  batch_size=image_net_config.train['batch_size'],
                                  num_epochs=self._config.epochs, format_bgr=True)

        trainer.train(sess, update_ops_name=update_ops_name, learning_rate=self._config.learning_rate,
                      decay_steps=self._config.decay_steps)

        save_model_to_meta(sess, meta_path=os.path.join(self._config.logdir, 'finetuned_model'))


# pylint: disable-msg=too-many-locals
def aimet_channel_pruning(sess: tf.Session, input_op_names: List[str], output_op_names: List[str], data_loader,
                          evaluator: aimet_common_defs.EvalFunction, working_dir: str) -> Tuple[tf.Session,
                                                                                                aimet_common_defs.CompressionStats]:
    """
    Compresses the model using AIMET's Tensorflow Channel Pruning auto mode compression scheme.

    :param sess: The sess graph to compress
    :param input_op_names: The list of input op name of the sess.graph
    :param output_op_names: The list of output op name of the sess.graph
    :param data_loader: Input data loader class object
    :param evaluator: Evaluator used during compression
    :param working_dir: Dir path to save compressed TensorFlow meta file
    :return: A tuple of compressed sess graph and its statistics
    """

    # Please refer to the API documentation for more details.

    # Desired target compression ratio using Channel Pruning
    # This value denotes the desired compression % of the original model.
    # To compress the model to 20% of original model, use 0.2. This would
    # compress the model by 80%.
    # We are compressing the model by 50% here.
    target_comp_ratio = Decimal(0.5)

    # Number of compression ratio used by the API at each layer
    # API will evaluate 0.1, 0.2, ..., 0.9, 1.0 ratio (total 10 candidates)
    # at each layer
    num_comp_ratio_candidates = 10

    # Creating Greedy selection parameters:
    greedy_params = aimet_common_defs.GreedySelectionParameters(target_comp_ratio=target_comp_ratio,
                                                                num_comp_ratio_candidates=num_comp_ratio_candidates)

    # Ignoring first convolutional layer of the model for compression
    modules_to_ignore = [sess.graph.get_operation_by_name('conv1_conv/Conv2D')]

    # Creating Auto mode Parameters:
    auto_params = aimet_tensorflow_defs.ChannelPruningParameters.AutoModeParams(greedy_select_params=greedy_params,
                                                                                modules_to_ignore=modules_to_ignore)

    # AIMET uses upto 10 samples per image for reconstruction, hence the value of
    # num_reconstruction_samples should go upto 10 times of total no of images in the dataset
    # We are using total no of images for num_reconstruction_samples to speed up the process
    num_reconstruction_samples = image_net_config.dataset['val_images_len']

    # Creating Channel Pruning parameters with Auto Mode:
    params = aimet_tensorflow_defs.ChannelPruningParameters(input_op_names=input_op_names,
                                                            output_op_names=output_op_names,
                                                            data_set=data_loader.dataset,
                                                            batch_size=data_loader.batch_size,
                                                            num_reconstruction_samples=num_reconstruction_samples,
                                                            allow_custom_downsample_ops=True,
                                                            mode=aimet_tensorflow_defs.ChannelPruningParameters.Mode.auto,
                                                            params=auto_params)

    # Scheme is Channel Pruning:
    scheme = aimet_common_defs.CompressionScheme.channel_pruning

    # Cost metric is MAC, it can be MAC or Memory
    cost_metric = aimet_common_defs.CostMetric.mac

    # Input image shape
    image_shape = (1, image_net_config.dataset['image_channels'],
                   image_net_config.dataset['image_width'], image_net_config.dataset['image_height'])

    # Calling model compression using Channel Pruning:
    # Here evaluator is passed which is used by the API to evaluate the
    # accuracy for various compression ratio of each layer. To speed up
    # the process, only 10 batches of data is being used inside evaluator
    # (by passing eval_iterations=10) instead of running evaluation on
    # complete dataset.
    results = ModelCompressor.compress_model(sess=sess,
                                             working_dir=working_dir,
                                             eval_callback=evaluator,
                                             eval_iterations=10,
                                             input_shape=image_shape,
                                             compress_scheme=scheme,
                                             cost_metric=cost_metric,
                                             parameters=params)

    return results


def compress_and_finetune(config: argparse.Namespace):
    """
    1. Instantiates Data Pipeline for evaluation and training
    2. Loads the pretrained resnet50 model
    3. Calculates floating point accuracy
    4. Compression
        4.1. Compresses the model using AIMET Channel Pruning
        4.2. Logs the statistics
        4.3. Saves the compressed model
        4.4. Calculates and logs the accuracy of compressed model
    5. Finetuning
        5.1. Finetunes the compressed model
        5.2. Calculates and logs the accuracy of compressed-finetuned model

    :param config: This argparse.Namespace config expects following parameters:
                   tfrecord_dir: Path to a directory containing ImageNet TFRecords.
                                This folder should contain files starting with:
                                'train*': for training records and 'validation*': for validation records
                   logdir: Path to a directory for logging.
                   epochs: Number of epochs (type int) for finetuning.
                   learning_rate: A float type learning rate for model finetuning
                   decay_steps: A number used to adjust(decay) the learning rate after every decay_steps
                                epochs in finetuning.
    """

    # 1. Instantiates Data Pipeline for evaluation and training
    data_pipeline = ImageNetDataPipeline(config)

    # 2. Loads the pretrained resnet50 keras model
    input_shape = (image_net_config.dataset['image_width'],
                   image_net_config.dataset['image_height'],
                   image_net_config.dataset['image_channels'])
    tf.keras.backend.clear_session()
    model = ResNet50(weights='imagenet', input_shape=input_shape)
    sess = tf.keras.backend.get_session()
    add_image_net_computational_nodes_in_graph(sess, model.output, image_net_config.dataset['images_classes'])
    update_ops_name = [op.name for op in model.updates]

    # 3. Calculates floating point accuracy
    accuracy = data_pipeline.evaluate(sess)
    logger.info("Original Model Top-1 accuracy = %.2f", accuracy)

    # 4. Compression
    logger.info("Starting Model Compression...")

    # 4.1. Compresses the model using AIMET Channel Pruning
    # Here 'labels' has been added into input_op_names as the data_loader.data_set gives
    # a tuple of (images, labels) and aimet channel pruning API checks the length of
    # input_op_names against the length of data_set output. The 'labels' value will be
    # fed but not utilized though.
    compressed_sess, stats = aimet_channel_pruning(sess=sess,
                                                   input_op_names=['input_1', 'labels'],
                                                   output_op_names=[model.output.name.split(":")[0]],
                                                   data_loader=data_pipeline.data_loader(),
                                                   evaluator=data_pipeline.evaluate, working_dir=config.logdir)

    # 4.2. Logs the statistics
    logger.info(stats)
    with open(os.path.join(config.logdir, 'log.txt'), "w") as outfile:
        outfile.write("%s\n\n" % stats)

    # 4.3. Saves the compressed model
    save_model_to_meta(compressed_sess, meta_path=os.path.join(config.logdir, 'compressed_model'))

    # 4.4. Calculates and logs the accuracy of compressed model
    accuracy = data_pipeline.evaluate(compressed_sess)
    logger.info("Compressed Model Top-1 accuracy = %.2f", accuracy)

    logger.info("...Model Compression Done")

    # 5. Finetuning
    logger.info("Starting Model Finetuning...")

    # 5.1. Finetunes the compressed model
    # Since Channel Pruning replaces few BNs by different BNs with 'reduced_' added in their original name,
    # update_ops_name list should be updated accordingly
    compr_graph_all_ops_name = [op.name for op in compressed_sess.graph.get_operations()]
    update_ops_name_after_CP = []
    for op_name in update_ops_name:
        if 'reduced_'+op_name in compr_graph_all_ops_name:
            update_ops_name_after_CP.append('reduced_'+op_name)
        else:
            update_ops_name_after_CP.append(op_name)

    data_pipeline.finetune(compressed_sess, update_ops_name=update_ops_name_after_CP)

    # 5.2. Calculates and logs the accuracy of compressed-finetuned model
    accuracy = data_pipeline.evaluate(compressed_sess)
    logger.info("Finetuned Compressed Model Top-1 accuracy = %.2f", accuracy)

    logger.info("...Model Finetuning Done")


if __name__ == '__main__':
    default_logdir = os.path.join("benchmark_output", "channel_pruning_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    parser = argparse.ArgumentParser(
        description='Apply Channel Pruning on pretrained ResNet50 model and finetune it for ImageNet dataset')

    parser.add_argument('--tfrecord_dir', type=str,
                        required=True,
                        help="Path to a directory containing ImageNet TFRecords.\n\
                              This folder should contain files starting with:\n\
                              'train*': for training records and 'validation*': for validation records")

    parser.add_argument('--logdir', type=str,
                        default=default_logdir,
                        help="Path to a directory for logging.\
                              Default value is 'benchmark_output/channel_pruning_<Y-m-d-H-M-S>'")

    parser.add_argument('--epochs', type=int,
                        default=15,
                        help="Number of epochs for finetuning.\n\
                              Default is 15")
    parser.add_argument('--learning_rate', type=float,
                        default=1e-3,
                        help="A float type learning rate for model finetuning.\n\
                              default is 0.001")
    parser.add_argument('--decay_steps', type=int,
                        default=5,
                        help="A number used to adjust(decay) the learning rate after every decay_steps epochs in finetuning.\n\
                              default is 5")

    _config = parser.parse_args()

    os.makedirs(_config.logdir, exist_ok=True)

    fileHandler = logging.FileHandler(os.path.join(_config.logdir, "test.log"))
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    compress_and_finetune(_config)
