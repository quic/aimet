# /usr/bin/env python3.5
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

""" Sub-sample data for weight reconstruction for channel pruning feature """

from typing import List
import math
import numpy as np
import tensorflow as tf

# Import aimet specific modules
import aimet_tensorflow.utils.common
import aimet_tensorflow.utils.op.conv
from aimet_tensorflow.layer_database import Layer, LayerDatabase

from aimet_common.utils import AimetLogger
from aimet_common.input_match_search import InputMatchSearch

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ChannelPruning)


class DataSubSampler:
    """
    Utilities to sub-sample data for weight reconstruction
    """
    @classmethod
    def get_sub_sampled_data(cls, orig_layer: Layer, pruned_layer: Layer, inp_op_names: List,
                             orig_layer_db: LayerDatabase, comp_layer_db: LayerDatabase, data_set: tf.data.Dataset,
                             batch_size: int, num_reconstruction_samples: int) -> (np.ndarray, np.ndarray):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        """
        Get all the input data from pruned model and output data from original model

        :param orig_layer: layer in original model database
        :param pruned_layer: layer in pruned model database
        :param inp_op_names : input Op names, should be same in both models
        :param orig_layer_db: original model database, un-pruned, used to provide the actual outputs
        :param comp_layer_db: comp. model database, this is potentially already pruned in the upstreams layers of given
         layer name
        :param data_set: tf.data.Dataset object
        :param batch_size : batch size
        :param num_reconstruction_samples: The number of reconstruction samples
        :return: input_data, output_data
        """
        # Grow GPU memory as needed at the cost of fragmentation.
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # pylint: disable=no-member

        # create an iterator and iterator.get_next() Op in the same graph as dataset
        # TODO: currently dataset (user provided) and iterator are in the same graph, and the iterator is
        #  being created every time this function is called. Use re-initialize iterator
        sess = tf.compat.v1.Session(graph=data_set._graph, config=config)  # pylint: disable=protected-access

        with sess.graph.as_default():
            iterator = tf.compat.v1.data.make_one_shot_iterator(data_set)
            next_element = iterator.get_next()

        # hard coded value
        samples_per_image = 10

        total_num_of_images = int(num_reconstruction_samples / samples_per_image)

        # number of possible batches - round up
        num_of_batches = math.ceil(total_num_of_images / batch_size)

        all_sub_sampled_inp_data = list()
        all_sub_sampled_out_data = list()

        for _ in range(num_of_batches):

            try:
                # get the data
                batch_data = sess.run(next_element)

                # output data from original model
                feed_dict = aimet_tensorflow.utils.common.create_input_feed_dict(orig_layer_db.model.graph,
                                                                                 inp_op_names, batch_data)
                output_data = orig_layer_db.model.run(orig_layer.module.outputs[0], feed_dict=feed_dict)

                # input data from compressed model
                feed_dict = aimet_tensorflow.utils.common.create_input_feed_dict(comp_layer_db.model.graph,
                                                                                 inp_op_names, batch_data)
                input_data = comp_layer_db.model.run(pruned_layer.module.inputs[0], feed_dict=feed_dict)

                # get the layer attributes (kernel_size, stride, padding)
                layer_attributes = aimet_tensorflow.utils.op.conv.get_layer_attributes(sess=orig_layer_db.model,
                                                                                       op=orig_layer.module,
                                                                                       input_op_names=
                                                                                       orig_layer_db.starting_ops,
                                                                                       input_shape=
                                                                                       orig_layer_db.input_shape)

                # channels_last (NHWC) to channels_first data format (NCHW - Common format)
                input_data = np.transpose(input_data, (0, 3, 1, 2))
                output_data = np.transpose(output_data, (0, 3, 1, 2))

                # get the sub sampled input and output data
                sub_sampled_inp_data, sub_sampled_out_data = InputMatchSearch.subsample_data(layer_attributes,
                                                                                             input_data,
                                                                                             output_data,
                                                                                             samples_per_image)
                all_sub_sampled_inp_data.append(sub_sampled_inp_data)
                all_sub_sampled_out_data.append(sub_sampled_out_data)

            except tf.errors.OutOfRangeError:

                raise StopIteration("There are insufficient batches of data in the provided dataset for the purpose of"
                                    " weight reconstruction! Either reduce number of reconstruction samples or increase"
                                    " data in dataset")

        # close the session
        sess.close()

        # accumulate total sub sampled input and output data
        return np.vstack(all_sub_sampled_inp_data), np.vstack(all_sub_sampled_out_data)
