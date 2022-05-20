# /usr/bin/env python3.5
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

# pylint: disable=too-many-lines

""" Implementation of the SVD model compression technique for TensorFlow """

import os
from functools import reduce
import operator
from enum import Enum
import numpy as np
import tensorflow as tf

from aimet_tensorflow import graph_editor
from aimet_tensorflow.common import core, graph_eval
import aimet_common.libpymo as pymo
from aimet_common import statistics_util as stats_u
from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)

_SVD_TYPES = {'svd': pymo.TYPE_SINGLE,
              'ssvd': pymo.TYPE_SUCCESSIVE}
_SVD_LAYER_TYPES = {'Conv2D': pymo.LAYER_TYPE_CONV,
                    'MatMul': pymo.LAYER_TYPE_FC}

_MIN_LAYER_DIM_FOR_SVD = 10
_SVD_SUPPORTED_LAYER_TYPES = ['Conv2D', 'MatMul']


class CostMetric(Enum):
    """ Enumeration of metrics to measure cost of a model/layer """
    mac = 1
    memory = 2


class LayerAttributes:
    """ Holds attributes for a given layer """

    def __init__(self, layer_ref, cost, weight_shape):
        """
        Constructor
        :param layer_ref: Reference to the layer object in TensorFlow
        :param cost: Cost of the layer
        :param weight_shape: Shape of the output activation of the layer
        """
        self.layer_ref = layer_ref
        self.cost = cost
        self.weight_shape = weight_shape


class Svd:
    """A class for performing singular value decomposition on a tensorflow model.

    The Svd class enables model compression through singular value decomposition (SVD).
    It can analyze convolution and fully connected layers and perform
    some analysis to find the optimal ranks for balancing compression and the
    accuracy of the network.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, graph, checkpoint, metric, output_file='./svd_graph', svd_type='svd',
                 num_layers=0, layers=None, layer_ranks=None, num_ranks=20, gpu=True, debug=False, no_evaluation=False,
                 layer_selection_threshold=0.6):
        """
        Constructor for the Svd class

        Constructs the Svd class from a set of options passed in at construction. The class takes
        a number of named arguments which are detailed below.

        :param graph: The file path to the meta graph.
        :param checkpoint: The file path to the tensorflow checkpoint file.
        :param metric: The metric to use for determining the optimal compression. Either
                'mac' for optimizing compression to minimize multiplies and accumulates or 'memory' which
                optimizes for overall memory footprint. Defaults to 'memory'
        :param output_file: The file path for saving the compressed tensorflow graph.
                aimet will save to the directory specified, using output_file as a filename prefix
        :param svd_type: Indicates which algorithm should be used, either
                'svd' or 'ssvd'. Defaults to 'svd'.
        :param num_layers: The number of layers to compress. Defaults to '0' which uses a
                heuristic to determine the optimal number of layers to compress.
        :param layers: A list of op names to compress. All other layers will be ignored.
                Overrides num_layers and sets it to the length of this list.
        :param layer_ranks: required only if no_evaluation is set to True. A list of tuples to compress
                layers specified in layers argument.
        :param num_ranks: The number of ranks (compression_points) to evaluate for compression.
                Defaults to 20. Value should be greater than 2.
        :param gpu: Indicates if the algorithm should run on GPU or CPU. Defaults to GPU. To
                use CPU set to false
        :param debug: If true debug messages will be printed. Defaults to False.
        :param no_evaluation: If true, ranks will be set manually from user. Defaults to False.
        :param layer_selection_threshold: Threshold (0-1) to use to select the top layers in the network

        :raises: ValueError: An error occurred processing one of the input parameters.
        """
        # pylint: disable=too-many-arguments

        self._sanity_check_constructor_parameters(layer_selection_threshold, layers, no_evaluation, num_layers,
                                                  num_ranks, svd_type)

        self._gpu = gpu
        self._debug = debug
        self._default_meta_graph = graph
        self._default_checkpoint = checkpoint
        self._output_file = output_file
        self._output_dir = os.path.dirname(output_file)
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)
        logger.info('Saving SVD output as: %s', output_file)

        self.svd_type = _SVD_TYPES[svd_type]

        self._metric = metric

        self._num_layers = num_layers

        self._selected_layers = []
        self._networkCost = None

        if layers:
            logger.debug('Attempting to compress: %s', layers)
            self._layers_to_compress = layers
        else:
            self._layers_to_compress = []

        if num_ranks < 0:
            raise ValueError("num_ranks must be >= 0")
        self._num_ranks = num_ranks

        if layer_ranks:
            self._layer_ranks = layer_ranks
            self._num_layer_ranks = len(layer_ranks)
            logger.debug('Attempting to compress model with user provided ranks : %s', layer_ranks)

        # Setup the SVD instance and load the graph
        self._svd = pymo.GetSVDInstance()
        self._no_eval = no_evaluation
        self._layer_selection_threshold = layer_selection_threshold
        self._model_performance_candidate_ranks = list()

        # Todo: Need to look at these attributes and see how to handle them better
        # Very likely these attributes don't need to be object attributes
        self._generator = None
        self._eval_names = None
        self._eval_func = None
        self._iterations = None
        self._run_graph = None
        self._baseline_perf = None
        self._error_margin = None
        self._compressible_ops = None

    @staticmethod
    def _sanity_check_constructor_parameters(layer_selection_threshold, layers, no_evaluation, num_layers,
                                             num_ranks, svd_type):
        if svd_type not in _SVD_TYPES:
            raise ValueError('Invalid SVD mode: ' + svd_type)
        if no_evaluation:
            if not layers:
                raise ValueError('Both layers and layer_rank parameters are needed for Manual mode')
        if layer_selection_threshold < 0 or layer_selection_threshold > 1:
            raise ValueError('Layer selection threshold should be between 0 and 1')
        if not no_evaluation:
            if num_ranks <= 2:
                raise ValueError('Number of ranks should be greater than 2 for auto mode')
        if num_layers < 0:
            raise ValueError("num_layers must be >= 0")

    def _compute_per_layer_compression_ratio(self, split_layers_shape, output_shape, original_layer_shape, op_type):
        """
        Updates the per layer statistics

        :param orig_layer: The layer before it was split
        :param split_layers: List of split layers
        :return: The compression ratio of split layers
        """
        orig_layer_cost = self._compute_layer_cost(original_layer_shape, output_shape, op_type)

        split_layers_mem_cost = 0
        split_layers_mac_cost = 0

        for layer_shape in split_layers_shape:
            mem_cost, mac_cost = self._compute_layer_cost(layer_shape, output_shape, op_type)
            if not isinstance(mem_cost, int):
                mem_cost = mem_cost.value
            if not isinstance(mac_cost, int):
                mac_cost = mac_cost.value
            split_layers_mem_cost += mem_cost
            split_layers_mac_cost += mac_cost

        if self._metric is CostMetric.memory:
            savings = orig_layer_cost[0] - split_layers_mem_cost
            ratio = savings / orig_layer_cost[0]
            logger.debug('Original Layer Cost: %s   Memory Compression Ratio: %s', orig_layer_cost[0], ratio)
        else:
            savings = orig_layer_cost[1] - split_layers_mac_cost
            ratio = savings / orig_layer_cost[1]
            logger.debug('Original Layer Cost: %s   MAC Compression Ratio: %s', orig_layer_cost[1], ratio)

        return ratio

    @staticmethod
    def _reset_session(sess):
        """
        Reset the given tf.compat.v1.Session
        :param sess: tf.compat.v1.Session
        :return: None
        """
        tf.compat.v1.reset_default_graph()
        sess.close()

    @staticmethod
    def _load_graph(graph, meta_graph, checkpoint):
        """
        Load a graph and checkpoint and create a new tf.compat.v1.Session
        :param graph: TF graph
        :param meta_graph: Meta file
        :param checkpoint: Checkpoint file
        :return: Newly created session
        """
        logger.info('Loading graph: %s', meta_graph)
        sess = tf.compat.v1.Session(graph=graph)

        # Open the graph and retore the parameters
        saver = tf.compat.v1.train.import_meta_graph(meta_graph)
        saver.restore(sess, checkpoint)
        return sess, saver

    @staticmethod
    def _get_layer_type(op):
        """
        Converts TF layer types into corresponding PyMo layer enumerated values
        :param op: TF op
        :return: PyMo enumerated value corresponding to the type of op
        """
        if op.type in _SVD_LAYER_TYPES:
            return _SVD_LAYER_TYPES[op.type]
        return pymo.LAYER_TYPE_OTHER

    class LayerSelectionScheme(Enum):
        """ Enumeration of schemes supported to select layers for SVD compression """
        manual = 1
        top_n_layers = 2
        top_x_percent = 3

    @staticmethod
    def _pick_compression_layers(sess, cost_metric, layer_select_scheme, **kwargs):
        """
        Pick layers for SVD compression given parameters
        :param sess: tf.compat.v1.Session
        :param cost_metric: Metric to use for evaluating layer cost (either in terms of memory or mac)
        :param layer_select_scheme: Layer selection scheme to use
        :param kwargs: Keyword arguments that depend on which layer selection scheme is specified
            top_n_layers:: num_layers: Number of layers to pick
            top_x_percent:: percent_thresh: Top layers up to this parameter will be selected
            manual:: layers_to_compress: List of layers (names) to compress
        :return:
        """
        # pylint: disable=too-many-locals,too-many-branches

        if not isinstance(cost_metric, CostMetric):
            raise TypeError("cost_metric is not of type CostMetric")

        if not isinstance(layer_select_scheme, Svd.LayerSelectionScheme):
            raise TypeError("layer_selection_scheme is not of type Svd.LayerSelectionScheme")

        # Find all compressible ops
        query = core.OpQuery(sess.graph)
        compressible_ops = query.get_weight_ops()
        compressible_ops = [op for op in compressible_ops if op.type in _SVD_SUPPORTED_LAYER_TYPES]

        layer_attributes_list = Svd._create_layer_attributes_list(compressible_ops, sess)
        network_cost = Svd._compute_network_cost(layer_attributes_list)

        # Heuristic1: Reject any ops whose param shape does not meet a base criterion
        pruned_list = []
        for layer_attributes in layer_attributes_list:
            h, w, n, c = layer_attributes.weight_shape
            if (n >= _MIN_LAYER_DIM_FOR_SVD) and ((c * h * w) >= _MIN_LAYER_DIM_FOR_SVD):
                pruned_list.append(layer_attributes)
            else:
                print("Pruning out {}: shape is {}".format(layer_attributes.layer_ref.name,
                                                           layer_attributes.weight_shape))

        # Reset layer_attributes_list for the next phase
        layer_attributes_list = pruned_list
        pruned_list = []

        # Sort the attribute list based on cost
        if cost_metric == CostMetric.memory:
            layer_attributes_list.sort(key=lambda x: x.cost[0], reverse=True)
        else:
            layer_attributes_list.sort(key=lambda x: x.cost[1], reverse=True)

        if layer_select_scheme == Svd.LayerSelectionScheme.top_n_layers:
            num_layers = kwargs['num_layers']
            pruned_list = layer_attributes_list[:num_layers]

        elif layer_select_scheme == Svd.LayerSelectionScheme.top_x_percent:
            percent_thresh = kwargs['percent_thresh']
            accum_cost = 0.
            total_cost = network_cost[0] if (cost_metric == CostMetric.memory) else network_cost[1]

            for layer in layer_attributes_list:
                cost = layer.cost[0] if (cost_metric == CostMetric.memory) else layer.cost[1]

                if (100 * (cost + accum_cost)/total_cost) < percent_thresh:
                    pruned_list.append(layer)
                    accum_cost += cost

        elif layer_select_scheme == Svd.LayerSelectionScheme.manual:
            layers_to_compress = kwargs['layers_to_compress']
            for layer in layer_attributes_list:
                if layer.layer_ref.name in layers_to_compress:
                    pruned_list.append(layer)

            if not pruned_list:
                raise RuntimeError('No suitable layers found in the model.')
        return pruned_list, network_cost


    @staticmethod
    def _create_layer_attributes_list(ops_to_use, sess):
        """
        Creates list of layer attributes given a set of TF ops
        :param ops_to_use: TF ops to collect layer attributes for
        :param sess: tf.compat.v1.Session to use
        :return: Created list of layer attributes
        """
        query = core.OpQuery(sess.graph)
        layer_attributes_list = []
        for op in ops_to_use:

            weight_shape = query.get_weights_for_op(op).eval(session=sess).shape
            if op.type == 'MatMul':
                n, c = weight_shape
                weight_shape = (1, 1, n, c)
            output_dims = op.outputs[0].shape

            cost = Svd._compute_layer_cost(weight_shape, output_dims, op.type)


            layer_attributes_list.append(LayerAttributes(op, cost, weight_shape))

        return layer_attributes_list

    @staticmethod
    def _compute_network_cost(layer_attributes_list):
        """
        Compute aggregate cost of the layers included in the layer attributes list
        :param layer_attributes_list: List of layer attributes
        :return: Computed cost
        """
        mac_cost = 0
        mem_cost = 0
        for layer_attributes in layer_attributes_list:
            op_mem_cost, op_mac_cost = layer_attributes.cost
            mem_cost += op_mem_cost
            mac_cost += op_mac_cost

        return mem_cost, mac_cost

    @staticmethod
    def _compute_layer_cost(weights_shape, output_dims, op_type):
        """
        Compute cost of a layer
        :param weights_shape: Shape of the weights of this layer
        :param output_dims: Shape of the output of this layer
        :param op_type: Type of this TF op
        :return: Computed layer cost
        """
        # for outputs, TF uses dims [N,H,W,C]
        mem_cost = reduce(operator.mul, weights_shape)

        if op_type == 'Conv2D':
            mac_cost = mem_cost * int(output_dims[1]) * int(output_dims[2])
        elif op_type == 'MatMul':
            mac_cost = mem_cost

        return mem_cost, mac_cost

    def _compute_compression_ratio(self, sess, cost_metric):
        """
        Compute compression ratio
        :param sess: tf.compat.v1.Session
        :return: Computed compression ratio
        """
        query = core.OpQuery(sess.graph)
        compressible_ops = query.get_weight_ops()
        compressible_ops = [op for op in compressible_ops if op.type in _SVD_SUPPORTED_LAYER_TYPES]

        layer_attributes_list = Svd._create_layer_attributes_list(compressible_ops, sess)
        selected_layers_ops = [layer.layer_ref.name for layer in self._selected_layers]
        layer_attributes_list = [layer for layer in layer_attributes_list if layer.layer_ref.name not in selected_layers_ops]
        compressed_network_cost = Svd._compute_network_cost(layer_attributes_list)

        if cost_metric is CostMetric.memory:
            savings = self._networkCost[0] - compressed_network_cost[0]
            ratio = savings/self._networkCost[0]

        else:
            savings = self._networkCost[1] - compressed_network_cost[1]
            ratio = savings/self._networkCost[1]

        return ratio

    def _store_net_stats(self, sess):
        """
        Store layer attributes in the PyMo library instance
        :param sess: tf.compat.v1.Session
        :return: None
        """
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements

        if self._metric == CostMetric.memory:
            pymo_metric = pymo.COST_TYPE_MEMORY
        else:
            pymo_metric = pymo.COST_TYPE_MAC

        self._svd.SetCostMetric(pymo_metric)

        # Layer-selection
        if self._layers_to_compress:
            selected_layers, network_cost = self._pick_compression_layers(sess,
                                                                          self._metric,
                                                                          self.LayerSelectionScheme.manual,
                                                                          layers_to_compress=self._layers_to_compress)
        elif self._num_layers > 0:
            selected_layers, network_cost = self._pick_compression_layers(sess,
                                                                          self._metric,
                                                                          self.LayerSelectionScheme.top_n_layers,
                                                                          num_layers=self._num_layers)
        else:
            percent_thresh = self._layer_selection_threshold * 100
            selected_layers, network_cost = self._pick_compression_layers(sess,
                                                                          self._metric,
                                                                          self.LayerSelectionScheme.top_x_percent,
                                                                          percent_thresh=percent_thresh)

        self._networkCost = network_cost

        print("Selected Layers:")
        for layer in selected_layers:
            print(layer.layer_ref.name)

        self._selected_layers = selected_layers

        # Get the op query module and query for all Conv/FC layers
        query = core.OpQuery(sess.graph)
        self._compressible_ops = query.get_weight_ops()

        # Set up the layer attributes for each Conv/FC layer (this also checks for trailing
        # bias adds
        for i, op in enumerate(self._compressible_ops):

            # If op is not a selected layer, skip
            if not any(op is layer.layer_ref for layer in selected_layers):
                continue

            attr = pymo.LayerAttributes()
            layerName = op.name
            output_dims = op.outputs[0].shape # TF uses dims [N,H,W,C]
            attr.layerType = self._get_layer_type(op)
            if self.svd_type == pymo.TYPE_SINGLE:
                attr.mode = self._svd.GetCompressionType(attr.layerType, 'single')
            else:
                attr.mode = self._svd.GetCompressionType(attr.layerType, 'successive')

            if op.type == 'Conv2D' or op.type == 'MatMul':
                logger.info('Setting layer attributes for: %s', layerName+'('+op.type+')')

                # Get weights
                weights = query.get_weights_for_op(op).eval(session=sess)
                w_shape = weights.shape
                logger.debug('Got weight shape: %s', w_shape)

                # Check for bias op
                bias = None
                if (i+1) < len(self._compressible_ops):
                    bias = query.get_bias_for_op(self._compressible_ops[i+1])
                    if bias is not None:
                        bias = bias.eval(session=sess)
                        logger.debug('Got %s w/bias. Shape: %s', op.type, str(bias.shape))

                if op.type == 'Conv2D':
                    attr.shape = [w_shape[3], w_shape[2], w_shape[0], w_shape[1]]   # TF Conv weight order [KH,KW,ID,OD]
                    attr.activation_dims = (output_dims[1], output_dims[2])         # (H,W)

                    # CONV weights are stored in the order {H,W,I,O} in Tensorflow
                    # Re-order them to the form {O,I,H,W}
                    weights = np.transpose(weights, (3, 2, 0, 1))

                elif op.type == 'MatMul':
                    attr.shape = [w_shape[1], w_shape[0], 1, 1]   # TF FC weight order [ID,OD], SVD expects [OD,ID]
                    attr.activation_dims = (1, 1)
                    weights = np.transpose(weights, (1, 0))

                # blobs is a numpy array... add to list then set
                params = [weights.flatten()]
                if bias is not None:
                    params.append(bias.flatten())
                attr.blobs = params

                # Save the attributes for this layer
                self._svd.StoreLayerAttributes(layerName, attr)

    def _compute_objective_score(self, model_perf, compression_score):
        """
        Compute objective score of a given compression model
        :param model_perf: Performance of compressed model
        :param compression_score: Compression ratio
        :return: Computed objective score
        """
        if model_perf + (self._error_margin / 100) >= self._baseline_perf:
            objective_score = 1 - model_perf + (1 - compression_score)
        else:
            objective_score = 1 + (1 - compression_score)      # treat lower accuracies as 0

        return objective_score

    def _split_conv_layer(self, sess, svd_ranks, attr, op_name, bias_op_name=None):
        """
        Split a given conv layer given a rank
        :param sess: tf.compat.v1.Session
        :param svd_ranks: Rank to split the layer with (two ranks in case of SSVD)
        :param attr: Reference to the corresponding layer attribute
        :param op_name: Name of the op to split
        :param bias_op_name: Name of the corresponding bias op (if any)
        :return: None
        """
        # pylint: disable=too-many-statements,too-many-branches,too-many-locals

        logger.info('Splitting conv op: %s', op_name)

        # Retrieve the op(s) from the current graph
        op = sess.graph.get_operation_by_name(op_name)

        bias_op = None
        if bias_op_name:
            bias_op = sess.graph.get_operation_by_name(bias_op_name)

        # Create new 'conv_a' layer
        pad_mode = op.get_attr('padding')
        data_format = op.get_attr('data_format').decode('utf-8')
        strides = op.get_attr('strides')

        # Print current conv weight shape
        query = core.OpQuery(sess.graph)
        w_shape = query.get_weights_for_op(op).get_shape().as_list()
        logger.debug('Original %s weight shape: %s', op.name, str(w_shape))
        split_weights, weight_sizes = [], []
        split_biases, bias_sizes = [], []

        # TF weights are in [H,W,I,O] order. We must reshape the split weights to SVD format [O,I,H,W]
        # and then transpose back
        # Conv a weights are: [1, 1, w_shape[2], svd_ranks[0]]
        split_conv_a_w_shape = (svd_ranks[0], w_shape[2], 1, 1)
        conv_a_weights = np.zeros(split_conv_a_w_shape)     # transpose(2,3,1,0)
        split_weights.append(conv_a_weights.flatten().tolist())
        weight_sizes.append(conv_a_weights.size)
        if bias_op:
            conv_a_bias = np.zeros(svd_ranks[0])
            split_biases.append(conv_a_bias.flatten().tolist())
            bias_sizes.append(conv_a_bias.size)

        num_filters = w_shape[3]
        if len(svd_ranks) >= 2 and attr.mode == pymo.TYPE_SUCCESSIVE:
            # Output channels = output_rank (s)
            num_filters = svd_ranks[1]

        # Conv b weights are: [w_shape[0],w_shape[1],svd_ranks[0],num_filters]
        split_conv_b_w_shape = (num_filters, svd_ranks[0], w_shape[0], w_shape[1])
        conv_b_weights = np.zeros(split_conv_b_w_shape)
        conv_b_bias = np.zeros(num_filters)
        split_weights.append(conv_b_weights.flatten().tolist())
        weight_sizes.append(conv_b_weights.size)
        if bias_op:
            split_biases.append(conv_b_bias.flatten().tolist())
            bias_sizes.append(conv_b_bias.size)

        # Only create a third conv layer when performing successive SVD
        if len(svd_ranks) >= 2 and attr.mode == pymo.TYPE_SUCCESSIVE:
            # Conv c weights are: [1,1,num_filters,w_shape[3]]
            split_conv_c_w_shape = (w_shape[3], num_filters, 1, 1)
            conv_c_weights = np.zeros(split_conv_c_w_shape)
            conv_c_bias = np.zeros(w_shape[3])
            split_weights.append(conv_c_weights.flatten().tolist())
            weight_sizes.append(conv_c_weights.size)
            if bias_op:
                split_biases.append(conv_c_bias.flatten().tolist())
                bias_sizes.append(conv_c_bias.size)

        # Split the weights and biases according to the number of layers and ranks
        split_weights = self._svd.SplitLayerWeights(op.name, split_weights, weight_sizes, svd_ranks)
        split_biases = self._svd.SplitLayerBiases(op.name, split_biases, bias_sizes, svd_ranks)
        if split_weights:
            conv_a_name = op.name+'_a'
            conv_a_weights = np.array(split_weights[0]).reshape(split_conv_a_w_shape).transpose(2, 3, 1, 0)
            conv_a_w = tf.Variable(initial_value=conv_a_weights, name=conv_a_name+'_w', dtype=tf.float32)
            logger.debug('%s weight shape: %s', conv_a_name, str(conv_a_weights.shape))

            # Create conv_a using default strides (1,1)
            # pylint: disable=no-member
            conv_acts = tf.nn.conv2d(op.inputs[0], conv_a_w, strides=[1, 1, 1, 1], data_format=data_format,
                                     padding=pad_mode, name=op.name+'_a')  # dilation_rate=dilation_rate
            if bias_op:
                conv_a_bias = tf.Variable(initial_value=split_biases[0], name=conv_a_name+'_bias', dtype=tf.float32)
                conv_acts = conv_acts + conv_a_bias     # tf.nn.bias_add(conv_acts, split_biases[0])

        if len(split_weights) > 1:
            # Create conv_b
            conv_b_name = op.name+'_b'
            conv_b_weights = np.array(split_weights[1]).reshape(split_conv_b_w_shape).transpose(2, 3, 1, 0)
            conv_b_w = tf.Variable(initial_value=conv_b_weights, name=conv_b_name+'_w', dtype=tf.float32)
            logger.debug('%s weight shape: %s', conv_b_name, str(conv_b_weights.shape))

            # pylint: disable=no-member
            conv_acts = tf.nn.conv2d(conv_acts, conv_b_w, strides=strides, data_format=data_format, padding=pad_mode, name=conv_b_name) #dilation_rate=dilation_rate
            if bias_op:
                conv_b_bias = tf.Variable(initial_value=split_biases[1], name=conv_b_name+'_bias', dtype=tf.float32)
                conv_acts = conv_acts + conv_b_bias     # tf.nn.bias_add(conv_acts, split_biases[1])
        ratio = self._compute_per_layer_compression_ratio([conv_a_w.shape, conv_b_w.shape], conv_acts.shape, w_shape, "Conv2D")
        # Only create a third conv layer when performing successive SVD
        if len(split_weights) > 2 and len(svd_ranks) >= 2 and attr.mode == pymo.TYPE_SUCCESSIVE:
            # Create conv_c, using default strides (1,1)
            conv_c_name = op.name+'_c'
            conv_c_weights = np.array(split_weights[2]).reshape(split_conv_c_w_shape).transpose(2, 3, 1, 0)
            conv_c_w = tf.Variable(initial_value=conv_c_weights, name=conv_c_name+'_w', dtype=tf.float32)
            logger.debug('%s weight shape: %s', conv_c_name, str(conv_c_weights.shape))

            # pylint: disable=no-member
            conv_acts = tf.nn.conv2d(conv_acts, conv_c_w, strides=[1, 1, 1, 1], data_format=data_format,
                                     padding=pad_mode, name=conv_c_name)
            if bias_op:
                conv_c_bias = tf.Variable(initial_value=split_biases[2], name=conv_c_name+'_bias', dtype=tf.float32)
                conv_acts = conv_acts + conv_c_bias     # tf.nn.bias_add(conv_acts, split_biases[2])

        consumers = []
        rerouted_inputs = [bias_op.outputs[0]] if bias_op else [op.outputs[0]]
        for inp in rerouted_inputs:
            for consumer in inp.consumers():
                consumers.append(consumer)
        _ = graph_editor.reroute_ts(conv_acts, rerouted_inputs, can_modify=consumers)

        return ratio

    def _split_fc_layer(self, sess, svd_ranks, op_name, bias_op_name=None):
        """
        Split a given conv layer given a rank
        :param sess: tf.compat.v1.Session
        :param svd_ranks: Rank to split the layer with (two ranks in case of SSVD)
        :param op_name: Name of the op to split
        :param bias_op_name: Name of the corresponding bias op (if any)
        :return: None
        """
        # pylint: disable=too-many-statements, too-many-locals

        logger.info('Splitting fully connected op: %s', op_name)

        # Retrieve the op(s) from the current graph
        op = sess.graph.get_operation_by_name(op_name)
        bias_op = None
        if bias_op_name:
            bias_op = sess.graph.get_operation_by_name(bias_op_name)

        # Print current conv weight shape
        query = core.OpQuery(sess.graph)
        w_shape = query.get_weights_for_op(op).get_shape().as_list()
        logger.debug('Original %s weight shape: %s', op.name, str(w_shape))
        split_weights, weight_sizes = [], []
        split_biases, bias_sizes = [], []

        # FC  weights are: [w_shape[2],svd_ranks[0]] in [I,O] order.
        # We must reshape the split weights to SVD format [O,I] and then transpose to NHWC
        split_fc_a_w_shape = (svd_ranks[0], w_shape[0])
        fc_a_weights = np.zeros(split_fc_a_w_shape)
        fc_a_bias = np.zeros(svd_ranks[0])
        split_weights.append(fc_a_weights.flatten().tolist())
        weight_sizes.append(fc_a_weights.size)
        if bias_op:
            split_biases.append(fc_a_bias.flatten().tolist())
            bias_sizes.append(fc_a_bias.size)

        # FC b weights are: [svd_ranks[0],num_filters] in [H,W,I,O] order.
        # We must reshape the split weights to SVD format [O,I,H,W] and then transpose to NHWC
        split_fc_b_w_shape = (w_shape[1], svd_ranks[0])
        fc_b_weights = np.zeros(split_fc_b_w_shape)
        split_weights.append(fc_b_weights.flatten().tolist())
        weight_sizes.append(fc_b_weights.size)
        if bias_op:
            fc_b_bias = np.zeros(w_shape[1])
            split_biases.append(fc_b_bias.flatten().tolist())
            bias_sizes.append(fc_b_bias.size)

        # Split the weights and biases according to the number of layers and ranks
        split_weights = self._svd.SplitLayerWeights(op.name, split_weights, weight_sizes, svd_ranks)
        split_biases = self._svd.SplitLayerBiases(op.name, split_biases, bias_sizes, svd_ranks)

        if split_weights:
            fc_a_name = op.name+'_a'
            fc_a_weights = np.array(split_weights[0]).reshape(split_fc_a_w_shape).transpose(1, 0)
            fc_a_w = tf.Variable(initial_value=fc_a_weights, name=fc_a_name+'_w', dtype=tf.float32)
            logger.debug('%s weight shape: %s', fc_a_name, str(fc_a_weights.shape))

            # Create fc_a using default strides (1,1)
            fc_acts = tf.matmul(op.inputs[0], fc_a_w, name=fc_a_name)
            if bias_op:
                fc_a_bias = tf.Variable(initial_value=split_biases[0], name=fc_a_name+'_bias', dtype=tf.float32)
                fc_acts = fc_acts + fc_a_bias

        if len(split_weights) > 1:
            # Create fc_b
            fc_b_name = op.name+'_b'
            fc_b_weights = np.array(split_weights[1]).reshape(split_fc_b_w_shape).transpose(1, 0)
            fc_b_w = tf.Variable(initial_value=fc_b_weights, name=fc_b_name+'_w', dtype=tf.float32)
            logger.debug('%s weight shape: %s', fc_b_name, str(fc_b_weights.shape))
            fc_acts = tf.matmul(fc_acts, fc_b_w, name=fc_b_name)
            if bias_op:
                fc_b_bias = tf.Variable(initial_value=split_biases[1], name=fc_b_name+'_bias', dtype=tf.float32)
                fc_acts = fc_acts + fc_b_bias
        ratio = self._compute_per_layer_compression_ratio([fc_a_w.shape, fc_b_w.shape], fc_acts.shape, w_shape, 'MatMul')
        consumers = []
        rerouted_inputs = [bias_op.outputs[0]] if bias_op else [op.outputs[0]]
        for inp in rerouted_inputs:
            for consumer in inp.consumers():
                consumers.append(consumer)
        _ = graph_editor.reroute_ts(fc_acts, rerouted_inputs, can_modify=consumers)
        return ratio

    def _split_layers(self, sess, rank_index, use_best_ranks):
        """
        Split all the selected layers given a rank index
        :param sess: tf.compat.v1.Session
        :param rank_index: Rank index to use for finding the ranks
        :param use_best_ranks: Use the best rank index (for final compressed network)
        :return: None
        """
        layer_stats = list()
        for i, op in enumerate(self._compressible_ops):

            # If op is not a selected layer, skip
            if not any(op is layer.layer_ref for layer in self._selected_layers):
                continue

            # Bias is taken care of as part of the Conv/FC op
            if op.type in ['Add', 'BiasAdd']:
                continue

            # Get the stored attributes for this op
            attr = self._svd.GetLayerAttributes(op.name)
            if not attr:
                raise RuntimeError("Layer attributes not available for layer"+op.name)

            if use_best_ranks:
                svd_ranks = attr.bestRanks
            else:
                svd_ranks = self._svd.GetCandidateRanks(op.name, rank_index)
            if svd_ranks:
                bias_op = None
                if i+1 < len(self._compressible_ops):
                    bias_op = self._compressible_ops[i+1]
                    bias_op = bias_op.name if bias_op.type in ['Add', 'BiasAdd'] else None
                if op.type in ['Conv2D']:
                    ratio = self._split_conv_layer(sess, svd_ranks, attr, op.name, bias_op)
                elif op.type in ['MatMul']:
                    ratio = self._split_fc_layer(sess, svd_ranks, op.name, bias_op)
            per_layer_stats = stats_u.SvdStatistics.PerSelectedLayer(op.name, svd_ranks, ratio)
            layer_stats.append(per_layer_stats)
        return layer_stats

    def _create_compressed_network(self, sess, rank_index, use_best_ranks):
        """
        Create a compressed network for a given rank index
        :param sess: tf.compat.v1.Session
        :param rank_index: Rank index to use for finding the ranks
        :param use_best_ranks: Use the best rank index (for final compressed network)
        :return: None
        """
        # Split the network layers and update the connections
        per_layer_stats = self._split_layers(sess, rank_index, use_best_ranks)
        return per_layer_stats

    def _perform_rank_selection(self):
        """
        Perform rank selection procedure
        :return: None
        """
        # pylint: disable=too-many-locals
        stats_per_rank_index = list()
        self._svd.ComputeNetworkCost()
        self._num_ranks = self._svd.SetCandidateRanks(self._num_ranks)

        if not self._num_ranks:
            raise RuntimeError('No good candidate ranks found for compressing specified layers.')

        # Ranks are in order from least compression to highest
        best_index = -1
        optimal_score = 0.0

        for rank_index in range(self._num_ranks):
            g = tf.Graph()
            with g.as_default():
                # Create a new network for each rank_index
                self._svd.PrintCandidateRanks(rank_index, False)

                # Load the default graph so we are operating on a fresh copy of the original graph
                sess, saver = self._load_graph(g, self._default_meta_graph, self._default_checkpoint)
                per_layer_stats = self._create_compressed_network(sess, rank_index, False)

                # Save the temp model
                output_file = os.path.join(self._output_dir, 'svd_rank_index_' + str(rank_index))
                self._save_graph(sess, saver, output_file)

            # Reset the session and start a new graph for loading the compressed model
            self._reset_session(sess)

            g = tf.Graph()
            with g.as_default():

                # In TF after making changes to the graph you must save and reload, then evaluate
                sess, saver = self._load_graph(g, output_file+'.meta', output_file)
                model_perf = self._run_graph(sess, self._generator, self._eval_names, self._eval_func, self._iterations)
                logger.info('%s performance: %s', output_file, str(model_perf))
                self._model_performance_candidate_ranks.append(model_perf * 100)

                # Estimate relative compression score for this rank_index
                compression_score = self._compute_compression_ratio(sess, self._metric)
                objective_score = self._compute_objective_score(model_perf, compression_score)
                rank_data = stats_u.SvdStatistics.PerRankIndex(rank_index=rank_index, model_accuracy=model_perf,
                                                               model_compression_ratio=compression_score,
                                                               layer_stats_list=per_layer_stats)
                stats_per_rank_index.append(rank_data)

                logger.info('Compressed network with rank_index %i/%i: accuracy = %f percent '
                            'with %f percent compression (%r option) and an objective score of %f',
                            rank_index, self._num_ranks, model_perf * 100, compression_score * 100,
                            self._metric, objective_score)

                if rank_index == 0:
                    optimal_score = objective_score
                    logger.info('Initializing objective score to %f at rank index %i', optimal_score, rank_index)

                if model_perf + self._error_margin/100 < self._baseline_perf:
                    logger.info('Model performance %f falls below %f percent of baseline performance %f'
                                ' Ending rank selection', model_perf, self._error_margin, self._baseline_perf)
                    break

                else:
                    if objective_score <= optimal_score:
                        optimal_score = objective_score
                        logger.info('Found a better value for the objective score %f at rank_index %i',
                                    optimal_score, rank_index)
                        best_index = rank_index

        if best_index != -1:
            self._svd.StoreBestRanks(best_index)
            memory_compression_ratio = self._compute_compression_ratio(sess, CostMetric.memory)
            mac_compression_ratio = self._compute_compression_ratio(sess, CostMetric.mac)
            stats = stats_u.SvdStatistics(self._baseline_perf, model_perf, self._metric, best_index,
                                          mem_comp_ratio=memory_compression_ratio, mac_comp_ratio=mac_compression_ratio,
                                          rank_stats_list=stats_per_rank_index)
            # close the session and reset the default graph
            self._reset_session(sess)
            return stats

        # close the session and reset the default graph
        self._reset_session(sess)
        raise RuntimeError('No suitable ranks found to compress model within defined error bounds.')

    def manual_rank_svd(self):
        """
        Set provided ranks in the PyMo library
        :return: None
        """
        #  Store total net cost
        self._svd.ComputeNetworkCost()

        # Ensure proper layer names are provided in no_eval mode
        if not self._layer_ranks:
            raise ValueError('Layer names MUST be specified in no_eval mode.')

        # Ensure layer_ranks is in list of tuples format
        if not all(isinstance(item, tuple) for item in self._layer_ranks):
            raise ValueError('layer_ranks should be in list of tuples format for both SVD and SSVD')

        # Check number of input ranks match with number of input layers
        if len(self._layers_to_compress) != self._num_layer_ranks:
            raise ValueError('Number of Input SVD ranks does not match number of layers.')

        for layer_name, rank in zip(self._layers_to_compress, self._layer_ranks):
            rank_list = list()
            rank_list.append(rank[1])
            if self.svd_type == _SVD_TYPES['ssvd']:
                rank_list.append(rank[1])
            self._svd.StoreBestRanks(layer_name, rank_list)
        stats = self._stats_for_manual_rank_svd()
        return stats

    @staticmethod
    def _save_graph(sess, saver, output_graph):
        """
        Utility function to save a graph
        :param sess: tf.compat.v1.Session
        :param saver: TF save
        :param output_graph: Filename and path for saving the output
        :return:
        """
        logger.info('Saving graph: %s', output_graph)
        saver.save(sess, output_graph)
        _ = tf.compat.v1.summary.FileWriter(os.path.dirname(output_graph)+"/models", sess.graph)

    def _save_compressed_network(self):
        """
        Create and save a compressed network (using the best ranks identified)
        :return:
        """
        logger.info('Saving final compressed network')
        g = tf.Graph()
        with g.as_default():
            sess, saver = self._load_graph(g, self._default_meta_graph, self._default_checkpoint)
            per_layer_stats = self._create_compressed_network(sess, 0, True)

            # Save the final network
            self._save_graph(sess, saver, self._output_file)
        self._reset_session(sess)
        return per_layer_stats

    def _stats_for_manual_rank_svd(self):
        per_layer_stats = self._save_compressed_network()
        g = tf.Graph()
        with g.as_default():
            # Load and evaluate the final network
            sess, _ = self._load_graph(g, self._output_file+'.meta', self._output_file)
            model_perf = self._run_graph(sess, self._generator, self._eval_names, self._eval_func, self._iterations)
            logger.info('%s performance: %s', self._output_file, str(model_perf))

            # Estimate relative compression score for this rank_index
            self._svd.PrintCandidateRanks(0, True)
            # Estimate relative compression score for this rank_index
            compression_score = self._compute_compression_ratio(sess, self._metric)
            logger.info('Evaluating final model using layer(s): %s. '
                        'Final accuracy = %f percent with %f percent compression (%r option).',
                        self._eval_names, model_perf*100, compression_score*100, self._metric)

            memory_compression_ratio = self._compute_compression_ratio(sess,
                                                                       CostMetric.memory)
            mac_compression_ratio = self._compute_compression_ratio(sess,
                                                                    CostMetric.mac)
            rank_data = stats_u.SvdStatistics.PerRankIndex(rank_index=0, model_accuracy=model_perf,
                                                           model_compression_ratio=compression_score,
                                                           layer_stats_list=per_layer_stats)
            rank_data_list = list()
            rank_data_list.append(rank_data)
            stats = stats_u.SvdStatistics(self._baseline_perf, model_perf, self._metric, 0,
                                          mem_comp_ratio=memory_compression_ratio,
                                          mac_comp_ratio=mac_compression_ratio,
                                          rank_stats_list=rank_data_list)
            return stats

    def compress_net(self, generator, eval_names=None, run_graph=graph_eval.evaluate_graph,
                     eval_func=graph_eval.default_eval_func, error_margin=2, iterations=100):
        """
        Compresses the network using SVD

        Runs rank selection on the network, and compresses it using the method and parameters
        passed during construction of the Svd object.

        :param generator: The generator which should be used for generating data for quantization
        :param eval_names: The list of names to use for calculating model performance
        :param run_graph: The function to use for running data through the graph and evaluating
                    the network's performance. This function must return only a single number representing the
                    avg performance of the model over the dataset batches.
                    See the 'graph_eval' module's 'evaluate_graph' function for the prototype
        :param eval_func: The function to use for evaluating the network performance. This function should always
                    return a single number that can be used for comparing different graph's performance.
                    (The default is accuracy)
        :param error_margin: The acceptable degradation in network accuracy from the original.
                    1 for 1% drop, etc. Defaults to 2%.
        :param iterations: The number of iterations (data batches) to run through the network for analysis
        :return: An object containing compression statistics

        :raises: - ValueError: An invalid parameter was passed
                 - RuntimeError: An error occurred analyzing or compressing the network. The associated error
                   and other information will be returned with the error.
        """

        self._generator = generator

        if not eval_names:
            eval_names = ['accuracy']

        self._eval_names = eval_names
        self._run_graph = run_graph
        self._eval_func = eval_func
        if error_margin <= 0:
            raise ValueError('Invalid error_margin: '+str(error_margin)+'. Must pass error_margin > 0')
        self._error_margin = error_margin
        if iterations <= 0:
            raise ValueError('Invalid iterations: '+str(iterations)+'. Number of iterations must be > 0')
        self._iterations = iterations

        # Get baseline accuracy, then store the network stats
        g = tf.Graph()
        with g.as_default():
            sess, _ = self._load_graph(g, self._default_meta_graph, self._default_checkpoint)
            self._baseline_perf = run_graph(sess, generator, eval_names, eval_func, iterations)
            logger.info('Baseline performance: %f', self._baseline_perf)
            self._store_net_stats(sess)

        self._reset_session(sess)

        if self._no_eval:
            # Set Manual rank
            stats = self.manual_rank_svd()
        else:
            # Perform rank selection
            stats = self._perform_rank_selection()
            self._save_compressed_network()

        return stats
