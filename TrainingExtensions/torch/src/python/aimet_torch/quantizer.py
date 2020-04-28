# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018,2019, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Implements Training Extension code for the AIMET Quantization feature """

from __future__ import absolute_import
from __future__ import division

import os
from collections import namedtuple
from typing import Tuple, List, Union
import torch

# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_torch.batch_norm_fold import PassThroughOp
from aimet_torch import utils
from aimet_torch.qc_quantize_op import QcQuantizeWrapper, QcQuantizeStandAloneBase, QcQuantizeOpMode, \
    QcPostTrainingWrapper
from aimet_torch import save_utils as su


class Quantizer:
    """A class for performing quantization on a pytorch model.

    The Quantizer class enables quantization of a pytorch model by analyzing data run through
    the network and calculating the optimal quantization encodings based on the provided algorithm
    and bit width.
    """

    def __init__(self, model=None, quant_mode='tf_enhanced', round_mode='nearest', use_cuda=True):
        """
        :param model: The input model to add quantization ops to
        :param quant_mode: Indicates which quantization algorithm should be used, either
                'tf' or 'tf_enhanced'. Defaults to 'tf_enhanced'.
        :param round_mode: The round scheme to used. One of: 'nearest' or 'stochastic'. Default
                is 'nearest'.
        :param use_cuda: Indicates on which hardware which the quantization algorithm should run. Currently
                defaults to GPU (True).
        :raises: ValueError: An error occurred processing one of the input parameters.
        """

        if quant_mode not in ('tf_enhanced', 'tf'):
            raise ValueError('Parameter quantization mode is not a valid selection. Valid selections are tf, '
                             'tf_enhanced')

        if round_mode not in ('nearest', 'stochastic'):
            raise ValueError('Parameter round mode is not a valid selection. Valid selections are nearest or '
                             'stochastic')

        self._model = model
        self._use_cuda = use_cuda
        self._quant_mode = quant_mode
        self._round_mode = round_mode

        self._logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

    # --------------------------------------------------------------------------------------
    @staticmethod
    def __is_leaf_module(module):
        """Utility function to determine if the given module is a leaf module - that is, does not have children modules
        :return:
            True if the module is a leaf, False otherwise
        """
        module_list = list(module.modules())
        return bool(len(module_list) == 1)

    # ---------------------------------------------------------------------------------------
    def __is_quantizable_module(self, module_ref, layers_to_ignore, layer_type_to_be_quantized):
        """ Function to check if a module is eligible for quantization.
            If the module is NOT an PyTorch module type or if the module was already
            Quantized or if the module is in the layers_to_ignore list, don't quantize.
        """

        if isinstance(module_ref, (QcQuantizeWrapper, PassThroughOp)):
            self._logger.debug("Module %s already Quantized", module_ref)
            return False

        if module_ref in layers_to_ignore:
            self._logger.debug("Module %s is in layers_to_ignore list", module_ref)
            return False

        if layer_type_to_be_quantized:

            # iterate over all the layer types specified by user
            for layer_type in layer_type_to_be_quantized:
                if isinstance(module_ref, layer_type):
                    self._logger.debug("Module %s is Quantizable and of type %s", module_ref, layer_type)
                    return True

            # if don't find match in layer_type_to_be_quantized then return False
            return False

        self._logger.debug("Module %s is Quantizable", module_ref)
        return True

    # --------------------------------------------------------------------------------------
    def __add_quantization_wrappers(self, module, params):
        """Recursively add quantization wrappers to all appropriate modules starting with module
        """
        for module_name, module_ref in module.named_children():

            self._logger.debug("nn.Module found : %s", module_ref)
            # check if the module already quantized then ignore
            if isinstance(module_ref, (QcQuantizeWrapper, QcQuantizeStandAloneBase)):
                self._logger.debug("Module %s already Quantized", module_ref)
                continue

            # check if the module is leaf or not
            if self.__is_leaf_module(module_ref):
                # check if the module is quantizable
                if self.__is_quantizable_module(module_ref, params.layers_to_ignore, params.layer_type_to_be_quantized):

                    skip_output = False

                    if module_ref in params.skip_output_activation:
                        skip_output = True
                        self._logger.info("Skipping output activation quantization of Module %s ", module_ref)

                    # Create a new QcQuantize wrapper module
                    quantized_module = QcPostTrainingWrapper(module_ref, params.weight_bw, params.act_bw,
                                                             params.round_mode, params.quant_scheme,
                                                             is_output_quantized=(not skip_output))

                    setattr(module, module_name, quantized_module)

            # recursively call children modules
            else:
                self.__add_quantization_wrappers(module_ref, params)

    # --------------------------------------------------------------------------------------
    @staticmethod
    def __get_qc_quantized_layers(model):
        quantized_layers = []
        for name, module in model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                quantized_layers.append((name, module))
        return quantized_layers

    QuantParams = namedtuple('QuantParams', ['weight_bw',
                                             'act_bw',
                                             'round_mode',
                                             'quant_scheme',
                                             'layers_to_ignore',
                                             'layer_type_to_be_quantized',
                                             'skip_output_activation'])

    # --------------------------------------------------------------------------------------
    def quantize_net(self, bw_params, bw_acts,
                     run_model, iterations, layers_to_ignore=None, layer_type_to_be_quantized=None,
                     skip_output_activation=None):
        """
        Quantizes the network based on the parameters set during Quantizer construction

        The quantizer performs all quantization steps automatically. The steps are:
         #. Find all quantized ops in the network
         #. Quantize the parameters (weights, biases, etc)
         #. Run data through the network collecting statistics for activation quantization data
         #. Generate the encodings for the network
         #. Store the encodings in the corresponding quantized layers

        :param bw_params: The bit width to use for quantizing parameters in the network. Valid range is 4-32 both-inclusive.
        :param bw_acts: The bit width to use for quantizing activations in the network. Valid range is 4-32 both-inclusive.
        :param run_model: The function to use for running data through the graph and evaluating
            the network's performance. This function must return only a single number representing the avg performance
            of the model over the dataset batches.
        :param iterations: The number of iterations (data batches) to run through the network for analysis.
            If set to None, will iterate over all the test data
        :param layers_to_ignore: List of layers NOT to be quantized
        :param layer_type_to_be_quantized: List of Layer types to be quantized
        :param skip_output_activation: List of Layers to skip output activations
        :return: None
        :raises: - ValueError: An invalid parameter was passed
                 - RuntimeError: An error occurred analyzing or compressing the network. The associated error
                   and other information will be returned with the error.
        """

        # Cache and validate parameters
        if bw_params < 4 or bw_params > 32:
            raise ValueError('Parameter bitwidth must be between 4 and 32, not '+str(bw_params))
        if bw_acts < 4 or bw_acts > 32:
            raise ValueError('Activation bitwidth must be between 4 and 32, not '+str(bw_acts))

        # Validate that number of iterations is >0
        if iterations is not None and iterations < 1:
            raise ValueError('Parameter iterations must be a valid value > 0, not '+str(iterations))

        if layers_to_ignore is None:
            layers_to_ignore = []

        if layer_type_to_be_quantized is None:
            layer_type_to_be_quantized = []

        if skip_output_activation is None:
            skip_output_activation = []

        params = Quantizer.QuantParams(bw_params, bw_acts, self._round_mode, self._quant_mode,
                                       layers_to_ignore, layer_type_to_be_quantized, skip_output_activation)

        # Add quantization layers
        self.__add_quantization_wrappers(self._model, params)

        # Get list of all QcQuantize layers
        quantized_layers = self.__get_qc_quantized_layers(self._model)

        # Turn off bias quantization for all layers
        for module in self._model.modules():
            if isinstance(module, QcQuantizeWrapper) and 'bias' in module.param_quantizers:
                module.param_quantizers['bias'].enabled = False

        self._analyze_and_store_encodings(iterations, quantized_layers, run_model)

        self._logger.info('Completed quantization!')

    def _analyze_and_store_encodings(self, iterations, quantized_layers, run_model):
        """
        Analyze quantization encodings for all layers and store in the layer objects
        :param iterations: Number of iterations to use for analyzing
        :param quantized_layers: Layers to analyze encodings for
        :param run_model: Eval function callback
        :return: None
        """

        # Set the quant lib reference for all the QcQuantize ops
        # And set the mode to analysis
        for name, layer in quantized_layers:
            layer.set_mode(QcQuantizeOpMode.ANALYSIS)

        # Run forward iterations so we can collect statistics to compute the appropriate encodings
        self._model.eval()
        with torch.no_grad():
            _ = run_model(self._model, iterations, use_cuda=self._use_cuda)

        # Get the computed per-layer encodings and log them
        for name, layer in quantized_layers:
            layer.compute_encoding()

            # Before we return we set the mode to active - meaning ready for quantize/de-quantize
            # for layers with valid_encoding, otherwise we set to pass through
            if layer.output_quantizer.encoding:
                layer.set_mode(QcQuantizeOpMode.ACTIVE)
                encoding = layer.output_quantizer.encoding
                self._logger.debug("Encoding for %s: min=%f, max=%f, offset=%f. delta=%f, bw=%f",
                                   name, encoding.min, encoding.max, encoding.delta, encoding.offset, encoding.bw)
            else:
                layer.set_mode(QcQuantizeOpMode.PASSTHROUGH)

    def compute_and_save_weight_encodings(self, path: str, filename_prefix: str,
                                          input_shape: Union[Tuple, List[Tuple]]):
        """
        Save the quantized model weight encodings

        :param path: path where to store model pth and encodings
        :param filename_prefix: filename to store exported weight encodings in json format
        :param input_shape: shape of the input parameter to the model
        :return: None
        """

        device = utils.get_device(self._model)
        self._model.cpu()
        inputs = utils.create_rand_tensors_given_shapes(input_shape)

        # compute weight encodings
        weight_encoding_dict = {}
        weight_encoding_dict_with_onnx_names = {}
        quantized_layers = self.__get_qc_quantized_layers(self._model)
        pytorch_onnx_names_dict = su.SaveUtils.get_name_of_op_from_graph(self._model, *inputs)
        for layer_name, layer in quantized_layers:
            if isinstance(layer, QcQuantizeWrapper):
                layer_wt_encoding = layer.compute_weight_encodings()
                # skip dictionary update for no weight encoding case
                if layer_wt_encoding is not None:
                    value = (layer_wt_encoding.max,
                             layer_wt_encoding.min,
                             layer_wt_encoding.delta,
                             layer_wt_encoding.offset,
                             layer_wt_encoding.bw)
                    weight_encoding_dict[layer_name] = value
                    if layer_name in pytorch_onnx_names_dict:
                        weight_encoding_dict_with_onnx_names[pytorch_onnx_names_dict[layer_name]] = value
        # export weight encodings to output json file
        su.SaveUtils.save_weight_encodings_to_json(path=path, filename_prefix=filename_prefix,
                                                   weight_encoding_dict=weight_encoding_dict,
                                                   weight_encoding_dict_with_onnx_names=
                                                   weight_encoding_dict_with_onnx_names)

        self._model.to(device)

    def save_checkpoint(self, path: str, filename_prefix: str, input_shape: Union[Tuple, List[Tuple]]):
        """
        Save the quantized model and encodings

        :param path: path where to store model pth and encodings
        :param filename_prefix: filename
        :param input_shape: shape of input to model
        :return: None
        """
        # save the quantized model and encodings
        model_filename = filename_prefix + '.pth'
        model_path = os.path.join(path, model_filename)

        # save the model
        torch.save(self._model, model_path)

        # save the encodings
        su.SaveUtils().save_encodings_to_json(model=self._model, path=path, filename_prefix=filename_prefix,
                                              input_shape=input_shape)
        # save the weight encodings
        self.compute_and_save_weight_encodings(path=path, filename_prefix=filename_prefix,
                                               input_shape=input_shape)

    def load_checkpoint(self, path: str, filename_prefix: str) -> torch.nn.Module:
        """
        Load the quantized model

        :param path: path to .pth and .encodings files
        :param filename_prefix: Prefix for the .pth and .encodings files
        :return: Quantized model
        """
        model_filename = filename_prefix + '.pth'
        model_path = os.path.join(path, model_filename)

        # load the model
        model = torch.load(model_path)

        self._check_if_loaded_model_matches_original_model(model)

        return model

    def _check_if_loaded_model_matches_original_model(self, loaded_model):
        """
        Raises an exception if the newly loaded model does not match the model in this object instance
        :param loaded_model: Model loaded
        :return: None
        """

        # check if both models are same
        original_model_names = list()
        model_names = list()
        for name, _ in loaded_model.named_modules():
            model_names.append(name)
        for name, _ in self._model.named_modules():
            original_model_names.append(name)
        # check if user provided loaded_model names contains all names in original loaded_model
        if not all(name in model_names for name in original_model_names):
            raise RuntimeError('There is mismatch between quantized loaded_model(pth file) and original loaded_model!')
