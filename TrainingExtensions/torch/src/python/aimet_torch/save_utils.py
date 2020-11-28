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

""" Utilities to save a models and related parameters """

import os
from typing import Tuple

import json
import torch

from aimet_torch.qc_quantize_op import QcQuantizeStandalone, QcQuantizeWrapper
from aimet_torch.meta.old_connectedgraph import trace_and_parse
from aimet_torch import utils


class SaveUtils:
    """ Utility class to save a models and related parameters """

    @staticmethod
    def get_name_of_op_from_graph(model: torch.nn.Module, *dummy_inputs: Tuple[torch.Tensor]):
        """

        :param model:
        :param dummy_input:
        :return:
        """

        # This code is intended to be deleted and is not currently invoked from any test
        # pylint: disable=no-member
        pytorch_onnx_names_dict = {}
        list_of_x_nodes = trace_and_parse(model, dummy_inputs)
        for xnode in list_of_x_nodes:
            if xnode.op == 'output' or \
                    xnode.op == 'Parameter' or \
                    xnode.op == 'onnx::Constant' or xnode.name.find('input/0') == 0:
                pass
            else:
                xname = xnode.name
                dotted_name = SaveUtils.parse_named_op_xname(xname)
                if dotted_name.endswith("._module_to_wrap"):
                    dotted_name = dotted_name[:-len('._module_to_wrap')]
                pytorch_onnx_names_dict[dotted_name] = xname
        return pytorch_onnx_names_dict

    def save_encodings_to_json(self, model, path, filename_prefix, input_shape):
        """
        Save quantization encodings for the given model in json format
        :param model: Model to save
        :param path: Directory path to save
        :param filename_prefix: Filename of the file to save
        :param input_shape: shape of the input parameter to the model
        :return: None
        """
        # pylint: disable=too-many-locals
        device = utils.get_device(model)
        model.cpu()

        encodings_path_onnx_names = os.path.join(path, filename_prefix + '_onnx_names' + '.encodings')
        encodings_path_python_names = os.path.join(path, filename_prefix + '_pytorch_names' + '.encodings')
        encoding_dict_with_pytorch_names = {}
        encoding_dict_with_onnx_names = {}

        inputs = utils.create_rand_tensors_given_shapes(input_shape)

        pytorch_onnx_names_dict = self.get_name_of_op_from_graph(model, *inputs)

        for layer_name, layer in model.named_modules():

            if isinstance(layer, QcQuantizeStandalone):
                value = (layer.output_quantizers[0].encoding.max,
                         layer.output_quantizers[0].encoding.min,
                         layer.output_quantizers[0].encoding.delta,
                         layer.output_quantizers[0].encoding.offset,
                         layer.output_quantizers[0].bitwidth,  # hack - standalone layers have no parameters
                         layer.output_quantizers[0].bitwidth)
                encoding_dict_with_onnx_names[layer_name] = value
                encoding_dict_with_pytorch_names[layer_name] = value

            elif isinstance(layer, QcQuantizeWrapper):

                # This is a hack to keep this working for now.. Need to create new json definitions
                # The reality is that layers may have more than one parameters, or even 0 parameters,
                # this code does not handle that currently
                if layer.param_quantizers:
                    param_bw = next(iter(layer.param_quantizers.values())).bitwidth
                else:
                    param_bw = layer.output_quantizers[0].bitwidth

                value = (layer.output_quantizers[0].encoding.max,
                         layer.output_quantizers[0].encoding.min,
                         layer.output_quantizers[0].encoding.delta,
                         layer.output_quantizers[0].encoding.offset,
                         param_bw,
                         layer.output_quantizers[0].encoding.bw)
                if layer_name in pytorch_onnx_names_dict:
                    encoding_dict_with_onnx_names[pytorch_onnx_names_dict[layer_name]] = value
                    encoding_dict_with_pytorch_names[layer_name] = value

        if not encoding_dict_with_onnx_names:
            raise RuntimeError('Could not find any QcQuantizeOps in the model for saving encodings!')

        with open(encodings_path_onnx_names, 'w') as fp:
            json.dump(encoding_dict_with_onnx_names, fp, sort_keys=True, indent=4)

        with open(encodings_path_python_names, 'w') as fp:
            json.dump(encoding_dict_with_pytorch_names, fp, sort_keys=True, indent=4)

        model.to(device)

    @staticmethod
    def save_weight_encodings_to_json(path, filename_prefix, weight_encoding_dict, weight_encoding_dict_with_onnx_names):
        """
        Save quantization encodings for the given model in json format
        :param model: Model to save
        :param path: Directory path to save
        :param filename_prefix: Filename of the file to save
        :param weight_encoding_dict: dictionary with pytorch names as key and weight encodings as value
        :param weight_encoding_dict_with_onnx_names: dictionary with onxx names as key and weight encodings as value
        :return: None
        """

        weight_encoding_path = os.path.join(path, filename_prefix + '_pytorch_names_weight.encodings')
        weight_encoding_onxx_path = os.path.join(path, filename_prefix + '_onxx_names_weight.encodings')

        if not weight_encoding_dict:
            raise RuntimeError('Could not find any QcQuantizeOps in the model for saving encodings!')

        with open(weight_encoding_path, 'w') as wt_fp, open(weight_encoding_onxx_path, 'w') as wt_onxx_fp:
            json.dump(weight_encoding_dict, wt_fp, sort_keys=True, indent=4)
            json.dump(weight_encoding_dict_with_onnx_names, wt_onxx_fp, sort_keys=True, indent=4)

    @staticmethod
    def save_weights(quantized_model, model, path, filename, save_as_onnx, common="_module_to_wrap.",
                     model_input_tensor=None):
        """
        Given a quantized model, save weights for the layers in the original model

        :param quantized_model: Model after quantization
        :param model: Original model before quantization
        :param path: Directory path to save the weights to
        :param filename: Filename of the file to save the weights to
        :param save_as_onnx: If True, save in ONNX format, else weights are saved in PTH format
        :param common: ??
        :param model_input_tensor: ??
        :return:
        """
        dict_state = {}

        for k in quantized_model.state_dict():
            if common in k:
                key = k.replace("_module_to_wrap.", "")
                dict_state[key] = quantized_model.state_dict().get(k)

        model1_dict = model.state_dict()
        model1_dict.update(dict_state)
        model.load_state_dict(model1_dict)
        if save_as_onnx:
            filename = filename + '.onnx'
            final_path = os.path.join(path, filename)
            torch.onnx.export(model,                            # model being run
                              model_input_tensor,               # model input (or a tuple for multiple inputs)
                              final_path,                       # where to save the model (can be a file or file-like object)
                              export_params=True)
        else:
            filename = filename + '.pth'
            final_path = os.path.join(path, filename)
            torch.save(model.state_dict(), final_path)

    @staticmethod
    def parse_named_op_xname(xname):
        """ Parses the xname for named operations."""
        # e.g. VGG / Sequential[features] / Conv2d[0] / Conv_33
        xparts = xname.split('/')
        module_name_parts = []
        for part in xparts[1:-1]:
            bracket_pos = part.find('[')
            if bracket_pos < 0:
                module_name_parts.append(part)
            else:
                var_name = part[bracket_pos + 1:-1]
                module_name_parts.append(var_name)

        return '.'.join(module_name_parts)

    @staticmethod
    def remove_quantization_wrappers(module):
        """
        Removes quantization wrappers from model (in place)
        :param module: Model
        """
        for module_name, module_ref in module.named_children():
            if isinstance(module_ref, QcQuantizeWrapper):
                #
                setattr(module, module_name, module_ref._module_to_wrap)  # pylint: disable=protected-access
            # recursively call children modules
            else:
                SaveUtils.remove_quantization_wrappers(module_ref)
