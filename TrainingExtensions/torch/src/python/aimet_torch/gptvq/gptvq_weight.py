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
"""Top level API for GPTVQ - Post-Training Quantization (PTQ)"""

import json
import os
import tempfile
from typing import Union, Tuple, Optional, Dict

import torch
from torch import nn

import aimet_torch.v2.quantization as Q
from aimet_common.defs import QuantScheme
from aimet_common.utils import Spinner
from aimet_torch import utils
from aimet_torch.gptvq.defs import GPTVQSupportedModules, GPTVQParameters
from aimet_torch.gptvq.gptvq_optimizer import GPTVQOptimizer
from aimet_torch.quantsim import ExportableQuantModule
from aimet_torch.save_utils import SaveUtils
from aimet_torch.utils import get_named_module
from aimet_torch.v2.nn import BaseQuantizationMixin
from aimet_torch.v2.quantsim import QuantizationSimModel


# pylint: disable=protected-access
class GPTVQ:
    """
    Weight rounding mechanism for GPTVQ
    """
    @classmethod
    def apply_gptvq(
        cls,
        model: nn.Module,
        dummy_input: Union[torch.Tensor, Tuple],
        gptvq_params: GPTVQParameters,
        param_encoding_path: str,
        file_name_prefix: str = "gptvq",
        config_file_path: Optional[str] = None,
    ):
        """
        Returns model with optimized weight rounding of GPTVQ supportable modules
        and saves the corresponding parameter quantization encodings to a separate JSON file
        that can be imported by QuantizationSimModel for inference or QAT

        :param model: PyTorch model to GPTVQ
        :param dummy_input: Dummy input to the model. Used to parse model graph. If the model has more than one input,
                            pass a tuple. User is expected to place the tensors on the appropriate device
        :param gptvq_params: Dataclass holding GPTVQ parameters
        :param param_encoding_path: Path where to store parameter encodings
        :param file_name_prefix: Prefix to use for filename of the encodings file
        :param config_file_path: Configuration file path for model quantizers
        :return: Model with GPTVQ applied weights and saves corresponding parameter encodings JSON file at provided path
        """
        sim = cls._get_quantsim(model, dummy_input, gptvq_params, config_file_path)
        cls._apply_gptvq(model, sim, dummy_input, gptvq_params)
        cls._export_encodings_to_json(param_encoding_path, file_name_prefix, sim, gptvq_params.rows_per_block)
        SaveUtils.remove_quantization_wrappers(sim.model)

        return sim.model

    @classmethod
    def _get_quantsim(cls,
                      model: nn.Module,
                      dummy_input: Union[torch.Tensor, Tuple],
                      gptvq_params: GPTVQParameters,
                      config_file_path: Optional[str]) -> QuantizationSimModel:
        """
        Instantiate QuantizationSimModel object and
        replace param quantizers to be compatible with vector quantization

        :param model: Original PyTorch model
        :param dummy_input: Dummy input to be passed in QuantizationSimModel initialization
        :param gptvq_params: Dataclass holding GPTVQ parameters
        :param config_file_path: Config file path to be passed in QuantizationSimModel initialization
        :return: QuantizationSimModel with replaced param quantizers
        """
        sim = QuantizationSimModel(
            model,
            dummy_input=dummy_input,
            quant_scheme=QuantScheme.post_training_tf,
            default_param_bw=gptvq_params.vector_bw,
            config_file=config_file_path,
        )
        cls._replace_param_quantizers(sim, gptvq_params.rows_per_block)
        cls._compute_param_encodings(sim)
        return sim

    @staticmethod
    def _replace_param_quantizers(sim: QuantizationSimModel, rows_per_block: int):
        """
        Replace param quantizers to be compatible with vector quantization
        if modules are GPTVQ supportable modules

        :param sim: QuantizationSimModel object
        :param rows_per_block: The number of rows per block
        """
        for module in sim.model.modules():
            if (isinstance(module, BaseQuantizationMixin) and
                    isinstance(module.get_original_module(), GPTVQSupportedModules)):
                param_quantizer = module.param_quantizers["weight"]
                weight_shape = module.weight.shape
                q = Q.affine.QuantizeDequantize(
                    shape=(weight_shape[0] // rows_per_block, 1),
                    block_size=(rows_per_block, weight_shape[1]),
                    bitwidth=param_quantizer.bitwidth,
                    symmetric=param_quantizer.symmetric,
                ).to(module.weight.device)
                module.param_quantizers["weight"] = q

    @staticmethod
    def _compute_param_encodings(sim: QuantizationSimModel):
        """
        Remove input/output quantizers and compute param encodings for GPTVQ supportable modules

        :param sim: QuantizationSimModel object
        """
        for module in sim.model.modules():
            if isinstance(module, BaseQuantizationMixin):
                # pylint: disable=protected-access
                module._remove_activation_quantizers()
                if not isinstance(module, GPTVQSupportedModules):
                    module._remove_param_quantizers()
                    continue

                weight_quantizer = module.param_quantizers["weight"]
                with weight_quantizer.compute_encodings():
                    _ = weight_quantizer(module.weight)

    @classmethod
    def _apply_gptvq(cls,
                     original_model: nn.Module,
                     sim: QuantizationSimModel,
                     dummy_input: Union[torch.Tensor, Tuple],
                     gptvq_params: GPTVQParameters):
        """
        Apply GPTVQ algorithm to optimize weights

        :param original_model: Original PyTorch model
        :param sim: QuantizationSimModel object to optimize weight
        :param dummy_input: Dummy input to model to be used to parse model graph
        :param gptvq_params: Dataclass holding GPTVQ parameters
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            cached_dataset = utils.CachedDataset(gptvq_params.data_loader, gptvq_params.num_batches, temp_dir)
            modules = utils.get_ordered_list_of_modules(original_model, dummy_input)
            for name, _ in modules:
                quant_module = get_named_module(sim.model, name)
                if not isinstance(quant_module, GPTVQSupportedModules):
                    continue

                with Spinner(f"Started GPTVQ optimization of {name}"):
                    GPTVQOptimizer.gptvq_module(
                        quant_module=quant_module,
                        gptvq_params=gptvq_params,
                        sim=sim,
                        forward_fn=gptvq_params.forward_fn,
                        cached_dataset=cached_dataset,
                    )

    @classmethod
    def _export_encodings_to_json(cls,
                                  path: str,
                                  filename_prefix: str,
                                  sim: QuantizationSimModel,
                                  rows_per_block: int):
        """
        Save GPTVQ applied parameter encodings to JSON file

        :param path: path where to store param encodings
        :param filename_prefix: filename to store exported weight encodings in JSON format
        :param sim: QuantizationSimModel object
        :param rows_per_block: The number of rows per block
        """
        # Create a dictionary to export to JSON file
        param_encodings = {}

        for name, quant_module in sim.model.named_modules():
            if isinstance(quant_module, GPTVQSupportedModules):
                if "weight" in quant_module.param_quantizers:
                    cls._update_param_encodings_dict(
                        quant_module, name, param_encodings, rows_per_block
                    )

        # Unify the encoding format to be same as that of full encoding export file
        encoding = {"param_encodings": param_encodings}
        # export encodings to JSON file
        os.makedirs(os.path.abspath(path), exist_ok=True)
        encoding_file_path = os.path.join(path, f"{filename_prefix}.encodings")
        with open(encoding_file_path, "w") as encoding_fp:
            json.dump(encoding, encoding_fp, sort_keys=True, indent=4)

    @staticmethod
    def _update_param_encodings_dict(quant_module: ExportableQuantModule,
                                     name: str,
                                     param_encodings: Dict,
                                     rows_per_block: int):
        """
        Update block level encodings to per-channel encodings

        :param quant_module: quant module
        :param name: name of module
        :param param_encodings: Dictionary of param encodings
        :param rows_per_block: The number of rows per block
        """
        for orig_param_name, encodings in quant_module.export_param_encodings().items():
            if orig_param_name == "weight" and encodings:
                per_channel_encodings = []
                # Transform block encodings to per-channel encodings
                # blocks_per_column x 1 --> (blocks_per_column x rows_per_block) x 1
                for encoding in encodings:
                    per_channel_encodings.extend([encoding for _ in range(rows_per_block)])
                param_encodings[f"{name}.{orig_param_name}"] = per_channel_encodings
