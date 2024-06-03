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
import collections
import contextlib
import json
import os
from typing import Union, Tuple, Optional, Dict, List, Set

import torch
from torch import nn

from aimet_common.defs import QuantScheme
from aimet_common.utils import Spinner, AimetLogger
from aimet_torch import utils
from aimet_torch.gptvq.defs import GPTVQSupportedModules, GPTVQParameters
from aimet_torch.gptvq.gptvq_optimizer import GPTVQOptimizer
from aimet_torch.gptvq.utils import compute_hessian_tensor
from aimet_torch.quantsim import ExportableQuantModule
from aimet_torch.save_utils import SaveUtils
from aimet_torch.utils import get_named_module
from aimet_torch.v2.nn import BaseQuantizationMixin
from aimet_torch.v2.quantization import DequantizedTensor
from aimet_torch.v2.quantization.affine import QuantizeDequantize
from aimet_torch.v2.quantization.encoding_analyzer import EncodingAnalyzer
from aimet_torch.v2.quantsim import QuantizationSimModel


_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class _VectorQuantizeDequantize(QuantizeDequantize):
    def __init__(
        self,
        shape,
        bitwidth: int,
        symmetric: bool,
        encoding_analyzer: EncodingAnalyzer = None,
        block_size: Optional[Tuple] = None,
    ):
        super().__init__(shape, bitwidth, symmetric, encoding_analyzer, block_size)
        # Below flag should be enabled only after optimizing weight and setting it to module weight
        self._do_bypass = False

    # pylint: disable=redefined-builtin
    def forward(self, input: torch.Tensor) -> DequantizedTensor:
        if self._do_bypass:
            output = input.as_subclass(DequantizedTensor)
            output.encoding = self.get_encoding()
            return output

        return super().forward(input)


# pylint: disable=protected-access, too-many-arguments
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
        module_names_to_exclude: Optional[List[str]] = None,
        block_level_module_names: Optional[List[List[str]]] = None,
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
        :param module_names_to_exclude: Module names which are excluded during GPTVQ optimization
        :param block_level_module_names: List of module name lists to optimize block level GPTVQ optimization instead of leaf module level
        :param file_name_prefix: Prefix to use for filename of the encodings file
        :param config_file_path: Configuration file path for model quantizers
        :return: Model with GPTVQ applied weights and saves corresponding parameter encodings JSON file at provided path
        """
        sim = cls._get_quantsim(model, dummy_input, gptvq_params, config_file_path)
        if module_names_to_exclude is None:
            module_names_to_exclude = []

        with cls._disable_quantizers_for_gptvq_optimization(sim):
            cls._apply_gptvq(model, sim, dummy_input, gptvq_params, set(module_names_to_exclude), block_level_module_names)

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
                q = _VectorQuantizeDequantize(
                    shape=(weight_shape[0] // rows_per_block, 1),
                    bitwidth=param_quantizer.bitwidth,
                    symmetric=param_quantizer.symmetric,
                    block_size=(rows_per_block, weight_shape[1]),
                ).to(module.weight.device)
                module.param_quantizers["weight"] = q

    @staticmethod
    def _disable_quantizers_for_gptvq_optimization(sim: QuantizationSimModel,
                                                   target_modules = None) -> contextlib.ExitStack:
        """
        Get context managers to disable quantizers temporarily

        :param sim: QuantizationSimModel object
        :return: List of context managers to disable quantizers
        """
        target_modules = target_modules if target_modules else set()
        exit_stack = contextlib.ExitStack()
        for module in sim.model.modules():
            if not isinstance(module, BaseQuantizationMixin):
                continue

            if module in target_modules:
                exit_stack.enter_context(module._remove_activation_quantizers())
            elif isinstance(module, GPTVQSupportedModules) and not target_modules:
                exit_stack.enter_context(module._remove_activation_quantizers())
            else:
                exit_stack.enter_context(module._remove_all_quantizers())

        return exit_stack

    @classmethod
    def _get_block_level_module_names(
        cls,
        original_model: nn.Module,
        dummy_input: Union[torch.Tensor, Tuple],
        block_level_modules_names: Optional[List[List[str]]],
    ) -> List[List[str]]:
        """
        Return block level module name list

        :param original_model: Original torch model
        :param dummy_input: Dummy input
        :param block_level_modules_names: User provided block level module names
        :return: Block level module name list
        """
        if block_level_modules_names:
            _logger.info("GPTVQ optimization will be applied to user provided block level modules")
            return block_level_modules_names

        modules = utils.get_ordered_list_of_modules(original_model, dummy_input)
        _logger.info("GPTVQ optimization will be applied to GPTVQ supportable leaf level modules")
        return [[name] for name, _ in modules]

    @classmethod
    def _apply_gptvq(
            cls,
            original_model: nn.Module,
            sim: QuantizationSimModel,
            dummy_input: Union[torch.Tensor, Tuple],
            gptvq_params: GPTVQParameters,
            module_names_to_exclude: Set[str],
            block_level_module_names: Optional[List[List[str]]],
    ):
        """
        Apply GPTVQ algorithm to optimize weights

        :param original_model: Original PyTorch model
        :param sim: QuantizationSimModel object to optimize weight
        :param dummy_input: Dummy input to model to be used to parse model graph
        :param gptvq_params: Dataclass holding GPTVQ parameters
        :param module_names_to_exclude: Module names which are excluded during GPTVQ optimization
        :param block_level_module_names: List of module name lists to optimize block level GPTVQ optimization instead of leaf module level
        """
        block_level_module_names = cls._get_block_level_module_names(
            original_model, dummy_input, block_level_module_names
        )
        for module_names in block_level_module_names:
            name_to_quant_module = cls._get_applicable_name_to_module_dict(
                module_names, sim, module_names_to_exclude
            )

            name_to_hessian = {}
            for name, quant_module in name_to_quant_module.items():
                with Spinner(f"Sampling Hessian tensor of {name}"):
                    name_to_hessian[name] = compute_hessian_tensor(
                        quant_module, gptvq_params, sim
                    )

            for name, quant_module in name_to_quant_module.items():
                assert isinstance(quant_module, BaseQuantizationMixin), "%s is not BaseQuantizationMixin" % quant_module
                assert quant_module.param_quantizers["weight"], "%s does not have weight quantizer" % quant_module

                with Spinner(f"Started GPTVQ optimization of {name}"), torch.no_grad():
                    GPTVQOptimizer.weight_update(
                        module=quant_module,
                        gptvq_params=gptvq_params,
                        hessian=name_to_hessian[name],
                    )

    @staticmethod
    def _get_applicable_name_to_module_dict(
        module_names: List[str],
        sim: QuantizationSimModel,
        module_names_to_exclude: Set[str],
    ) -> Dict[str, BaseQuantizationMixin]:
        """
        Generate GPTVQ applicable name to module dictionary

        :param module_names: Module name list
        :param sim: QuantizationSimModel object
        :param module_names_to_exclude: Module names to exclude GPTVQ optimization
        :return: Module name to Quantization module dictionary
        """
        name_to_quant_module = collections.OrderedDict()
        for name in module_names:
            quant_module = get_named_module(sim.model, name)
            if (
                isinstance(quant_module, GPTVQSupportedModules)
                and name not in module_names_to_exclude
            ):
                name_to_quant_module[name] = quant_module
        return name_to_quant_module

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
