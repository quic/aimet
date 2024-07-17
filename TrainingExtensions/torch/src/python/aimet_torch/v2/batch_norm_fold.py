"""This module enables batch norm folding in QuantSim v2"""
from typing import List, Tuple, Iterable
import torch
from aimet_torch import utils
from aimet_torch.batch_norm_fold import BatchNormFold as BatchNormFoldV1
from aimet_torch.batch_norm_fold import _BatchNormFoldingNotSupported, LayerType, BatchNormType, _supported_batchnorms, _supported_layers
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import FakeQuantizationMixin
from torch.nn.modules.conv import _ConvTransposeNd
import aimet_torch.v2.quantization as Q
from aimet_torch.v2.nn.base import BaseQuantizationMixin
from aimet_common.utils import AimetLogger

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.BatchNormFolding)


class BatchNormFold(BatchNormFoldV1):
    """Handles batch norm folding logic"""

    @staticmethod
    def fold_all_batch_norms_to_scale(sim: QuantizationSimModel) -> List[Tuple[BaseQuantizationMixin, BaseQuantizationMixin]]:
        """
        Fold all batch_norm layers in a model into the quantization scale parameter
        of the corresponding conv layers

        :param sim: QuantizationSimModel
        :return: A list of pairs of layers [(Conv/Linear, BN layer that got folded)]
        """
        # pylint: disable=protected-access
        assert sim.model is not None
        assert sim.connected_graph is not None

        model = sim.model
        connected_graph = sim.connected_graph

        module_to_qmodule = {}
        for module_name, module in sim.connected_graph._name_to_module.items():
            if hasattr(sim.model, module_name):
                module_to_qmodule[module] = getattr(sim.model, module_name)

            if '.' in module_name and hasattr(sim.model, module_name.split('.')[1]):
                module_to_qmodule[module] = getattr(sim.model, module_name.split('.')[1])

        conv_bn_pairs, bn_conv_pairs, _ = BatchNormFoldV1._find_all_batch_norms_to_fold(connected_graph)

        conv_bn_pairs = [
            (module_to_qmodule[conv], module_to_qmodule[bn]) for conv, bn in conv_bn_pairs
        ]
        bn_conv_pairs = [
            (module_to_qmodule[bn], module_to_qmodule[conv]) for bn, conv in bn_conv_pairs
        ]

        BatchNormFold._fold_given_batch_norms(model, conv_bn_pairs, bn_conv_pairs)
        return conv_bn_pairs + [(conv, bn) for bn, conv in bn_conv_pairs]

    @staticmethod
    def fold_given_batch_norms(model, layer_pairs):
        """
        Fold a given set of batch_norm layers into conv layers

        :param model: Model
        :param layer_pairs: Pairs of conv and batch_norm layers to use for folding
        :return: None
        """
        # pylint: disable=protected-access
        conv_bn_pairs = []
        bn_conv_pairs = []

        def is_batchnorm(module: torch.nn.Module) -> bool:
            return isinstance(module, _supported_batchnorms)

        def is_conv_linear(module: torch.nn.Module) -> bool:
            return isinstance(module, _supported_layers)

        for x, y in layer_pairs:
            if is_batchnorm(x):
                assert is_conv_linear(y)
                bn = x
                conv = y
                bn_conv_pairs.append((bn, conv))
            else:
                assert is_conv_linear(x)
                assert is_batchnorm(y)
                conv = x
                bn = y
                conv_bn_pairs.append((conv, bn))

        BatchNormFold._fold_given_batch_norms(model, conv_bn_pairs, bn_conv_pairs)

    @staticmethod
    def _fold_given_batch_norms(model,
                                conv_bn_pairs: Iterable[Tuple[torch.nn.Module, torch.nn.Module]],
                                bn_conv_pairs: Iterable[Tuple[torch.nn.Module, torch.nn.Module]]):
        """
        Fold a given set of batch_norm layers into conv layers

        :param model: Model
        :param conv_bn_pairs: List of (conv, bn) pairs to fold
        :param bn_conv_pairs: List of (bn, conv) pairs to fold
        :return: None
        """
        # pylint: disable=protected-access
        for bn, conv in bn_conv_pairs:
            if isinstance(conv, BaseQuantizationMixin):
                raise RuntimeError(f"Forward folding to scale is not possible. Got {conv}")

        bn_modules = []

        def _fold(conv, bn, fold_backward):
            is_quantized = isinstance(conv, BaseQuantizationMixin) or isinstance(bn, BaseQuantizationMixin)
            try:
                if is_quantized:
                    assert isinstance(conv, BaseQuantizationMixin) and isinstance(bn, BaseQuantizationMixin)
                    BatchNormFold._fold_to_scale(conv, bn)
                    bn_modules.append(bn)
                else:
                    BatchNormFold._fold_to_weight(conv, bn, fold_backward=fold_backward)
            except _BatchNormFoldingNotSupported as e:
                bn_name = utils.get_layer_name(model, bn)
                conv_name = utils.get_layer_name(model, conv)
                _logger.warning(
                    "Failed to fold %s to %s. [Reason] %s", bn_name, conv_name, str(e)
                )
            else:
                bn_modules.append(bn if is_quantized else bn)


        with utils.in_eval_mode(model), torch.no_grad():
            for conv, bn in conv_bn_pairs:
                _fold(conv, bn, fold_backward=True)

            for bn, conv in bn_conv_pairs:
                _fold(conv, bn, fold_backward=False)

            BatchNormFold._delete_bn_from_model(model, bn_modules)

    @staticmethod
    def _fold_to_scale(conv: BaseQuantizationMixin, bn: BaseQuantizationMixin):
        """
        Fold BatchNorm into the scale and bias of the given layer.

        :param conv: Quantized conv or linear layer.
        :param bn: Quantized bn layer.
        """
        # pylint: disable=protected-access, too-many-locals, too-many-branches, too-many-statements
        output_quantizer = conv.output_quantizers[0]

        if output_quantizer:
            raise _BatchNormFoldingNotSupported(
                "BatchNorm should belong to the same supergroup with the layer to be folded to."
            )

        if "bias" in conv.param_quantizers:
            bias_quantizer = conv.param_quantizers["bias"]
            if bias_quantizer:
                raise _BatchNormFoldingNotSupported(
                    "Can't fold BatchNorm to scale if bias quantizer is enabled."
                )

        weight_quantizer = conv.param_quantizers["weight"]

        if isinstance(conv, _ConvTransposeNd) and conv.groups != 1:
            raise _BatchNormFoldingNotSupported(
                "BatchNorm folding to scale is not supported for grouped ConvTransposeNd."
            )

        # Add quantization noise to the BN params (bn weight & bn bias) before folding.
        # NOTE: Quantization of foldable batchnorms is automatically disabled when
        #       initializing quantsim. However, it is still safer to call _quantize_params here
        #       as we can't guarantee this is always the case.
        #       For example, the user can manually enable quantization of batchnorms, etc...
        #       (FYI: _quantize_params takes effect only when the parameter quantizers are enabled)

        BatchNormFold._fold_to_weight(conv, bn, fold_backward=True)

        gamma = bn.weight
        sigma = torch.sqrt(bn.running_var + bn.eps)
        result = gamma / sigma
        new_encoding_min = torch.zeros_like(weight_quantizer.min)
        new_encoding_max = torch.zeros_like(weight_quantizer.max)

        for i, elem in enumerate(weight_quantizer.min):
            if result[i] >= 0:
                new_encoding_max[i] = weight_quantizer.max[i] * result[i]
                new_encoding_min[i] = elem * result[i]
            else:
                new_encoding_max[i] = weight_quantizer.min[i] * result[i]
                new_encoding_min[i] = weight_quantizer.max[i] * result[i]

        weight_quantizer.min.copy_(new_encoding_min)
        weight_quantizer.min.copy_(new_encoding_max)

        # Copy batchnorm's output quantizers to conv output quantizers
        for conv_output_quantizer, bn_output_quantizer in\
                zip(conv.output_quantizers, bn.output_quantizers):
            if bn_output_quantizer is None:
                conv_output_quantizer = None

            if bn_output_quantizer:
                if not conv_output_quantizer:
                    conv_output_quantizer = Q.affine.QuantizeDequantize(shape=bn_output_quantizer.shape, bitwidth=bn_output_quantizer.bitwidth,
                                               symmetric=bn_output_quantizer._symmetric)
                conv_output_quantizer.min.copy_(bn_output_quantizer.min)
                conv_output_quantizer.max.copy_(bn_output_quantizer.max)
                conv_output_quantizer.bitwidth = bn_output_quantizer.bitwidth

            bn_output_quantizer = None

        if "bias" not in conv.param_quantizers:
            bias_quantizer = Q.affine.QuantizeDequantize(shape=weight_quantizer.shape, bitwidth=weight_quantizer.bitwidth,
                                               symmetric=weight_quantizer._symmetric)
            conv.param_quantizers["bias"] = bias_quantizer

    @staticmethod
    def _fold_to_weight(conv_linear: LayerType, bn: BatchNormType, fold_backward: bool):
        """
        Fold BatchNorm into the weight and bias of the given layer.

        :param conv_linear: Conv or linear layer to fold BN into.
        :param bn: BatchNorm to fold.
        """
        # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
        # However depthwise conv layers are always N, 1, H, W whether transposed-conv or not, so no need to transpose
        if isinstance(conv_linear, torch.nn.ConvTranspose2d) and conv_linear.groups == 1:
            conv_linear.weight.data = conv_linear.weight.data.permute(1, 0, 2, 3)

        if conv_linear.bias is None:
            out_channels = conv_linear.out_features if isinstance(conv_linear, torch.nn.Linear)\
                        else conv_linear.out_channels
            bias = torch.zeros(out_channels,
                            device=conv_linear.weight.device,
                            dtype=conv_linear.weight.dtype)
            conv_linear.bias = torch.nn.Parameter(bias)

        BatchNormFoldV1._call_py_batch_norm_fold(conv_linear.weight, conv_linear.bias, bn, fold_backward=fold_backward)

        # Transpose weight back to N, C, H, W for transposed Conv2D, for non-depthwise layers
        if isinstance(conv_linear, torch.nn.ConvTranspose2d) and conv_linear.groups == 1:
            conv_linear.weight.data = conv_linear.weight.data.permute(1, 0, 2, 3)

    @staticmethod
    def _delete_bn_from_model(model: torch.nn.Module, bn_layer_list: Iterable[BatchNormType]):
        quantized_identity = FakeQuantizationMixin.from_module(torch.nn.Identity())
        utils.replace_modules_with_instances_of_new_type(model, bn_layer_list, quantized_identity)

# Global variables for compatibility
fold_all_batch_norms = BatchNormFold.fold_all_batch_norms_to_weight
fold_all_batch_norms_to_scale = BatchNormFold.fold_all_batch_norms_to_scale
fold_given_batch_norms = BatchNormFold.fold_given_batch_norms
# pylint: disable=protected-access
_is_valid_bn_fold = BatchNormFoldV1._is_valid_bn_fold
_find_all_batch_norms_to_fold = BatchNormFoldV1._find_all_batch_norms_to_fold
