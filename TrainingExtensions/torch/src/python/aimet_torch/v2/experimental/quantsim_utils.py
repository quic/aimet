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
""" Experimental quantsim utilities """

from typing import overload, Callable, Type
import torch

from aimet_common.utils import AimetLogger
from aimet_common.connected_graph.product import Product
import aimet_torch.elementwise_ops as aimet_ops
from aimet_torch.meta.connectedgraph import Op
from aimet_torch.v2.quantization.affine.quantizer import AffineQuantizerBase
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch import utils

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

_MATH_INVARIANT_OPS = (
    aimet_ops.Reshape,
    aimet_ops.Permute,
    aimet_ops.Shape,
    aimet_ops.Cast,
    aimet_ops.ChannelShuffle,
    torch.nn.ChannelShuffle,
    torch.nn.Identity
)


def _is_math_invariant_op(module: torch.nn.Module):
    return isinstance(module, _MATH_INVARIANT_OPS)


@overload
def propagate_output_encodings(sim: QuantizationSimModel, module_type: Type[torch.nn.Module]):
    """ Propagate output encodings of the given module type """


@overload
def propagate_output_encodings(sim: QuantizationSimModel, qmodule: torch.nn.Module):
    """ Propagate output encodings of qmodule """


@overload
def propagate_output_encodings(sim: QuantizationSimModel, condition: Callable[[torch.nn.Module], bool]):
    """ Propagate output encodings of all the modules that satisfies the given condition. """


def propagate_output_encodings(sim: QuantizationSimModel, arg):
    """ Propagate output encodings of all the modules that satisfies the given condition. """

    if isinstance(arg, type) and issubclass(arg, torch.nn.Module):
        module_type = arg
        condition = lambda module: isinstance(module, module_type)
    elif isinstance(arg, torch.nn.Module):
        qmodule = arg
        condition = lambda module: module is qmodule
    else:
        condition = arg

    if not sim.connected_graph:
        raise RuntimeError

    _propagate_output_encodings(sim, condition)


def _propagate_output_encodings(sim: QuantizationSimModel,
                                condition: Callable[[torch.nn.Module], bool]):
    """ Propagate output encodings of all the modules that satisfies the given condition. """
    # pylint: disable=redefined-builtin
    cg = sim.connected_graph
    qmodel = sim.model

    def get_qmodule(op: Op):
        orig_module = op.get_module()
        if not orig_module:
            return None

        full_name = cg._module_to_name[orig_module] # pylint: disable=protected-access
        _, *module_names = full_name.split('.')

        if not module_names:
            return None

        module_name = '.'.join(module_names)
        return utils.get_named_module(qmodel, module_name)

    def _set_src_qtzr(x: Product, consumer: Op, qtzr):
        producer = x.producer

        if not producer:
            # ``x`` is a root input (i.e. has no producer).
            # In this case, set the input quantizer of the consumer to ``qtzr``
            i = consumer.inputs.index(x)
            qmodule = get_qmodule(consumer)
            qmodule.input_quantizers[i] = qtzr
            assert qmodule.input_quantizers[i] is not None
            return

        qmodule = get_qmodule(producer)

        if qmodule:
            # There exists a qmodule associated with the graph node ``producer``
            # In this case, set the output quantizer of the producer to ``qtzr``
            outputs = getattr(producer, 'output_products', [producer.output])
            i = outputs.index(x)
            if qmodule.output_quantizers[i] is not None:
                qmodule.output_quantizers[i] = qtzr

        if not qmodule or _is_math_invariant_op(qmodule):
            # 1. There is no qmodule associated with the graph node ``producer``, or
            # 2. qmodule is a math invariant op (reshape, permute, etc).
            # In these cases, propagate encoding further to the ancestors
            for input in producer.inputs:
                _set_src_qtzr(input, consumer=producer, qtzr=qtzr)


    for op in reversed(cg.ordered_ops):
        qmodule = get_qmodule(op)

        if not qmodule:
            continue

        if len(qmodule.output_quantizers) != 1:
            raise RuntimeError

        if not condition(qmodule):
            continue

        qtzr, = qmodule.output_quantizers

        for input in op.inputs:
            _set_src_qtzr(input, consumer=op, qtzr=qtzr)

def clip_weights_to_7f7f(sim: 'QuantizationSimModel'):
    """
    Clip sim model weights which are 16 bit symmetric to have a max of 0x7f7f when quantized.

    :param sim: Quantsim model to clip weights for
    """
    affected_layers = []
    for name, quant_layer in sim.named_qmodules():
        # pylint: disable=too-many-boolean-expressions
        if 'weight' in quant_layer.param_quantizers and \
                quant_layer.param_quantizers['weight'] is not None and \
                quant_layer.param_quantizers['weight'].bitwidth == 16 and \
                isinstance(quant_layer.param_quantizers['weight'], AffineQuantizerBase) and \
                quant_layer.param_quantizers['weight'].symmetric and \
                quant_layer.param_quantizers['weight'].is_initialized():
            clipped_weight = torch.minimum(quant_layer.weight,
                                           quant_layer.param_quantizers['weight'].get_scale() * 0x7f7f)
            with torch.no_grad():
                quant_layer.weight.copy_(clipped_weight)

            affected_layers.append(name)
    logger_str = f'Clipping weights of the following layers to 0x7f7f max quantized value: {affected_layers}'
    logger.debug(logger_str)
