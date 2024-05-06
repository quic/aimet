from typing import overload, Callable, Type

import torch

from aimet_torch.meta.connectedgraph import Op
from aimet_common.connected_graph.product import Product
from aimet_torch import utils
import aimet_torch.elementwise_ops as aimet_ops
from aimet_torch.v2.quantsim import QuantizationSimModel


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
    """ Propagate output encodings of the givein module type """


@overload
def propagate_output_encodings(sim: QuantizationSimModel, qmodule: torch.nn.Module):
    """ Propagate output encodings of qmodule """


@overload
def propagate_output_encodings(sim: QuantizationSimModel, condition: Callable[[torch.nn.Module], bool]):
    """ Propagate output encodings of all the modules that satisfies the given condition. """


def propagate_output_encodings(sim: QuantizationSimModel, arg):
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


def _propagate_output_encodings(sim: QuantizationSimModel, condition: Callable[[torch.nn.Module], bool]):
    """ Propagate output encodings of all the modules that satisfies the given condition. """
    cg = sim.connected_graph
    qmodel = sim.model

    def get_qmodule(op):
        orig_module = op.get_module()
        full_name = cg._module_to_name[orig_module]
        _, *module_names = full_name.split('.')
        if module_names:
            module_name = '.'.join(module_names)
            return utils.get_named_module(qmodel, module_name)
        raise None

    def _set_src_qtzr(x: Product, consumer: Op, qtzr):
        producer = x.producer

        if not producer:
            i = consumer.inputs.index(x)
            qmodule = get_qmodule(consumer)
            qmodule.input_quantizers[i] = qtzr
            return

        qmodule = get_qmodule(producer)

        if not qmodule or _is_math_invariant_op(qmodule):
            for input in producer.inputs:
                _set_src_qtzr(input, consumer=producer, qtzr=qtzr)
        else:
            outputs = getattr(producer, 'output_products', [producer.output])
            i = outputs.index(x)
            qmodule.output_quantizers[i] = qtzr


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
