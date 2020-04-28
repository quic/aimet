# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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

import os
import unittest
import pytest

from aimet_torch.meta.graphmeta import GraphMeta

INPUT_NODES = (
    ('rootmod', 'type0', 0),
    ('rootmod.submod1', 'type1', 0),
    ('rootmod.submod2', 'type2', 0),
    ('rootmod.submod1.leaf1', 'type3', 333),
    ('rootmod.submod2.leaf2', 'type4', 244),
    ('rootmod.submod2.leaf3', 'type3', 333),
)

INVOCATIONS = [
    (INPUT_NODES[0][0], 0),
    (INPUT_NODES[1][0], 0),
    (INPUT_NODES[3][0], 200),
    (INPUT_NODES[2][0], 0),
    (INPUT_NODES[4][0], 300),
    (INPUT_NODES[5][0], 200),
    (INPUT_NODES[1][0], 0),
    (INPUT_NODES[3][0], 200),
]


class TestTrainingExtensionsGraphMeta(unittest.TestCase):
    def test_duplicate_node_name(self):
        meta = self.create_test_meta()
        full_name, namespc, name, typ, parms = self._split_test_node_attrs(INPUT_NODES[-1])
        with pytest.raises(AssertionError):
            meta.add_node(full_name, namespc, name, typ, parms)

    def test_invalid_node_name(self):
        meta = self.create_test_meta()
        with pytest.raises(KeyError):
            meta.get_node_stats('unknown_node')

    def test_invalid_invocation(self):
        meta = self.create_test_meta()
        with pytest.raises(KeyError):
            meta.add_invocation('unknown_node', macs=0)

    def test_valid_meta(self):
        model_name = 'SimpleModel'
        meta = self.create_test_meta(model_name)
        assert meta.model_name == model_name
        assert meta.num_nodes == len(INPUT_NODES)
        node_names = list(meta.yield_node_names())
        assert len(node_names) == len(INPUT_NODES)
        for idx in range(len(INPUT_NODES)):
            assert node_names[idx] == INPUT_NODES[idx][0]
        stats_by_type = meta.get_stats_by_type()
        assert len(stats_by_type) == len(set(node[1] for node in INPUT_NODES))
        assert sum(stats['nodes'] for stats in stats_by_type.values()) == len(INPUT_NODES)
        assert sum(stats['parms'] for stats in stats_by_type.values()) == meta.num_parameters
        assert meta.num_invocations == len(INVOCATIONS)
        invocations = list(meta.yield_node_names_invoked(no_dups=False))
        assert len(invocations) == len(INVOCATIONS)
        invocations = list(meta.yield_node_names_invoked(no_dups=True))
        assert len(invocations) == len(set(entry[0] for entry in INPUT_NODES))

        stats_by_type = meta.get_stats_by_type()
        assert len(stats_by_type) == len(set(entry[1] for entry in INPUT_NODES))
        assert sum(stats['uses'] for stats in stats_by_type.values()) == len(INVOCATIONS)
        assert sum(stats['macs'] for stats in stats_by_type.values()) == \
               sum(entry[1] for entry in INVOCATIONS)

        sum_parms_static = 0
        sum_uses = 0
        sum_parms_dyn = 0
        sum_macs = 0
        for node_name in invocations:  # used 'no_dups' so result holds unique nodes
            stats = meta.get_node_stats(node_name)
            parms_static, uses, macs = stats['parms'], stats['uses'], stats['macs']
            sum_parms_static += parms_static
            sum_uses += uses
            sum_parms_dyn += uses * parms_static
            sum_macs += macs
        assert sum_parms_static == sum(entry[2] for entry in INPUT_NODES)
        assert sum_parms_static == meta.num_parameters
        assert sum_uses == len(INVOCATIONS)
        assert sum_parms_dyn == sum_parms_static + INPUT_NODES[-1][2]
        assert sum_macs == sum(entry[1] for entry in INVOCATIONS)
        assert sum_macs == meta.num_macs

    def test_data_to_dump(self):
        meta = self.create_test_meta()
        num_parms, num_invocations, num_macs, type_lines = meta._gather_data_to_dump()
        assert num_parms == meta.num_parameters
        assert num_invocations == len(INVOCATIONS)
        assert num_macs == meta.num_macs
        assert len(type_lines) == len(set(node[1] for node in INPUT_NODES))

    @classmethod
    def create_test_meta(cls, model_name='TestModel', data=None):
        meta = GraphMeta(model_name)
        if not data:
            data = INPUT_NODES
        for node in data:
            full_name, namespc, name, typ, parms_static = \
                cls._split_test_node_attrs(node, prefix=model_name)
            meta.add_node(full_name, namespc, name, typ, parms_static)
        for inv in INVOCATIONS:
            meta.add_invocation(inv[0], inv[1])
        return meta

    @staticmethod
    def _split_test_node_attrs(node, prefix=''):
        full_name, typ, parms = node
        namespc, name = os.path.splitext(full_name)
        if name:
            name = name[1:]  # skip period char
        else:
            namespc, name = prefix, namespc
        return full_name, namespc, name, typ, parms
