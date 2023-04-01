# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Utility for checking model architechture prior to using AIMET feature. """

from typing import Dict, Callable, List, Set, Union, Tuple
import torch

from aimet_common.utils import AimetLogger
from aimet_torch.meta.operation import Op
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.arch_checker.arch_checker_rules import (get_node_check_dict,
                                                         get_pattern_check_list)

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

class ArchChecker:
    """
    ArchChecker object to check pytorch model architechture and suggest better architechture prior
    to training.
    """
    _node_check_dict = get_node_check_dict()
    _pattern_checks = get_pattern_check_list()

    @staticmethod
    def add_node_check(check_target_type: torch.nn.Module, arch_check: Callable):
        """
        Add extra checks for node checks in architecture checker.
        :param check_target_type: layer type to be checked.
        :param arch_check: node checker function.
        """
        # All torch activations are combined as a single type TorchActivations.
        if check_target_type in ArchChecker._node_check_dict:
            ArchChecker._node_check_dict[check_target_type].append(arch_check)
        else:
            ArchChecker._node_check_dict[check_target_type] = [arch_check]

    @staticmethod
    def add_pattern_check(arch_check: Callable):
        """
        Add extra checks for pattern checks in architecture checker.
        :param arch_check: pattern checker function.
        :param check_kind: Check kind. Different check kind requires different arguments.
        """
        ArchChecker._pattern_checks.append(arch_check)

    @staticmethod
    def check_model_arch(model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple])-> Dict:
        """
        Check each node in the model using checks in _node_check_dict. Record only the nodes and
        failed tests.
        :param model: Torch model to be checked.
        :param dummy_input: A dummy input to the model. Can be a Tensor or a Tuple of Tensors
        :return arch_checker_report: {op.dotted_name_op: NodeErrorReportObject }
        """
        connected_graph = ConnectedGraph(model, dummy_input)
        arch_checker_report = {}

        # Run all node checkes
        _checker_report = ArchChecker.run_node_checks(connected_graph)
        update_arch_report(arch_checker_report, _checker_report)

        # Run all pattern checkes
        _checker_report = ArchChecker.run_patten_check(connected_graph)
        update_arch_report(arch_checker_report, _checker_report)

        return arch_checker_report

    @staticmethod
    def run_node_checks(connected_graph: ConnectedGraph):
        """
        Walk through connected_graph and applies node checks on each node.
        :param connected_graph: Connected_graph object.
        :return arch_checker_report: {op.dotted_name_op: NodeErrorReportObject }
        """
        arch_checker_report = {}

        for op in connected_graph.ordered_ops:
            module = op.get_module()
            if module and isinstance(module, tuple(ArchChecker._node_check_dict.keys())):
                checks = ArchChecker._node_check_dict[type(module)]
                failed_checks_set = ArchChecker.check_module(module, checks)

                if failed_checks_set:
                    new_arch_checker_report = generate_arch_checker_report(op, failed_checks_set)
                    update_arch_report(arch_checker_report, new_arch_checker_report)
                    logger.info("Graph/Node: %s: %s fails check: %s", op.dotted_name_op, module,
                                failed_checks_set)

        return arch_checker_report

    @staticmethod
    def run_patten_check(connected_graph):
        """
        Applies pattern checks on connected graph.
        :param connected_graph: Connected_graph object.
        :return arch_checker_report: {op.dotted_name_op: NodeErrorReportObject }
        """
        arch_checker_report = {}
        for _check in ArchChecker._pattern_checks:
            failed_check_ops = _check(connected_graph)

            if failed_check_ops:
                new_arch_checker_report = generate_arch_checker_report(failed_check_ops, _check.__name__)
                update_arch_report(arch_checker_report, new_arch_checker_report)
                for op in failed_check_ops:
                    logger.info("Graph/Node: %s: %s fails check: %s", op.dotted_name_op, op.get_module(), {_check.__name__})

        return arch_checker_report

    @staticmethod
    def check_module(module: torch.nn.Module, check_list: List) -> Set:
        """
        Check a torch.nn.modules with the check_list, return check names for failed checks.
        :param module: module to be checked.
        :param check_list: List of checks.
        :return set of failed check names.
        """
        failed_checks_list = []

        for _check in check_list:
            if not _check(module):
                failed_checks_list.append(_check.__name__)

        return set(failed_checks_list)

def update_arch_report(arch_checker_report: Dict, new_checker_report: Dict):
    """
    Merge new_checker_report to arch_checker_report.
    :param arch_checker_report: {op.dotted_name_op: error_report_object }
    :param new_checker_report: {op.dotted_name_op: error_report_object }
    """
    for new_key, new_error_report_object in new_checker_report.items():
        if new_key in arch_checker_report:
            arch_checker_report[new_key].add_failed_checks(new_checker_report[new_key].failed_checks)
        else:
            arch_checker_report[new_key] = new_error_report_object

# More public method will be added in the future in arch_checker_report.
# pylint: disable=too-few-public-methods
class NodeErrorReportObject:
    """ Error report object for each op. """
    def __init__(self, op, failed_checks: Set[str]) -> None:
        self.dotted_name: Op = op.dotted_name_op
        self.op_type = op.type
        self.failed_checks = set()
        self.add_failed_checks(failed_checks)

    def add_failed_checks(self, failed_checks: Set[str]):
        """ Update self.failed_checks. """
        self.failed_checks.update(failed_checks)

def generate_arch_checker_report(op: Union[List, List[Op]], failed_check: Union[Set[str], str])-> Dict[str, NodeErrorReportObject]:
    """
    Get new_arch_checker_report with op and failed_check.
    :param op: Op for node_check or list of Op for pattern check.
    :param failed_check: Set of failed check's name for node_check or str of a failed check.
    """
    # Node check returns a single Op with set of str(failed_check.__name__)
    if isinstance(op, Op):
        return {op.dotted_name_op: NodeErrorReportObject(op, failed_check)}

    # Pattenr check returns a list Op with single str(failed_check.__name__)
    return {_op.dotted_name_op: NodeErrorReportObject(_op, {failed_check}) for _op in op}
