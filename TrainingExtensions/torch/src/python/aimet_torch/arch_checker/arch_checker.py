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
from aimet_torch.meta import connectedgraph_utils
from aimet_torch.arch_checker.arch_checker_rules import get_check_dict

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

class ArchChecker:
    """
    ArchChecker object to check pytorch model architechture and suggest better architechture prior
    to training.
    """
    _arch_check_dict = get_check_dict()

    @staticmethod
    def add_check(check_target_type: torch.nn.Module, arch_check: Callable):
        """
        Add extra checks for architecture checker.
        :param check_target_type: layer type to be checked.
        :param arch_check: architecture checker function.
        :param check_kind: Check kind. Different check kind requires different arguments.
        """
        if check_target_type in ArchChecker._arch_check_dict:
            ArchChecker._arch_check_dict[check_target_type].append(arch_check)
        else:
            ArchChecker._arch_check_dict[check_target_type] = [arch_check]

    @staticmethod
    def check_arch(model: torch.nn.Module, input_shapes: Union[Tuple, List[Tuple]])-> Dict:
        """
        Check each node in the model using checks in _arch_check_dict. Record only the nodes and
        failed tests.
        :param model: Torch model to be checked.
        :param input_shapes: Input shapes to the torch model
        :return arch_checker_report: Dictionary includes only modules that failed arch_check.
                                     Arch_checker_report:
                                     Dict{"module name": set("failed check", "failed check", ...) }
        """
        connected_graph = connectedgraph_utils.create_connected_graph_with_input_shapes(model,
                                                                                        input_shapes)

        arch_checker_report = {}

        for op in connected_graph.get_all_ops().values():
            module = op.get_module()
            if module and isinstance(module, tuple(ArchChecker._arch_check_dict.keys())):
                checks = ArchChecker._arch_check_dict[type(module)]
                failed_checks_set = ArchChecker.check_node(module, checks)

                if failed_checks_set:
                    update_arch_report(arch_checker_report, {op.dotted_name_op: failed_checks_set})

                    logger.info("Graph/Node: %s: %s fails check: %s", op.dotted_name_op, module,
                                failed_checks_set)

        return arch_checker_report

    @staticmethod
    def check_node(node: torch.nn.Module, check_list: List) -> Set:
        """ Check a node with the check_list, return check names for failed checks. """
        failed_checks_list = []

        for _check in check_list:
            if not _check(node):
                failed_checks_list.append(_check.__name__)

        return set(failed_checks_list)

def update_arch_report(arch_checker_report, new_checker_report):
    """ update arch_checker_report with imcoming report """
    for new_key, new_set in new_checker_report.items():
        if new_key in arch_checker_report:
            arch_checker_report[new_key].update(new_set)
        else:
            arch_checker_report[new_key] = new_set
