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
import os
from typing import Dict, List, Union, Set, Tuple
import torch
import pandas as pd

from aimet_torch.meta.operation import Op
from aimet_torch.arch_checker.constants import ArchCheckerReportConstants as report_const
from aimet_torch.arch_checker.arch_checker_rules import NODE_CHECK_DICT, PATTERN_CHECK_LIST

import aimet_common.connected_graph.operation
from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

# pylint: disable=too-few-public-methods
class OpStructure(aimet_common.connected_graph.operation.Op):
    """
    Subclass OpStructure inherited from aimet_common.connected_graph.operation.Op.
    Used to mark a structure of Ops.
    """
    def __init__(self, op_tuple: str):
        """
        Initializer for OpStructure
        :param op_tuple: structure with ops.
        """
        self.op_tuple = op_tuple
        dotted_name = get_dotted_name_op_from_op_tuple(op_tuple)
        first_op = op_tuple[0]
        while not isinstance(first_op, Op):
            first_op = first_op[0]

        last_op = op_tuple[-1]
        while not isinstance(last_op, Op):
            last_op = last_op[-1]

        output_shape = last_op.output_shape
        is_anonymous = True
        op_type = report_const.OP_STRUCT_OP_TYPE
        name = '_'.join([report_const.OP_STRUCT_OP_TYPE, first_op.name.split("_")[-1], last_op.name.split("_")[-1]])
        super().__init__(name, dotted_name, output_shape, is_anonymous, op_type)

    def get_module(self):
        """ Return all modules associated with this Op Tuple in connected_graph sequence. """
        return [op.get_module() if isinstance(op, Op) else OpStructure(op).get_module() for op in self.op_tuple]

def get_dotted_name_op_from_op_tuple(op_list):
    """ Structure of module always return a list. Must start with list_handler """
    def list_handler(op_list):
        """ Add "-> " between 2 Ops. """
        dotted_name_op = ""
        for op in op_list:
            if isinstance(op, tuple):
                next_name = tuple_handler(op)
            elif isinstance(op, list):
                next_name = list_handler(op)
            else:
                next_name = op.dotted_name_op
            dotted_name_op += next_name
            dotted_name_op += "-> "

        # remove last "-> "
        dotted_name_op = dotted_name_op[:-3]
        return dotted_name_op

    def tuple_handler(op_tuple):
        """ Add ", " between 2 Ops and attach () on the end of the strings. """

        dotted_name_op = "("

        for op in op_tuple:
            if isinstance(op, list):
                next_name = "("
                next_name += list_handler(op)
                next_name += ")"
            elif isinstance(op, tuple):
                next_name = tuple_handler(op)
            else:
                next_name = op.dotted_name_op
            dotted_name_op += next_name
            dotted_name_op += ", "

        #remove last ", "
        dotted_name_op = dotted_name_op[:-2]
        dotted_name_op += ")"

        return dotted_name_op

    return list_handler(op_list)

class NodeErrorReportObject:
    """ Error report object for each op. """
    def __init__(self, op: Union[Op, OpStructure], failed_checks: Set[str]) -> None:
        self.dotted_name_op = op.dotted_name_op
        self.op_type = op.type
        self.failed_checks = set()
        self.add_failed_checks(failed_checks)

    def add_failed_checks(self, failed_checks: Set[str]):
        """ Update self.failed_checks. """
        self.failed_checks.update(failed_checks)

    def get_issue_recomm_list(self) -> List[Tuple[str, str]]:
        """ Get list of issue and recomm tuple for node object. """
        return [_get_report_issue_recomm(_failed_check, self.op_type) for _failed_check in self.failed_checks]

class ArchCheckerReport:
    """
    ArchCheckerReport object to handle utilities for arch_checker_report.
    """
    def __init__(self, result_dir: str = "/tmp") -> None:
        self._raw_report = {}
        self._result_dir = result_dir
        self._filename = "arch_checker_report"

    def merge_new_raw_report(self, new_raw_report):
        """
        Merge new_raw_report to self.raw_report.
        :param new_raw_report: {op.dotted_name: NodeErrorReportObject }
        """
        _update_raw_report(self._raw_report, new_raw_report)

    def update_raw_report(self, op: Union[Op, List[Op], List[OpStructure]], failed_check: Union[Set[str], str]):
        """
        Update raw_report with op and failed check.
        :param op: Op, list of Ops or List of OpStructure.
        :param failed_check: failed_check.__name__ or set(failed_check.__name__)
        """
        _update_raw_report(self._raw_report, _generate_arch_checker_report(op, failed_check))

    def export_to_html(self):
        """ Map raw report to human-friendly messages in pandas dataframe and export to html file. """
        df = _get_pd_dataframe(self._raw_report)
        html_filename = self._get_write_path(".html")
        html_file_txt = df.to_html()

        with open(html_filename, "w") as html_file:
            html_file.write(html_file_txt)
        logger.info("Save arch_checker report to %s", html_filename)

    def _get_write_path(self, file_extension: str):
        """ Get path to write and attach corresponding file extension. """
        if file_extension[0] != ".":
            file_extension += "."
        return os.path.join(self._result_dir, self._filename) + file_extension

    @property
    def raw_report(self):
        """ Returns raw report. """
        return self._raw_report

    def reset_raw_report(self):
        """ Reset raw report to empty dictionary. """
        self._raw_report = {}

    @property
    def result_dir(self):
        """ Return _result_dir. """
        return self._result_dir

    @result_dir.setter
    def result_dir(self, _new_result_dir: str):
        """ Set new result_dir. """
        self._result_dir = _new_result_dir

def _get_pd_dataframe(raw_report):
    """
    Map a raw report to pandas dataframe.
    :param raw_report: {op.dotted_name_op: error_report_object}.
    :return df: pandas dataframe.
    """
    df = pd.DataFrame(columns=report_const.OUTPUT_CSV_HEADER)
    for dotted_name, node_error_report_object in raw_report.items():
        for issue, recomm in node_error_report_object.get_issue_recomm_list():
            tmp_df = dict.fromkeys(report_const.OUTPUT_CSV_HEADER, 'N/A')
            tmp_df[report_const.DF_GRAPH_NODENAME] = dotted_name
            tmp_df[report_const.DF_ISSUE] = issue
            tmp_df[report_const.DF_RECOMM] = recomm
            df = pd.concat([df, pd.DataFrame([tmp_df])], ignore_index=True)

    return df

def _update_raw_report(raw_report: Dict, new_raw_report: Dict):
    """
    Merge new_raw_report to raw_report.
    :param raw_report: {op.dotted_name_op: error_report_object }
    :param new_raw_report: {op.dotted_name_op: error_report_object }
    """
    for new_key, new_error_report_object in new_raw_report.items():
        if new_key in raw_report:
            raw_report[new_key].add_failed_checks(new_raw_report[new_key].failed_checks)
        else:
            raw_report[new_key] = new_error_report_object

def _get_report_issue_recomm(check_name: str, optype: str):
    """
    Get issue, recomm for optype from report_const.
    :param check_name: Check's name.
    :param optype: optype for module that fails the check.
    :return issue and recomm from report_const.
    """
    # Set default message as undefined messages.
    issue = report_const.UNDEFINED_ISSUE.format(check_name)
    recomm = report_const.UNDEFINED_RECOMM.format(check_name)

    if check_name in report_const.ERR_MSG_DICT:
        check_error_msg = report_const.ERR_MSG_DICT[check_name]

        # DF_ISSUE key not exist implies this is a optype specified message.
        if report_const.DF_ISSUE not in check_error_msg:
            # Get new check_error_msg with optype or return default message optype not exists.
            if optype in check_error_msg:
                check_error_msg = check_error_msg[optype]
            else:
                return issue, recomm

        issue = check_error_msg[report_const.DF_ISSUE]
        recomm = check_error_msg[report_const.DF_RECOMM]

    return issue, recomm

def _generate_arch_checker_report(op: Union[Op, List[Op], List[OpStructure]], failed_check: Union[Set[str], str])-> Dict[str, NodeErrorReportObject]:
    """
    Get new_arch_checker_report with op and failed_check.
    :param op: Op for node_check or list of Op for pattern check.
    :param failed_check: Set of failed check's name for node_check or str of a failed check.
    """
    # Node check returns a single Op with set of str(failed_check.__name__)
    if isinstance(op, Op):
        return {op.dotted_name_op: NodeErrorReportObject(op, failed_check)}

    # Pattern check that marks nodes returns list of Ops and a failed check.
    return {_op.dotted_name_op: NodeErrorReportObject(_op, {failed_check}) for _op in op}

def get_node_check_dict()-> Dict:
    """
    Get dictionary for node checks.
    :return check_dicts: {check target type: list of checks}.
    """
    return NODE_CHECK_DICT.copy()

def get_pattern_check_list()-> List:
    """
    Get a list of pattern checks.
    :return: List of pattern checks.
    """
    return PATTERN_CHECK_LIST.copy()

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

def check_type_deco(module_type):
    """
    Apply type check before doing node check.
    e.g., add node check for relu, results in add node check to TorchActivations. But this check is
    relu exclusive. Use this method to prevent node check being applied to other TorchActivations.
    :param module_type: type to pass this function.
    :return check_type_and_return_result: Return True if node is not right type for the func, return
                        func(node) is node is the right type.
    """
    def check_type_and_return_result(func):
        """
        :param func: function to be executed if isinstance(node, module_type)
        :return check_type: func(node) if node has type module_type. Return True when the func is
                    bypassed.
        """
        def check_type(node):
            if isinstance(node, module_type):
                return func(node)
            return True

        # Overwrite check_type's name with func.
        check_type.__name__ = func.__name__
        return check_type

    return check_type_and_return_result
