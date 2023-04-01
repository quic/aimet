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
from typing import Dict, List, Union, Set, Tuple, Callable
import pandas as pd

from aimet_torch.meta.operation import Op
from aimet_torch.arch_checker.constants import ArchCheckerReportConstants as report_const

from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

class PatternHandler():
    """ Object to handle pattern checkes. """
    def __init__(self, check: Callable):
        self.check = check

    def __call__(self, *args, **kwargs):
        """
        Run pattern check on PatternType_object, op_subset.
        """
        _, op_subset = args
        self.check(op_subset)

class NodeErrorReportObject:
    """ Error report object for each op. """
    def __init__(self, op, failed_checks: Set[str]) -> None:
        self.dotted_name = op.dotted_name_op
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
    def __init__(self) -> None:
        self._raw_report = {}
        self._export_path = "arch_checker_report.csv"

    def merge_new_raw_report(self, new_raw_report):
        """
        Merge new_raw_report to self.raw_report.
        :param new_raw_report: {op.dotted_name: NodeErrorReportObject }
        """
        _update_raw_report(self._raw_report, new_raw_report)

    def update_raw_report(self, op: Union[List, List[Op]], failed_check: Union[Set[str], str]):
        """
        Update raw_report with op and failed check.
        :param op: Op or list of Ops.
        :param failed_check: failed_check.__name__ or set(failed_check.__name__)
        """
        _update_raw_report(self._raw_report, _generate_arch_checker_report(op, failed_check))

    def export_checker_report_to_cvs(self, path: str = None):
        """
        Map raw report to issue and recommendations in report_const.ERR_MSG_DICT to form
        arch_checker_report and write csv file to self._export_path.
        """
        if path is not None:
            self.export_path = path

        df = pd.DataFrame(columns=report_const.OUTPUT_CSV_HEADER)

        for dotted_name, node_error_report_object in self._raw_report.items():
            for issue, recomm in node_error_report_object.get_issue_recomm_list():
                tmp_df = dict.fromkeys(report_const.OUTPUT_CSV_HEADER, 'N/A')
                tmp_df[report_const.DF_GRAPH_NODENAME] = dotted_name
                tmp_df[report_const.DF_ISSUE] = issue
                tmp_df[report_const.DF_RECOMM] = recomm
                df = pd.concat([df, pd.DataFrame([tmp_df])], ignore_index=True)

        logger.info("Save arch_checker report to %s", self._export_path)
        df.to_csv(self._export_path, sep='\t', index=True)

    @property
    def export_path(self):
        """ Returns export path. """
        return self._export_path

    @export_path.setter
    def export_path(self, path):
        """ Sets the export path. """
        _name, _extension = os.path.splitext(path)
        if _extension != ".csv":
            logger.info("Got ArchCheckerReport.export_path: \"%s\" is not csv file: ", path)
            path = os.path.join(_name, ".csv")
            logger.info("Overwrite ArchCheckerReport.export_path to: \"%s\"", path)

        logger.info("Set arch_checker_report path to %s", self._export_path)
        self._export_path = path

    @property
    def raw_report(self):
        """ Returns raw report. """
        return self._raw_report

    def reset_raw_report(self):
        """ Reset raw report to empty dictionary. """
        self._raw_report = {}

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

def _generate_arch_checker_report(op: Union[List, List[Op]], failed_check: Union[Set[str], str])-> Dict[str, NodeErrorReportObject]:
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
