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

import unittest
from unittest.mock import create_autospec

import aimet_torch.svd.svd_intf_defs_deprecated
from aimet_common.utils import AimetLogger
from aimet_torch.svd.model_stats_calculator import ModelStats as ms
from aimet_common import cost_calculator as cc

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


class TestModelStatsCalculator(unittest.TestCase):
    def test_compute_compression_ratio(self):
        logger.debug(self.id())
        network_cost = cc.Cost(50, 100)

        with unittest.mock.patch('aimet_common.cost_calculator.CostCalculator.compute_network_cost') as mock_func:
            mock_func.return_value = cc.Cost(40, 50)
            ratio = ms.compute_compression_ratio(None, aimet_torch.svd.svd_intf_defs_deprecated.CostMetric.memory, network_cost)

        self.assertEqual(0.2, ratio)

    def test_compute_objective_score(self):
        obj_score = ms.compute_objective_score(model_perf=0.2, compression_score=1.2, error_margin =1, baseline_perf=1)
        self.assertEqual(0.8, obj_score)
