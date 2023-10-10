# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Provides an interface for rounding rank or channels """
import abc
from decimal import Decimal

from aimet_common.defs import CostMetric
from aimet_common.layer_database import Layer
from aimet_common import utils


class CompRatioRounder(abc.ABC):
    """
    Models a ML Model Pruner
    """
    @abc.abstractmethod
    def round(self, layer: Layer, comp_ratio: Decimal, cost_metric: CostMetric) -> Decimal:
        """
        Rounds comp_ratio to nearest multiplicity
        :param layer: Layer
        :param comp_ratio: comp_ratio to be rounded
        :param cost_metric:
        :return: Updated comp_ratio
        """


class RankRounder(CompRatioRounder):
    """
    Rounds rank and finds the corresponding updated compression ratio for SVD
    """
    def __init__(self, multiplicity: int, cost_calculator):
        """
        :param multiplicity: Multiplicity to which rank is rounded-up
        :param cost_calculator: SpatialSvdCostCalculator() or WeightSvdCostCalculator()
        """
        self._multiplicity = multiplicity
        self._cost_calculator = cost_calculator

    def round(self, layer: Layer, comp_ratio: Decimal, cost_metric: CostMetric) -> Decimal:

        if self._multiplicity == 1:
            return comp_ratio

        # Find rank corresponding to a compression ratio
        rank = self._cost_calculator.calculate_rank_given_comp_ratio(layer, comp_ratio, cost_metric)

        # Finding rank corresponding to comp ratio 1
        max_rank = self._cost_calculator.calculate_rank_given_comp_ratio(layer, 1.0, cost_metric)

        rounded_rank_candidate = utils.round_up_to_multiplicity(self._multiplicity, rank, max_rank)
        if rank == rounded_rank_candidate:
            updated_comp_ratio = comp_ratio
        else:
            # For the rounded rank compute the new compression ratio
            updated_comp_ratio = self._cost_calculator.calculate_comp_ratio_given_rank(layer, rounded_rank_candidate
                                                                                       , cost_metric)

        assert 0 <= updated_comp_ratio <= 1
        assert comp_ratio <= updated_comp_ratio

        return updated_comp_ratio


class ChannelRounder(CompRatioRounder):
    """
    Rounds input channels to be kept and finds the corresponding updated compression ratio for Channel Pruning
    """
    def __init__(self, multiplicity: int):
        """
        :param multiplicity: Multiplicity to which rank is rounded-up
        """
        self._multiplicity = multiplicity

    def round(self, layer: Layer, comp_ratio: Decimal, cost_metric: CostMetric)->Decimal:

        if self._multiplicity == 1:
            updated_comp_ratio = comp_ratio
        else:
            # get number of input channels to keep
            in_channels = layer.weight_shape[1]
            keep_inp_channels = in_channels * comp_ratio
            # Round input channels
            keep_inp_channels = utils.round_up_to_multiplicity(self._multiplicity, keep_inp_channels, in_channels)

            # Find updated compression ratio
            updated_comp_ratio = Decimal(keep_inp_channels) / Decimal(in_channels)
            assert comp_ratio <= updated_comp_ratio
            assert 0 <= updated_comp_ratio <= 1

        return updated_comp_ratio
