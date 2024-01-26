# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 Qualcomm Innovation Center, Inc. All rights reserved.
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
# pylint: disable=redefined-builtin
# pylint: disable=missing-docstring

""" Computes statistics and encodings """

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic, Tuple, Optional
import torch
from aimet_torch.experimental.v2.utils import reduce, StatisticsNotFoundError


@dataclass
class _MinMaxRange:
    min: Optional[torch.Tensor] = None
    max: Optional[torch.Tensor] = None

class _Histogram:
    histogram: torch.Tensor = None
    bin_edges: torch.Tensor = None
    min: Optional[torch.Tensor] = None
    max: Optional[torch.Tensor] = None

_Statistics = TypeVar('_Statistics', _MinMaxRange, _Histogram)

class _Observer(Generic[_Statistics], ABC):
    """
    Observes and gathers statistics
    """
    def __init__(self, shape: tuple):
        self.shape = shape

    @abstractmethod
    def collect_stats(self, input_tensor: torch.Tensor) -> _Statistics:
        pass

    @abstractmethod
    def merge_stats(self, stats: _Statistics):
        pass

    @abstractmethod
    def reset_stats(self):
        pass

    @abstractmethod
    def get_stats(self) -> _Statistics:
        pass


class _MinMaxObserver(_Observer[_MinMaxRange]):
    """
    Observer for Min-Max calibration technique
    """
    def __init__(self, shape: tuple):
        super().__init__(shape)
        self.stats = _MinMaxRange()

    @torch.no_grad()
    def collect_stats(self, input_tensor: torch.Tensor) -> _MinMaxRange:
        new_min = reduce(input_tensor, shape=self.shape, reduce_op=torch.min).values
        new_max = reduce(input_tensor, shape=self.shape, reduce_op=torch.max).values
        return _MinMaxRange(new_min, new_max)

    @torch.no_grad()
    def merge_stats(self, stats: _MinMaxRange):
        updated_min = self.stats.min
        if stats.min is not None:
            if updated_min is None:
                updated_min = stats.min.clone()
            else:
                updated_min = torch.minimum(updated_min, stats.min)

        updated_max = self.stats.max
        if stats.max is not None:
            if updated_max is None:
                updated_max = stats.max.clone()
            else:
                updated_max = torch.maximum(updated_max, stats.max)

        self.stats = _MinMaxRange(updated_min, updated_max)

    def reset_stats(self):
        self.stats = _MinMaxRange()

    def get_stats(self) -> _MinMaxRange:
        return self.stats

class _HistogramObserver(_Observer[_Histogram]):
    """
    Observer for Histogram based calibration techniques (percentile, MSE)
    """
    def __init__(self, shape: tuple, num_bins: int):
        super().__init__(shape)
        self.stats = _Histogram()
        self.num_bins = num_bins

    @torch.no_grad()
    def collect_stats(self, input_tensor: torch.Tensor) -> _Histogram:
        # TODO
        raise NotImplementedError

    @torch.no_grad()
    def merge_stats(self, stats: _Histogram):
        # TODO
        raise NotImplementedError

    def reset_stats(self):
        self.stats = _Histogram()

    def get_stats(self) -> _Histogram:
        return self.stats

class EncodingAnalyzer(Generic[_Statistics], ABC):
    def __init__(self, observer: _Observer):
        self.observer = observer

    @torch.no_grad()
    def update_stats(self, input_tensor: torch.Tensor) -> _Statistics:
        new_stats = self.observer.collect_stats(input_tensor)
        self.observer.merge_stats(new_stats)
        return new_stats

    def reset_stats(self) -> None:
        self.observer.reset_stats()

    def compute_encodings(self, bitwidth: int, is_symmetric: bool) -> torch.Tensor:
        return self.compute_encodings_from_stats(self.observer.get_stats(), bitwidth, is_symmetric)

    def compute_dynamic_encodings(self, input_tensor: torch.Tensor, bitwidth: int,\
                                  is_symmetric: bool)-> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.compute_encodings_from_stats(
            self.observer.collect_stats(input_tensor), bitwidth, is_symmetric)

    @abstractmethod
    def compute_encodings_from_stats(self, stats: _Statistics, bitwidth: int, is_symmetric: bool)\
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        pass

class MinMaxEncodingAnalyzer(EncodingAnalyzer[_MinMaxRange]):
    """
    Encoding Analyzer for Min-Max calibration technique
    """
    def __init__(self, shape):
        observer = _MinMaxObserver(shape)
        super().__init__(observer)

    #pylint: disable=too-many-locals
    @torch.no_grad()
    def compute_encodings_from_stats(self, stats: _MinMaxRange, bitwidth: int, is_symmetric: bool)\
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if bitwidth <= 0:
            raise ValueError('Bitwidth cannot be less than or equal to 0.')

        if stats.min is None or stats.max is None:
            raise StatisticsNotFoundError('No statistics present to compute encodings.')

        updated_min = stats.min
        updated_max = stats.max

        tiny_num = torch.finfo(stats.min.dtype).tiny
        # enforces that 0 is within the min/max
        min_with_zero = torch.minimum(stats.min, torch.zeros_like(stats.min))
        max_with_zero = torch.maximum(stats.max, torch.zeros_like(stats.max))

         # adjusts any min/max pairing that are too close
        tensor_diff = (max_with_zero - min_with_zero) / ((2 **bitwidth) - 1)
        update_min = torch.where(tensor_diff < tiny_num, tiny_num * (2 **(bitwidth - 1)), 0.0)
        update_max = torch.where(tensor_diff < tiny_num, tiny_num * ((2 **(bitwidth - 1)) - 1), 0.0)
        updated_max = max_with_zero + update_max
        updated_min = min_with_zero - update_min

        # replace pos and neg inf respectively
        updated_max[torch.isposinf(updated_max)] = torch.finfo(stats.min.dtype).max
        updated_min[torch.isposinf(updated_min)] = torch.finfo(stats.min.dtype).max
        updated_max[torch.isneginf(updated_max)] = -torch.finfo(stats.min.dtype).max
        updated_min[torch.isneginf(updated_min)] = -torch.finfo(stats.min.dtype).max

        if is_symmetric:
            # ensures that min/max pairings are symmetric
            symmetric_min = torch.minimum(updated_min, -updated_max)
            symmetric_max = torch.maximum(-updated_min, updated_max)
            return symmetric_min, symmetric_max

        return updated_min, updated_max

class PercentileEncodingAnalyzer(EncodingAnalyzer[_Histogram]):
    """
    Encoding Analyzer for Percentile calibration technique
    """
    def __init__(self, shape: tuple, num_bins: int = 2048):
        observer = _HistogramObserver(shape=shape, num_bins=num_bins)
        super().__init__(observer)

    @torch.no_grad()
    def compute_encodings_from_stats(self, stats: _Histogram, bitwidth: int, is_symmetric: bool)\
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # TODO
        raise NotImplementedError

class SqnrEncodingAnalyzer(EncodingAnalyzer[_Histogram]):
    """
    Encoding Analyzer for SQNR Calibration technique
    """
    def __init__(self, shape: tuple, num_bins: int = 2048):
        observer = _HistogramObserver(shape=shape, num_bins=num_bins)
        super().__init__(observer)

    @torch.no_grad()
    def compute_encodings_from_stats(self, stats: _Histogram, bitwidth: int, is_symmetric: bool)\
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # TODO
        raise NotImplementedError

class MseEncodingAnalyzer(EncodingAnalyzer[_Histogram]):
    """
    Encoding Analyzer for Mean Square Error (MSE) Calibration technique
    """
    def __init__(self, shape: tuple, num_bins: int = 2048):
        observer = _HistogramObserver(shape=shape, num_bins=num_bins)
        super().__init__(observer)

    @torch.no_grad()
    def compute_encodings_from_stats(self, stats: _Histogram, bitwidth: int, is_symmetric: bool)\
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # TODO
        raise NotImplementedError
