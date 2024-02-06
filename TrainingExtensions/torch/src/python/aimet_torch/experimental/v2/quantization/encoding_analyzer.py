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
from typing import TypeVar, Generic, Tuple, Optional, List
import itertools
import torch
from aimet_torch.experimental.v2.utils import reduce, StatisticsNotFoundError, _is_expandable


@dataclass
class _MinMaxRange:
    min: Optional[torch.Tensor] = None
    max: Optional[torch.Tensor] = None

@dataclass
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
        if isinstance(shape, int):
            shape = (shape,)
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
        self.num_bins = num_bins
        self.num_histograms = torch.prod(torch.Tensor(self.shape), dtype=int).item()
        self.stats = []
        for _ in range(self.num_histograms):
            self.stats.append(_Histogram())


    @torch.no_grad()
    def collect_stats(self, input_tensor: torch.Tensor) -> List[_Histogram]:
        if not _is_expandable(self.shape, input_tensor.shape):
            raise RuntimeError(f"Shape {self.shape} is incompatible with input of shape {input_tensor.shape}")

        hist_stats = []
        input_shape = tuple(input_tensor.shape)
        histogram_shape = self.shape

        padded_histogram_shape = (
            *itertools.repeat(1, len(input_shape) - len(histogram_shape)),
            *histogram_shape
        )

        for hist_num in range(self.num_histograms):
            hist_input = input_tensor

            for axis, dim in enumerate(padded_histogram_shape):
                if dim == 1:
                    continue
                # elements in current axis, ex: could be W*C, C, or 1 for input_shape [H, W, C]
                numel = torch.prod(torch.Tensor(padded_histogram_shape[axis+1:]), dtype=int)
                # index where hist_input at current dimension will be sliced at
                index = (hist_num // numel) % dim
                hist_input = torch.unsqueeze(torch.select(hist_input, axis, index), axis)

            histogram, bin_edges = torch.histogram(hist_input.to(torch.float), self.num_bins)
            hist_stats.append(_Histogram(histogram, bin_edges, hist_input.min(), hist_input.max()))

        return hist_stats

    def _get_bin_num(self, bin_width: int, curr_min, data):
        if bin_width:
            return min(int((data - curr_min) / bin_width), self.num_bins - 1)
        return bin_width

    # pylint: disable=arguments-differ
    # pylint: disable=too-many-locals
    @torch.no_grad()
    def merge_stats(self, new_stats_list: List[_Histogram], input_tensor: torch.Tensor):
        if self.stats[0].histogram is None:
            self.stats = new_stats_list
            return

        hist_inputs = torch.reshape(input_tensor, (len(new_stats_list), -1))

        for index, new_stats in enumerate(new_stats_list):
            curr_stats = self.stats[index]
            curr_input = hist_inputs[index]

            updated_min = min(new_stats.min, curr_stats.min)
            updated_max = max(new_stats.max, curr_stats.max)

            # if the current histogram can capture new_stats within in its range
            if updated_min == curr_stats.min and updated_max == curr_stats.max:
                histogram_updates = curr_stats.histogram
            else:
                dest_bin_width = (updated_max - updated_min) / self.num_bins
                src_bin_width = (curr_stats.max - curr_stats.min) / self.num_bins
                histogram_updates = torch.zeros(self.num_bins)

                for curr_bin in range(self.num_bins):
                    curr_hist = curr_stats.histogram[curr_bin]
                    if curr_hist:
                        src_bin_start = curr_stats.min + src_bin_width * curr_bin
                        bin_index = self._get_bin_num(dest_bin_width, updated_min, src_bin_start)
                        dest_bin_end = updated_min + dest_bin_width * (bin_index + 1)

                        # split curr_hist if values in source bin cannot neatly fold into dest bin
                        split_hist_value = torch.round(((dest_bin_end - src_bin_start) / src_bin_width) * curr_hist)
                        dest_bin_updated = min(split_hist_value, curr_hist)
                        # update appropriate bin with either the full or split curr_hist value
                        histogram_updates[bin_index] += dest_bin_updated
                        # if curr_hist is split, update other bin that the remaining values fall into
                        if dest_bin_updated < curr_hist:
                            bin_index = self._get_bin_num(dest_bin_width, updated_min, src_bin_start + dest_bin_width)
                            histogram_updates[bin_index] += curr_hist - dest_bin_updated
            # create histogram given input tensor and full range
            expanded_histogram, expanded_bin_edges = torch.histogram(curr_input, self.num_bins, range=(updated_min.item(), updated_max.item()))
            expanded_histogram += histogram_updates
            self.stats[index] = _Histogram(expanded_histogram, expanded_bin_edges, updated_min, updated_max)

    def reset_stats(self):
        self.stats = []
        for _ in range(self.num_histograms):
            self.stats.append(_Histogram())

    def get_stats(self) -> List[_Histogram]:
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
    def __init__(self, shape: tuple):
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


def adjust_min_max(curr_min, curr_max, bitwidth, is_symmetric):
    # ensure that 0 is in the range
    curr_min = torch.minimum(curr_min, torch.zeros_like(curr_min))
    curr_max = torch.maximum(curr_max, torch.zeros_like(curr_max))

    # ensure that min/max are finite
    curr_min.clamp_(min=torch.finfo(curr_max.dtype).min, max=0)
    curr_max.clamp_(min=0, max=torch.finfo(curr_max.dtype).max)

    # ensure that min/max aren't too close
    tiny_num = torch.finfo(curr_min.dtype).tiny
    tensor_threshold = (curr_max - curr_min) / ((2 **bitwidth) - 1)
    curr_min[tensor_threshold < tiny_num] -= tiny_num * (2 **(bitwidth - 1))
    curr_max[tensor_threshold < tiny_num] += tiny_num * ((2 **(bitwidth - 1)) - 1)

    if is_symmetric:
        symmetric_min = torch.minimum(curr_min, -curr_max)
        symmetric_max = torch.maximum(-curr_min, curr_max)
        return symmetric_min, symmetric_max

    return curr_min, curr_max

# pylint: disable=arguments-differ
class PercentileEncodingAnalyzer(EncodingAnalyzer[_Histogram]):
    """
    Encoding Analyzer for Percentile calibration technique
    """
    def __init__(self, shape: tuple, num_bins: int = 2048):
        if num_bins <= 0:
            raise ValueError('Number of bins cannot be less than or equal to 0.')
        observer = _HistogramObserver(shape=shape, num_bins=num_bins)
        super().__init__(observer)

    @torch.no_grad()
    def update_stats(self, input_tensor: torch.Tensor) -> _Statistics:
        new_stats = self.observer.collect_stats(input_tensor)
        self.observer.merge_stats(new_stats, input_tensor)
        return new_stats

    def compute_dynamic_encodings(self, input_tensor: torch.Tensor, bitwidth: int,\
                                  is_symmetric: bool, percentile: float)-> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.compute_encodings_from_stats(
            self.observer.collect_stats(input_tensor), bitwidth, is_symmetric, percentile)

    def compute_encodings(self, bitwidth: int, is_symmetric: bool, percentile: float) -> torch.Tensor:
        return self.compute_encodings_from_stats(self.observer.get_stats(), bitwidth, is_symmetric, percentile)

    # pylint: disable=too-many-locals
    @torch.no_grad()
    def compute_encodings_from_stats(self, stats: List[_Histogram], bitwidth: int, is_symmetric: bool, percentile: float)\
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

        if bitwidth <= 0:
            raise ValueError('Bitwidth cannot be less than or equal to 0.')

        if percentile < 50 or percentile > 100:
            raise ValueError('Percentile value must be within 50-100 range')

        if stats[0].histogram is None:
            raise StatisticsNotFoundError('No statistics present to compute encodings.')

        encoding_min_list = []
        encoding_max_list = []

        for list_elem in stats:
            cum_sum = torch.cumsum(list_elem.histogram, dim=0)
            # trim percentile value from min and max
            max_index = torch.searchsorted(cum_sum, torch.quantile(cum_sum, percentile/100))
            min_index = torch.searchsorted(cum_sum, torch.quantile(cum_sum, 1 - percentile/100))

            if percentile == 100:
                min_index = 0
                max_index = -1
            curr_min = list_elem.bin_edges[min_index]
            curr_max = list_elem.bin_edges[max_index]
            # adjust min/max
            updated_min, updated_max = adjust_min_max(curr_min, curr_max, bitwidth, is_symmetric)
            encoding_min_list.append(updated_min.item())
            encoding_max_list.append(updated_max.item())

        encoding_min = torch.Tensor(encoding_min_list)
        encoding_min = torch.reshape(encoding_min, self.observer.shape)

        encoding_max = torch.Tensor(encoding_max_list)
        encoding_max = torch.reshape(encoding_max, self.observer.shape)
        return encoding_min, encoding_max

class SqnrEncodingAnalyzer(EncodingAnalyzer[_Histogram]):
    """
    Encoding Analyzer for SQNR Calibration technique
    """
    def __init__(self, shape: tuple, num_bins: int = 2048):
        if num_bins <= 0:
            raise ValueError('Number of bins cannot be less than or equal to 0.')
        observer = _HistogramObserver(shape=shape, num_bins=num_bins)
        super().__init__(observer)

    @torch.no_grad()
    def update_stats(self, input_tensor: torch.Tensor) -> _Statistics:
        new_stats = self.observer.collect_stats(input_tensor)
        self.observer.merge_stats(new_stats, input_tensor)
        return new_stats

    @torch.no_grad()
    def compute_encodings_from_stats(self, stats: _Histogram, bitwidth: int, is_symmetric: bool)\
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # TODO
        raise NotImplementedError
