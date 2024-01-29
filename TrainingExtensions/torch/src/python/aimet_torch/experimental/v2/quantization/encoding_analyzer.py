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
import numpy as np
from aimet_torch.experimental.v2.utils import reduce, StatisticsNotFoundError


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
        self.stats = _Histogram()
        self.num_bins = num_bins
        self.stats = _Histogram()
    
    @torch.no_grad()
    def update_stats(self, input_tensor: torch.Tensor) -> _Statistics:
        new_stats = self.observer.collect_stats(input_tensor)
        self.observer.merge_stats(new_stats, input_tensor)

    def _create_histogram(self, input_tensor, min_input, max_input):
        if torch.is_tensor(min_input):
            min_input = min_input.numpy()[0]
        
        if torch.is_tensor(max_input):
            max_input = max_input.numpy()[0]
        if len(self.shape) == 1:
            histogram, bin_edges = torch.histogram(input_tensor, self.num_bins, range=(min_input, max_input))
        else:
            histogram, bin_edges = torch.histogramdd(input_tensor, self.num_bins, range=(min_input, max_input))
        
        return histogram, bin_edges
    
    @torch.no_grad()
    def collect_stats(self, input_tensor: torch.Tensor) -> _Histogram:
        min_input = reduce(input_tensor, shape=self.shape, reduce_op=torch.min).values
        max_input = reduce(input_tensor, shape=self.shape, reduce_op=torch.max).values
        histogram, bin_edges = self._create_histogram(input_tensor, min_input, max_input)
       
        # optimize min/max values
        updated_min, updated_max = self.determine_optimal_params(_Histogram(histogram, bin_edges, min_input, max_input))
        updated_histogram, updated_bin_edges = self._create_histogram(input_tensor, updated_min, updated_max)
        return _Histogram(updated_histogram, updated_bin_edges, updated_min, updated_max)
    
    def _get_bin_num(self, bin_width: int, curr_min, curr_max):
        if bin_width:
            return min(int((curr_max - curr_min) / bin_width), self.num_bins - 1)
        return bin_width
    
    @torch.no_grad()
    def merge_stats(self, new_stats: _Histogram, input_tensor: torch.Tensor):
        if self.stats.min is None and self.stats.max is None:
            self.stats = new_stats
            return
        
        updated_min = min(new_stats.min, self.stats.min)
        updated_max = max(new_stats.max, self.stats.max)

        # find min/max after merging
        updated_min, updated_max = self.determine_optimal_params(_Histogram(expanded_histogram, None, updated_min, updated_max))

        # if the current histogram can capture new_stats within in its range
        if updated_min == self.stats.min and updated_max == self.stats.max:
            histogram_updates = self.stats.histogram
        else:
            dest_bin_width = (updated_max - updated_min) / self.num_bins
            src_bin_width = (self.stats.max - self.stats.min) / self.num_bins
            histogram_updates = np.zeros(self.num_bins)
        
            for curr_bin in range(self.num_bins):
                curr_hist = self.stats.histogram[curr_bin]
                if curr_hist:
                    src_bin_start = self.stats.min + src_bin_width * curr_bin
                    dest_bin_start = (src_bin_start - updated_min) / dest_bin_width
                    dest_bin_end = updated_min + dest_bin_width * (dest_bin_start + 1)
                    dest_bin_updated = min(torch.round((dest_bin_end - src_bin_start) / src_bin_width * curr_hist), curr_hist)
                    bin_index = self._get_bin_num(dest_bin_width, updated_min, src_bin_start)
                    histogram_updates[bin_index] += dest_bin_updated
                
                    if dest_bin_updated < curr_hist:
                        bin_index = self._get_bin_num(dest_bin_width, updated_min, src_bin_start + dest_bin_width)
                        histogram_updates[bin_index] += curr_hist - dest_bin_updated
            
        # create histogram given input tensor and full range
        expanded_histogram, expanded_bin_edges = self._create_histogram(input_tensor, updated_min, updated_max)
        expanded_histogram += histogram_updates

        self.stats = _Histogram(expanded_histogram, expanded_bin_edges, updated_min, updated_max)

    def _get_norm(
        self, delta_begin: torch.Tensor, delta_end: torch.Tensor, density: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute the norm of the values uniformaly distributed between
        delta_begin and delta_end.

        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
        norm = (
            delta_end * delta_end * delta_end - delta_begin * delta_begin * delta_begin
        ) / 3
        return density * norm
    
    def compute_quantization_error(self, stats: _Histogram, next_start_bin: int, next_end_bin: int):
        bin_width = (stats.max - stats.min) / self.num_bins

        dest_bins =  2 ** torch.iinfo(stats.max.dtype).bits
        dest_bins_width = bin_width * (next_end_bin - next_start_bin + 1) / dest_bins
        if dest_bins_width == 0.0:
            return 0.0

        src_bin = torch.arange(self.num_bins, device=self.histogram.device)
        # distances from the beginning of first dst_bin to the beginning and
        # end of src_bin
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width

        # which dest_bins the beginning and end of src_bin belong to?
        dst_bin_of_begin = torch.clamp(
            torch.div(src_bin_begin, dest_bins_width, rounding_mode='floor'), 0, self.dst_nbins - 1
        )
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dest_bins_width

        dst_bin_of_end = torch.clamp(
            torch.div(src_bin_end, dest_bins_width, rounding_mode='floor'), 0, self.dst_nbins - 1
        )
        density = self.histogram / bin_width

        norm = torch.zeros(self.bins, device=self.histogram.device)

        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dest_bins_width / 2
        norm += self._get_norm(delta_begin,
                               torch.ones(self.bins, device=self.histogram.device) * delta_end,
                               density)

        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self._get_norm(
            torch.tensor(-dest_bins_width / 2), torch.tensor(dest_bins_width / 2), density
        )

        dst_bin_of_end_center = dst_bin_of_end * dest_bins_width + dest_bins_width / 2

        delta_begin = -dest_bins_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += self._get_norm(torch.tensor(delta_begin), delta_end, density)

        return norm.sum().item()
    
    @torch.no_grad()
    def determine_optimal_params(self, new_stats: _Histogram):
        # Based on PyTorch's parameter search which minimizes L2 error
        bin_width = (new_stats.max - new_stats.min) / self.num_bins
        total = torch.sum(new_stats.histogram).item()
        cumulative_sum = torch.cumsum(new_stats.histogram, dim=0)

        stepsize = 1e-5  # granularity
        alpha = 0.0  # lower bound
        beta = 1.0  # upper bound
        start_bin = 0
        end_bin = self.num_bins - 1
        norm_min = float("inf")

        while alpha < beta:
            # Find the next step
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize

             # find the left and right bins between the quantile bounds
            left = start_bin
            right = end_bin
            while left < end_bin and cumulative_sum[left] < next_alpha * total:
                left = left + 1
            while right > start_bin and cumulative_sum[right] > next_beta * total:
                right = right - 1

            # decide the next move
            next_start_bin = start_bin
            next_end_bin = end_bin
            if (left - start_bin) > (end_bin - right):
                # move the start bin
                next_start_bin = left
                alpha = next_alpha
            else:
                # move the end bin
                next_end_bin = right
                beta = next_beta

            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue
            
            # calculate the quantization error using next_start_bin and next_end_bin
            norm = self.compute_quantization_error(new_stats, next_start_bin, next_end_bin)

            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        new_min = self.min_val + bin_width * start_bin
        new_max = self.min_val + bin_width * (end_bin + 1)
        return new_min, new_max
    
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
        if num_bins <= 0:
            raise ValueError('Number of bins cannot be less than or equal to 0.')
        observer = _HistogramObserver(min_max_shape=shape, num_bins=num_bins)
        super().__init__(observer)

    @torch.no_grad()
    def update_stats(self, input_tensor: torch.Tensor) -> _Statistics:
        new_stats = self.observer.collect_stats(input_tensor)
        self.observer.merge_stats(new_stats, input_tensor)
        return new_stats
    
    @torch.no_grad()
    def compute_encodings_from_stats(self, stats: _Histogram, bitwidth: int, is_symmetric: bool, percentile: float)\
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # TODO
        raise NotImplementedError

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

class MseEncodingAnalyzer(EncodingAnalyzer[_Histogram]):
    """
    Encoding Analyzer for Mean Square Error (MSE) Calibration technique
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
