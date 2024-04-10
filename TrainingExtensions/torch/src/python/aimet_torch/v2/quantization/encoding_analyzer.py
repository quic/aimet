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
from aimet_torch.v2.utils import reduce, StatisticsNotFoundError, _is_expandable


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


    # pylint: disable=too-many-locals
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

            hist_min, hist_max = self._handle_inf_inputs(hist_input)
            bin_edges = self._create_bin_edges(min_val=hist_min, max_val=hist_max, device=input_tensor.device)
            histogram = torch.histc(hist_input.to(torch.float), bins=self.num_bins, min=bin_edges[0], max=bin_edges[-1])

            # clip inf values to hist_min and hist_max and adjust for any fp errors
            histogram[0] += torch.sum(hist_input < bin_edges[0])
            histogram[-1] += torch.sum(hist_input > bin_edges[-1])

            hist_stats.append(_Histogram(histogram, bin_edges, hist_min, hist_max))

        return hist_stats

    # pylint: disable=no-self-use
    def _handle_inf_inputs(self, hist_input):
        if torch.all(torch.isinf(hist_input)):
            raise ValueError('Input tensor cannot contain only infinite values')

        min = hist_input[hist_input.isfinite()].min()
        max = hist_input[hist_input.isfinite()].max()

        return min, max

    def _create_bin_edges(self, min_val, max_val, device):
        # Adjust min/max values to be in line with PyTorch's torch.histc implementation
        if max_val == min_val:
            min_val = min_val - 0.5
            max_val = max_val + 0.5

        min_val, max_val = min_val.float(), max_val.float()

        step = (max_val - min_val) / self.num_bins

        return torch.arange(0, self.num_bins + 1, device=device) * step + min_val

    def _get_bin_num(self, bin_width: int, curr_min, data):
        bin_tensor = torch.full(data.shape, self.num_bins - 1, device=data.device)
        index_tensor = (data - curr_min) / bin_width
        return torch.minimum(index_tensor.to(torch.int32), bin_tensor)

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
                histogram_updates = torch.zeros(self.num_bins).to(input_tensor.device)

                src_bin_start = curr_stats.min + (src_bin_width * torch.arange(0, self.num_bins, device=input_tensor.device))
                dest_bin_index = self._get_bin_num(dest_bin_width, updated_min, src_bin_start)
                dest_bin_end = updated_min + dest_bin_width * (dest_bin_index + 1)

                # split curr_hist if values in source bin cannot neatly fold into dest bin
                split_hist_value = torch.round(((dest_bin_end - src_bin_start) / src_bin_width) * curr_stats.histogram)
                dest_bin_updates = torch.minimum(split_hist_value, curr_stats.histogram)

                # update appropriate bin with either the full or split curr_hist value
                for i, dest_bin in enumerate(dest_bin_index):
                    histogram_updates[dest_bin] += dest_bin_updates[i]

                # if curr_hist is split, update other bin that the remaining values fall into
                other_bins = torch.nonzero(torch.where(dest_bin_updates < curr_stats.histogram, 1, 0))
                other_bin_index = self._get_bin_num(dest_bin_width, updated_min, src_bin_start + dest_bin_width)
                other_bin_updates = curr_stats.histogram - dest_bin_updates
                for bin_num in other_bins:
                    histogram_updates[other_bin_index[bin_num]] += other_bin_updates[bin_num]

            # create histogram given input tensor and full range
            expanded_histogram = torch.histc(curr_input.to(torch.float), bins=self.num_bins, min=updated_min, max=updated_max)
            expanded_histogram += histogram_updates.to(expanded_histogram.device)

            # clip inf values to hist_min and hist_max
            expanded_histogram[0] += torch.sum(curr_input == -float('inf'))
            expanded_histogram[-1] += torch.sum(curr_input == float('inf'))

            expanded_bin_edges = self._create_bin_edges(min_val=updated_min, max_val=updated_max, device=expanded_histogram.device)
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
    def __init__(self, shape: tuple, num_bins: int = 2048, percentile: float = 100):
        if num_bins <= 0:
            raise ValueError('Number of bins cannot be less than or equal to 0.')

        observer = _HistogramObserver(shape=shape, num_bins=num_bins)
        super().__init__(observer)
        self.set_percentile(percentile)

    def set_percentile(self, percentile):
        """
        Set the clipping percentile of the encoding analyzer. The encoding analyzer will clip the (100% - percentile)
        largest and smallest observed values from the encoding range when computing encodings.

        :param percentile: Value from 50.0 to 100.0 indicating the clipping percentile
        """
        if percentile < 50 or percentile > 100:
            raise ValueError('Percentile value must be within 50-100 range')

        self.percentile = percentile

    @torch.no_grad()
    def update_stats(self, input_tensor: torch.Tensor) -> _Statistics:
        new_stats = self.observer.collect_stats(input_tensor)
        self.observer.merge_stats(new_stats, input_tensor)
        return new_stats

    # pylint: disable=too-many-locals
    @torch.no_grad()
    def compute_encodings_from_stats(self, stats: List[_Histogram], bitwidth: int, is_symmetric: bool)\
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

        if bitwidth <= 0:
            raise ValueError('Bitwidth cannot be less than or equal to 0.')

        if stats[0].histogram is None:
            raise StatisticsNotFoundError('No statistics present to compute encodings.')

        encoding_min_list = []
        encoding_max_list = []

        for list_elem in stats:
            cum_sum = torch.cumsum(list_elem.histogram, dim=0)
            # trim percentile value from min and max
            max_index = torch.searchsorted(cum_sum, cum_sum[-1] * self.percentile/100)
            min_index = torch.searchsorted(cum_sum, cum_sum[-1] * (1 - self.percentile/100))

            if self.percentile == 100:
                min_index = 0
                max_index = -1
            curr_min = list_elem.bin_edges[min_index]
            curr_max = list_elem.bin_edges[max_index]
            # adjust min/max
            updated_min, updated_max = adjust_min_max(curr_min, curr_max, bitwidth, is_symmetric)
            encoding_min_list.append(updated_min)
            encoding_max_list.append(updated_max)

        encoding_min = torch.tensor(encoding_min_list, device=stats[0].histogram.device)
        encoding_min = torch.reshape(encoding_min, self.observer.shape)

        encoding_max = torch.tensor(encoding_max_list, device=stats[0].histogram.device)
        encoding_max = torch.reshape(encoding_max, self.observer.shape)

        return encoding_min, encoding_max


class SqnrEncodingAnalyzer(EncodingAnalyzer[_Histogram]):
    """
    Encoding Analyzer for SQNR Calibration technique
    """
    def __init__(self,
                 shape: tuple,
                 num_bins: int = 2048, *,
                 asymmetric_delta_candidates=17,
                 symmetric_delta_candidates=101,
                 offset_candidates=21,
                 max_parallelism=64,
                 gamma=3.0):
        """
        :param shape: Shape of calculated encoding
        :param num_bins: number of bins to use per histogram
        :param asymmetric_delta_candidates: number of delta values to search over in asymmetric mode
        :param symmetric_delta_candidates: number of delta values to search over in symmetric mode
        :param offset_candidates: number of offset values to search over in asymmetric mode
        :param max_parallelism: maximum number of encodings to process parallely (higher number results in higher
            memory usage but faster computation)
        :param gamma: weighting factor on clipping noise (higher value results in less clipping noise)
        """
        if num_bins <= 0:
            raise ValueError('Number of bins cannot be less than or equal to 0.')
        observer = _HistogramObserver(shape=shape, num_bins=num_bins)
        super().__init__(observer)
        self.asym_delta_candidates = asymmetric_delta_candidates
        self.sym_delta_candidates = symmetric_delta_candidates
        self.num_offset_candidates = offset_candidates
        self.gamma = gamma
        self.max_parallelism = max_parallelism

    @torch.no_grad()
    def update_stats(self, input_tensor: torch.Tensor) -> _Statistics:
        new_stats = self.observer.collect_stats(input_tensor)
        self.observer.merge_stats(new_stats, input_tensor)
        return new_stats

    @torch.no_grad()
    def compute_encodings_from_stats(self, stats: List[_Histogram], bitwidth: int, is_symmetric: bool)\
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Searches for encodings which produce the lowest expected SQNR based on the histograms in stats

        :param stats: A list of _Histogram objects with length equal to the number of encodings to compute
        :param bitwidth: The bitwidth of the computed encodings
        :param is_symmetric: If True, computes symmetric encodings, else computes asymmetric encodings
        :return: Tuple of computed encodings (min, max) as tensors with shape self.shape
        """
        if stats[0].histogram is None:
            raise StatisticsNotFoundError('No statistics present to compute encodings.')
        if bitwidth <= 0:
            raise ValueError('Bitwidth cannot be less than or equal to 0.')
        num_steps = 2 ** bitwidth - 1
        chunked_stats = [stats[i:min(i+self.max_parallelism, len(stats))] for i in range(0, len(stats), self.max_parallelism)]
        best_deltas, best_offsets = [], []
        for stats_ in chunked_stats:
            test_deltas, test_offsets = self._pick_test_candidates(stats_, num_steps, is_symmetric)
            best_delta, best_offset = self._select_best_candidates(test_deltas, test_offsets, stats_, num_steps)
            best_deltas.append(best_delta)
            best_offsets.append(best_offset)
        best_offset = best_offsets[0] if is_symmetric else torch.cat(best_offsets)
        best_delta = torch.cat(best_deltas)
        min_enc = best_offset * best_delta
        max_enc = min_enc + num_steps * best_delta
        return min_enc.view(self.observer.shape).to(stats[0].max.dtype), \
               max_enc.view(self.observer.shape).to(stats[0].max.dtype)

    def _pick_test_candidates(self, stats, num_steps, symmetric):
        # min/max.shape = (num_histograms, )
        min_vals = torch.stack([stat.min for stat in stats])
        max_vals = torch.stack([stat.max for stat in stats])
        min_vals = torch.min(min_vals, torch.zeros_like(min_vals))
        max_vals = torch.max(max_vals, torch.zeros_like(max_vals))
        max_vals = torch.max(max_vals, min_vals + torch.finfo(min_vals.dtype).tiny * num_steps)
        if symmetric:
            return self._pick_test_candidates_symmetric(min_vals, max_vals, num_steps)
        return self._pick_test_candidates_asymmetric(min_vals, max_vals, num_steps)

    def _pick_test_candidates_asymmetric(self, min_vals, max_vals, num_steps):
        """
        Selects the set of deltas and offsets over which to search for the optimal encodings
        """
        # Note: casting to float32 for two reason:
        #       1) float16 on CPU is not well-supported in pytorch
        #       2) Computing int16 encodings using f16 can result in inf (2 ** 16 - 1 == inf in fp16)
        tensor_kwargs = {"device": min_vals.device, "dtype": torch.float32}
        max_delta = (max_vals - min_vals).to(torch.float32) / num_steps
        observed_offset = torch.round(min_vals / max_delta)
        observed_min = max_delta * observed_offset
        observed_max = observed_min + max_delta * num_steps
        num_deltas = self.asym_delta_candidates
        search_space = torch.arange(start=1, end=(1 + num_deltas), step=1, **tensor_kwargs)
        # test_deltas.shape = (num_histograms, num_tests)
        test_deltas = max_delta[:, None] * search_space[None, :] / (num_deltas - 1)
        # test_offsets.shape = (num_offsets)
        num_offsets = min(num_steps + 2, self.num_offset_candidates)
        test_offset_step = num_steps / (num_offsets - 2) # subtract 2 because we add the observed offset
        test_offsets = torch.round(torch.arange(start=-num_steps, end=test_offset_step, step=test_offset_step, **tensor_kwargs))
        test_offsets = test_offsets[None, :].expand(min_vals.shape[0], -1)
        # Add in the observed offset as a candidate, test_offsets.shape = (num_histograms, num_offsets + 1)
        test_offsets = torch.concat((test_offsets, observed_offset[:, None]), dim=1)
        return self._clamp_delta_offset_values(observed_min, observed_max, num_steps, test_deltas, test_offsets)

    def _pick_test_candidates_symmetric(self, min_vals, max_vals, num_steps):
        """
        Selects the set of deltas over which to search for the optimal symmetric encodings
        """
        tensor_kwargs = {"device": min_vals.device, "dtype": torch.float32}
        max_delta = 2 * torch.max(max_vals, -min_vals).to(torch.float32) / num_steps
        test_offsets = torch.full((1, ), (-num_steps) // 2, **tensor_kwargs)
        num_deltas = self.sym_delta_candidates
        search_space = torch.arange(start=1, end=(1 + num_deltas), step=1, **tensor_kwargs)
        test_deltas = max_delta[:, None] * search_space[None, :] / (num_deltas - 1)
        # test_deltas.shape = (num_histograms, num_deltas, 1)
        # test_offsets.shape = (1, 1, 1)
        min_delta = torch.Tensor([torch.finfo(test_deltas.dtype).tiny]).to(**tensor_kwargs)
        test_deltas = torch.max(test_deltas, min_delta)
        return test_deltas[:, :, None], test_offsets[:, None, None]

    @staticmethod
    def _clamp_delta_offset_values(min_vals, max_vals, num_steps, test_deltas, test_offsets):
        """
        Clamps delta/offset encodings such that represented range falls within the observed min/max range of inputs
        """
        # test_min shape = (num_histograms, num_deltas, num_offsets)
        test_min = test_deltas[:, :, None] * test_offsets[:, None, :]
        test_max = test_min + test_deltas[:, :, None] * num_steps
        # Clamp min/max to observed min/max
        test_min = torch.max(min_vals[:, None, None], test_min)
        test_max = torch.min(max_vals[:, None, None], test_max)
        # Recompute delta/offset with clamped min/max
        # Returned delta/offset shapes = (num_histograms, num_deltas, num_offsets)
        test_deltas = (test_max - test_min) / num_steps
        min_delta = torch.Tensor([torch.finfo(test_deltas.dtype).tiny]).to(device=test_deltas.device,
                                                                           dtype=test_deltas.dtype)
        test_deltas = torch.max(test_deltas, min_delta)
        test_offsets = torch.round(test_min / test_deltas)
        return test_deltas, test_offsets

    def _select_best_candidates(self, test_deltas, test_offsets, stats, num_steps):
        """
        Searches all pairs of (delta, offset) in test_deltas, test_offsets to find the set with the lowest expected SQNR
        """
        noise = self._estimate_clip_and_quant_noise(stats, test_deltas, test_offsets, num_steps, self.gamma)
        _, min_idx = torch.min(noise.flatten(start_dim=1), dim=1)
        best_delta = torch.gather(test_deltas.flatten(start_dim=1), dim=1, index=min_idx[:, None])
        if test_offsets.numel() == 1:
            best_offset = test_offsets
        else:
            best_offset = torch.gather(test_offsets.flatten(start_dim=1), dim=1, index=min_idx[:, None])
        return best_delta, best_offset

    # pylint: disable=too-many-locals
    @staticmethod
    def _estimate_clip_and_quant_noise(stats: List[_Histogram],
                                       test_deltas: torch.Tensor,
                                       test_offsets: torch.Tensor,
                                       num_steps: int,
                                       gamma: float = 1.0):
        """
        Calculates the error from quantization for each delta, offset pair in test_deltas, test_offsets.
        We approximately reconstruct x from hists by assuming all elements within a given bin fall exactly on the
        midpoint of that bin.

        :param stats: list of _Histogram objects of observed input values
        :param test_deltas: Tensor holding the values of all deltas to search with shape (num_hists, num_deltas, num_offsets)
        :param test_offsets: Tensor holding values of all offsets to search with shape (num_hists, num_deltas, num_offsets)
        :param num_steps: Number of quantization steps, i.e., (2 ** bitwidth) - 1
        :param gamma: Fudge factor to trade off between saturation cost and quantization cost. When gamma=1.0, this
                      approximates the MSE of the quantization function
        """
        tensor_kwargs = {"device": test_deltas.device, "dtype": test_deltas.dtype}
        hists = torch.stack([stat.histogram for stat in stats])
        bin_edges = torch.stack([stat.bin_edges for stat in stats])
        hist_delta = bin_edges[:, 1] - bin_edges[:, 0]
        # hist_midpoints is shape (hists, num_bins)
        hist_offsets = hist_delta[:, None] * torch.arange(0, bin_edges.shape[1] - 1, **tensor_kwargs)[None, :]
        hist_midpoints = (bin_edges[:, 0] + hist_delta/2)[:, None] + hist_offsets
        # hists_midpoints_qdq is shape (hists, num_deltas, num_offsets, num_bins)
        test_offsets_bcast = test_offsets[:, :, :, None]
        test_deltas_bcast = test_deltas[:, :, :, None]
        hist_midpoints_qdq = hist_midpoints[:, None, None, :].div(test_deltas_bcast).sub(test_offsets_bcast).round()
        if gamma != 1.0:
            clipped = torch.logical_or(hist_midpoints_qdq < 0,
                                       hist_midpoints_qdq > num_steps)
        hist_midpoints_qdq = hist_midpoints_qdq.clamp_(0, num_steps).add_(test_offsets_bcast).mul_(test_deltas_bcast)
        square_error = hist_midpoints_qdq.sub_(hist_midpoints[:, None, None, :]).pow_(2).mul_(hists[:, None, None, :])
        if gamma != 1.0:
            # Apply the gamma "fudge factor" to the clipped errors
            square_error = torch.where(clipped, square_error * gamma, square_error)
        return torch.sum(square_error, dim=-1)
