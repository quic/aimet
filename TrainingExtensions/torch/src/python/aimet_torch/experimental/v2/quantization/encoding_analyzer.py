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
# pylint: disable=all
from typing import TypeVar, Generic, Tuple, Type, Optional
import abc
from dataclasses import dataclass

import torch

from aimet_torch.experimental.v2.utils import reduce


@dataclass(frozen=True)
class _MinMaxRange:
    min: Optional[torch.Tensor] = None
    max: Optional[torch.Tensor] = None


class _Histogram:
    # TODO
    ...


_Statistics = TypeVar('_Statistics', _MinMaxRange, _Histogram)


class _Observer(Generic[_Statistics], abc.ABC):
    def __init__(self, shape):
        self.shape = shape

    @abc.abstractmethod
    def collect_stats(self, x: torch.Tensor) -> _Statistics:
        ...

    @abc.abstractmethod
    def merge_stats(self, stats: _Statistics):
        ...

    @abc.abstractmethod
    def reset_stats(self):
        ...

    @abc.abstractmethod
    def get_stats(self) -> _Statistics:
        ...


class _MinMaxObserver(_Observer[_MinMaxRange]):
    def __init__(self, shape):
        super().__init__(shape)
        self.stats = _MinMaxRange()

    @torch.no_grad()
    def collect_stats(self, x: torch.Tensor) -> _MinMaxRange:
        min = reduce(x, shape=self.shape, reduce_op=torch.min).values
        max = reduce(x, shape=self.shape, reduce_op=torch.max).values
        return _MinMaxRange(min, max)

    @torch.no_grad()
    def merge_stats(self, new_stats: _MinMaxRange):
        min = self.stats.min
        if new_stats.min is not None:
            if min is None:
                min = new_stats.min.clone()
            else:
                min = torch.minimum(min, new_stats.min)

        max = self.stats.max
        if new_stats.max is not None:
            if max is None:
                max = new_stats.max.clone()
            else:
                max = torch.maximum(max, new_stats.max)

        self.stats = _MinMaxRange(min, max)

    def reset_stats(self):
        self.stats = _MinMaxRange()

    def get_stats(self) -> _MinMaxRange:
        return self.stats


class _HistogramObserver(_Observer[_Histogram]):
    def __init__(self, shape):
        # TODO
        raise NotImplementedError

    @torch.no_grad()
    def collect_stats(self, x: torch.Tensor) -> _Histogram:
        # TODO
        raise NotImplementedError

    @torch.no_grad()
    def merge_stats(self, new_stats: _Histogram):
        # TODO
        raise NotImplementedError

    def reset_stats(self):
        # TODO
        raise NotImplementedError

    def get_stats(self) -> _Histogram:
        # TODO
        raise NotImplementedError


class _EncodingAnalyzer(Generic[_Statistics], abc.ABC):
    observer_cls: Type[_Observer[_Statistics]]

    def __init__(self, shape):
        self.observer = self.observer_cls(shape)

    @torch.no_grad()
    def update_stats(self, x: torch.Tensor) -> _Statistics:
        new_stats = self.observer.collect_stats(x)
        self.observer.merge_stats(new_stats)
        return new_stats

    def reset_stats(self) -> None:
        self.observer.reset_stats()

    def compute_encodings(self, symmetric: bool, bitwidth: int)\
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.compute_encodings_from_stats(self.observer.get_stats(), symmetric, bitwidth)

    def compute_dynamic_encodings(self, x: torch.Tensor, symmetric: bool, bitwidth: int)\
            -> Tuple[torch.Tensor, torch.Tensor]:
        return self.compute_encodings_from_stats(self.observer.collect_stats(x), symmetric, bitwidth)

    @abc.abstractmethod
    def compute_encodings_from_stats(self, stats: _Statistics, symmetric: bool, bitwidth: int)\
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        ...


class MinMaxEncodingAnalyzer(_EncodingAnalyzer[_MinMaxRange]):
    observer_cls = _MinMaxObserver

    @torch.no_grad()
    def compute_encodings_from_stats(self, stats: _MinMaxRange, symmetric: bool, bitwidth: int)\
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if stats.min is None or stats.max is None:
            return None, None

        if symmetric:
            min = torch.minimum(stats.min, -stats.max)
            max = torch.maximum(-stats.min, stats.max)
        else:
            min = stats.min
            max = stats.max

        return min, max


class PercentileEncodingAnalyzer(_EncodingAnalyzer[_Histogram]):
    observer_cls = _HistogramObserver

    @torch.no_grad()
    def compute_encodings_from_stats(self, stats: _Histogram, symmetric: bool, bitwidth: int)\
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # TODO
        raise NotImplementedError


class SqnrEncodingAnalyzer(_EncodingAnalyzer[_Histogram]):
    observer_cls = _HistogramObserver

    @torch.no_grad()
    def compute_encodings_from_stats(self, stats: _Histogram, symmetric: bool, bitwidth: int)\
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # TODO
        raise NotImplementedError


class MseEncodingAnalyzer(_EncodingAnalyzer[_Histogram]):
    observer_cls = _HistogramObserver

    @torch.no_grad()
    def compute_encodings_from_stats(self, stats: _Histogram, symmetric: bool, bitwidth: int)\
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # TODO
        raise NotImplementedError


def get_encoding_analyzer_cls(qscheme):
    if qscheme == 'minmax':
        return MinMaxEncodingAnalyzer

    raise ValueError
