# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Quant Analyzer for AIMET v2"""

import os
import itertools
import contextlib
from collections import defaultdict, namedtuple
from typing import Tuple, Dict, List, Type, Optional
import torch
import torch.nn as nn

from aimet_common.quant_analyzer import export_stats_histogram_plot
from aimet_common.utils import AimetLogger
from aimet_torch.quant_analyzer import QuantAnalyzer as V1QuantAnalyzer
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn.base import BaseQuantizationMixin
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.encoding_analyzer import _HistogramObserver, _Histogram


_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.QuantAnalyzer)


V1Encoding = namedtuple('V1Encoding', ['min', 'max'])
RestorableQuantizer = namedtuple('RestorableQuantizer', ['container', 'key', 'quantizer'])


class QuantAnalyzer(V1QuantAnalyzer):
    """
    QuantAnalyzer tool provides

     1) model sensitivity to weight and activation quantization
     2) per layer sensitivity analysis
     3) per layer encoding (min - max range)
     4) per PDF analysis and
     5) per layer MSE analysis
    """
    @staticmethod
    def _get_quantsim_cls() -> Type[QuantizationSimModel]:
        return QuantizationSimModel

    @staticmethod
    def _get_quant_wrapper_type() -> Tuple[Type]:
        return (BaseQuantizationMixin,)

    # pylint: disable=no-self-use
    def _create_and_export_stats_histogram_plot(self,
                                                quantizer: QuantizerBase,
                                                results_dir: str,
                                                title: str,
                                                ):
        """
        For given quantizer, create and export histogram (PDF) of statistics in html format.

        :param quantizer: Quantizer.
        :param results_dir: Directory to save the results.
        :param title: Title of the plot.
        """
        os.makedirs(results_dir, exist_ok=True)

        assert isinstance(quantizer.encoding_analyzer.observer, _HistogramObserver)
        v2_histograms = quantizer.encoding_analyzer.observer.get_stats()
        histograms = self._convert_to_v1_histograms(v2_histograms)
        encodings = self._get_quantizer_encodings(quantizer)

        for index, (histogram, encoding) in enumerate(zip(histograms, encodings)):
            export_stats_histogram_plot(histogram, encoding, results_dir, title=f"{title}_{index}")

    @staticmethod
    def _enable_disable_quant_wrapper(quant_wrapper: BaseQuantizationMixin,
                                      enabled_quant_wrappers: Dict[nn.Module, List[RestorableQuantizer]],
                                      enabled: bool):
        enabled_quantizers = enabled_quant_wrappers[quant_wrapper]

        for quantizer in enabled_quantizers:
            # Enable or disable quantizer
            quantizer.container[quantizer.key] = \
                (quantizer.quantizer if enabled else None)

    @staticmethod
    def _disable_param_quantizers(sim: QuantizationSimModel):
        # pylint: disable=protected-access
        ctx = contextlib.ExitStack()
        for _, quant_wrapper in sim.quant_wrappers():
            ctx.enter_context(quant_wrapper._remove_param_quantizers())
        return ctx

    @staticmethod
    def _disable_activation_quantizers(sim: QuantizationSimModel):
        # pylint: disable=protected-access
        ctx = contextlib.ExitStack()
        for _, quant_wrapper in sim.quant_wrappers():
            ctx.enter_context(quant_wrapper._remove_activation_quantizers())
        return ctx

    @staticmethod
    def _disable_quantizers(sim: QuantizationSimModel):
        # pylint: disable=protected-access
        ctx = contextlib.ExitStack()
        for _, quant_wrapper in sim.quant_wrappers():
            ctx.enter_context(quant_wrapper._remove_all_quantizers())
        return ctx

    @staticmethod
    def patch_quantsim_to_store_histogram(sim: QuantizationSimModel):
        """
        Utility function for patching quantizers in quantsim to keep histogram information
        """
        for _, quant_wrapper in sim.quant_wrappers():
            for quantizer in itertools.chain(quant_wrapper.input_quantizers,
                                             quant_wrapper.output_quantizers,
                                             quant_wrapper.param_quantizers.values()):
                if quantizer is None:
                    continue

                quantizer.encoding_analyzer.reset_stats = lambda: None

    @staticmethod
    def _convert_to_v1_histograms(histograms: List[_Histogram]) -> List:
        v1_histograms = []
        for hist in histograms:
            assert hist is not None, "Cannot find histogram data in quantsim\n" \
                "Please patch quantsim object before calling compute_encodings " \
                "using patch_quantsim_to_store_histogram method to store histogram data"
            hist_sum = torch.sum(hist.histogram).item()
            v1_hist = []
            for bin_edge, hist_value in zip(hist.bin_edges, hist.histogram):
                v1_hist.append((bin_edge.item(), hist_value.item() / hist_sum))
            v1_histograms.append(v1_hist)

        return v1_histograms

    @staticmethod
    def _is_quantizer_enabled(quantizer: Optional[QuantizerBase]):
        return quantizer is not None

    @classmethod
    def _get_quantizer_encodings(cls, quantizer: QuantizerBase) -> Optional[List]:
        v1_encodings = []

        encoding = quantizer.get_encoding()
        if not encoding:
            return None

        flatten_min = encoding.min.flatten()
        flatten_max = encoding.max.flatten()

        for encoding_min, encoding_max in zip(flatten_min, flatten_max):
            v1_encodings.append(V1Encoding(min=encoding_min.item(), max=encoding_max.item()))

        return v1_encodings

    @classmethod
    def _get_enabled_quantizers(cls, sorted_quant_wrappers: Dict[str, BaseQuantizationMixin])\
            -> Dict[BaseQuantizationMixin, List[QuantizerBase]]:
        """
        For given sorted quant wrappers dict, get enabled quantizers.

        :param sorted_quant_wrappers: Dictionary containing quant wrappers sorted based on occurrence.
        :return: Dictionary which maps a quant wrapper to a list of enabled quantizers in it.
        """
        enabled_quant_wrappers = defaultdict(list)

        for quant_wrapper in sorted_quant_wrappers.values():
            for key, quantizer in quant_wrapper.param_quantizers.items():
                if cls._is_quantizer_enabled(quantizer):
                    restorable_quantizer = RestorableQuantizer(quant_wrapper.param_quantizers, key, quantizer)
                    enabled_quant_wrappers[quant_wrapper].append(restorable_quantizer)
            for idx, quantizer in enumerate(quant_wrapper.output_quantizers):
                if cls._is_quantizer_enabled(quantizer):
                    restorable_quantizer = RestorableQuantizer(quant_wrapper.output_quantizers, idx, quantizer)
                    enabled_quant_wrappers[quant_wrapper].append(restorable_quantizer)
            for idx, quantizer in enumerate(quant_wrapper.input_quantizers):
                if cls._is_quantizer_enabled(quantizer):
                    restorable_quantizer = RestorableQuantizer(quant_wrapper.input_quantizers, idx, quantizer)
                    enabled_quant_wrappers[quant_wrapper].append(restorable_quantizer)

        return enabled_quant_wrappers
