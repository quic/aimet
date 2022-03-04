# /usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

"""Automatic Post-Training Quantization"""
import abc
import contextlib
import functools
import os
import pickle
from typing import Any, Callable, Optional, List, Union

import bokeh.model
import bokeh.embed

from aimet_common.utils import AimetLogger


_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AutoQuant)


class Cache:
    """Cache of the outputs of PTQ functions."""

    def __init__(self):
        self._cache_dir = None

    def mark(self, cache_key: str):
        """
        Mark functions that are subject to caching.
        The functions decorated with this mark will save/load the outputs
        to/from the cache directory if caching is enabled.

        :param cache_key: Used as a prefix of the name of the file that
            caches the results of the decorated function.
        :return: A decorator that registers the decorated functions.
        """
        def _wrap(fn: Callable, cache_key: str):
            @functools.wraps(fn)
            def caching_helper(*args, **kwargs):
                # If caching is disabled, Evalaute the result.
                if self._cache_dir is None:
                    return fn(*args, **kwargs)

                cache_file = os.path.join(self._cache_dir, f"{cache_key}.pkl")
                if os.path.exists(cache_file):
                    _logger.info("Loading result of %s from %s", cache_key, cache_file)
                    # Cached file exists (cache hit). Load from cache.
                    with open(cache_file, "rb") as f:
                        ret = pickle.load(f)
                else:
                    # No cached file (cache miss). Evaluate the result.
                    ret = fn(*args, **kwargs)
                    _logger.info("Caching result of %s to %s", cache_key, cache_file)

                    # Save results to the cache.
                    with open(cache_file, "wb") as f:
                        pickle.dump(ret, f)
                return ret
            return caching_helper

        return lambda fn: _wrap(fn, cache_key)

    @contextlib.contextmanager
    def enable(self, cache_dir: Optional[str]):
        """
        Enable caching.

        :param cache_dir: Directory to read/save the cached results from/to.
        """
        self._cache_dir = cache_dir
        try:
            if self._cache_dir is not None:
                os.makedirs(self._cache_dir, exist_ok=True)
                _logger.info("AutoQuant caching is enabled. Cache directory: %s", self._cache_dir)
            yield
        finally:
            self._cache_dir = None


class Diagnostics:
    """Diagnostics produced by _EvalSession"""

    def __init__(self):
        self._contents: List[_DiagnosticsContent] = []

    def add(self, content: Union[str, bokeh.model.Model]) -> None:
        """
        Add content of diagnostics.
        :param content: Content of diagnostics.
        :return: None
        """
        if isinstance(content, str):
            c = _PlainTextContent(content)
        elif isinstance(content, bokeh.model.Model):
            c = _BokehModelContent(content)
        else:
            raise RuntimeError
        self._contents.append(c)

    def is_empty(self) -> bool:
        """Return True if and only if the diagnostics is empty."""
        return not bool(self._contents)

    def contains_bokeh(self):
        """Return True if and only if the diagnostics contains a bokeh model object."""
        return any(
            isinstance(content, _BokehModelContent)
            for content in self._contents
        )

    def __bool__(self):
        return not self.is_empty()

    def __iter__(self):
        return iter(self._contents)


class _DiagnosticsContent(abc.ABC):
    """Content of diagnostics."""
    def __init__(self, content: Any):
        self._content = content

    @abc.abstractmethod
    def get_html_elem(self) -> str:
        """
        Render content as an html element.
        :return: Content rendered as an html element.
        """


class _PlainTextContent(_DiagnosticsContent):
    """Content of diagnostics in plain text."""
    def get_html_elem(self) -> str:
        return f"<div> {self._content} </div>"


class _BokehModelContent(_DiagnosticsContent):
    """Content of diagnostics as a bokeh model object."""
    def get_html_elem(self) -> str:
        bokeh_model = self._content
        script, div = bokeh.embed.components(bokeh_model)
        return f"{script}{div}"
