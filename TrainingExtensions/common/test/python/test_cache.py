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

import os
import shutil
import random
from typing import Callable

import numpy as np

from aimet_common.cache import Cache, SerializationProtocolBase


SEED = 18452
random.seed(SEED)
np.random.seed(SEED)


def _assert_equal_default(output, expected):
    assert type(output) == type(expected)
    assert output == expected


def _test_cache(fn,
                protocol: SerializationProtocolBase = None,
                assert_equal_fn: Callable = None):
    if not assert_equal_fn:
        assert_equal_fn = _assert_equal_default

    cache_dir = "/tmp/test_dir"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    try:
        cache = Cache()

        call_count = 0

        @cache.mark("test", protocol)
        def _fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return fn(*args, **kwargs)

        with cache.enable(cache_dir):
            ret = _fn()

        with cache.enable(cache_dir):
            _ret = _fn()

        assert_equal_fn(ret, _ret)
        assert call_count == 1
    finally:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)


def test_cache_number():
    _test_cache(lambda: random.random())


def test_cache_list():
    _test_cache(lambda: [random.random() for _ in range(10)])


def test_cache_tuple():
    _test_cache(lambda: tuple(random.random() for _ in range(10)))


def test_cache_none():
    _test_cache(lambda: None)


def test_cache_numpy_array():
    def assert_equal(x, y):
        assert np.array_equal(x, y)
    _test_cache(lambda: np.random.randn(10, 10), assert_equal_fn=assert_equal)
