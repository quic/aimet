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

"""Cache Implementation"""
import abc
import contextlib
import functools
import os
import pickle
from typing import Any, Callable, Optional, Generic, TypeVar

from aimet_common.utils import AimetLogger


_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


class CacheMiss(FileNotFoundError):
    """Exception to be raised upon cache miss."""


T = TypeVar("T")

class SerializationProtocolBase(abc.ABC, Generic[T]): # pylint: disable
    """Serialization protocol for objects of type T.

    Invariants:
        - if type(obj) == T: # NOTE: The types should be EXACTLY the same.
            self.save(obj, working_dir, filename_prefix);
            assert obj == self.load(working_dir, filename_prefix)

        - otherwise:
            self.save(obj, ...) should raise TypeError
    """

    @abc.abstractmethod
    def save(self, obj: T, working_dir: str, filename_prefix: str) -> None:
        """
        Save an object of type T.
        :param obj: Object to save.
        :param working_dir: Directory to save the file.
        :param filename_prefix: File name prefix.
        :raises: TypeError if obj is not of type T.
        """

    @abc.abstractmethod
    def load(self, working_dir: str, filename_prefix: str) -> T:
        """
        Load the saved object.
        :param working_dir: Directory to save the file.
        :param filename_prefix: File name prefix.
        :return: Loaded object.
        :raises: Cache miss if the combination of working_dir and
            filename_prefix fails to find a previously saved cache entry.
        """

    @classmethod
    def _type_error(cls, obj, expected_type):
        """Helper funtion for creating a commonly used type error."""
        obj_type = type(obj)
        msg = f"{cls.__name__} cannot serialize an object of type {obj_type} "\
              f"(expected type: {expected_type})."
        return TypeError(msg)


class _PickleSerializationProtocol(SerializationProtocolBase):
    """Serialization protocol for pickle-serializable objects"""

    @classmethod
    def _get_filename(cls, working_dir, filename_prefix):
        """Get the name of the file to save the pickle-serialized results to."""
        return os.path.join(working_dir, f"{filename_prefix}.pkl")

    def save(self, obj: Any, working_dir: str, filename_prefix: str) -> None:
        """
        Save a pickle-serializable object.
        :param obj: Object to save.
        :param working_dir: Directory to save the file.
        :param filename_prefix: File name prefix.
        :raises: TypeError if obj is not pickle-serializable.
        """
        filename = self._get_filename(working_dir, filename_prefix)
        with open(filename, "wb") as f:
            try:
                pickle.dump(obj, f)
            except pickle.PicklingError as e:
                raise TypeError from e

    def load(self, working_dir: str, filename_prefix: str) -> Any:
        """
        Load the saved object.
        :param working_dir: Directory to save the file.
        :param filename_prefix: File name prefix.
        :return: Loaded object.
        :raises: Cache miss if the combination of working_dir and
            filename_prefix fails to find a previously saved cache entry.
        """
        filename = self._get_filename(working_dir, filename_prefix)
        if os.path.exists(filename):
            # Cached file exists (cache hit). Load from cache.
            with open(filename, "rb") as f:
                return pickle.load(f)
        raise CacheMiss


class Cache:
    """
    Cache that performs return value caching.

    Being a return value cache, one should take extra care before applying it.
    Only to the funcitons that satisfy all of the following conditions can we
    safely apply return value caching:

      - The function should be STATELESS.
        The return value of the function should depend only on the inputs and
        not on any other external states, including the filesystem states.

      - The function should be IDEMPOTENT.
        Calling the function with the identical input should always return the same outputs.
        For example, `fold_all_batchnorms` doesn't satisfy this condition because
        calling `fold_all_batchnorms` with the identical model iteratively will
        return a list of folded pairs only at the first call, and never again.

    Additional pitfall:
      - The original object and the one loaded from the cache are EQUAL but NOT IDENTICAL.
        This is because the caching mechanism is fundamentally based on serialization.
    """

    def __init__(self):
        self._cache_dir = None

    def mark(self, cache_key: str, protocol: SerializationProtocolBase = None):
        """
        Mark functions that are subject to caching.
        The functions decorated with this mark will save/load the outputs
        to/from the cache directory if caching is enabled.

        :param cache_key: Used as a prefix of the name of the file that
            caches the results of the decorated function.
        :param protocol: Serialization protocol for the return values of the function.
            By default, we use pickle serialization protocol.
        :return: A decorator that registers the decorated functions.
        """
        # Use pickle serialization by default.
        protocol = protocol or _PickleSerializationProtocol()

        def _wrap(fn: Callable, cache_key: str):
            @functools.wraps(fn)
            def caching_helper(*args, **kwargs):
                # If caching is disabled, evalaute the result.
                if self._cache_dir is None:
                    return fn(*args, **kwargs)

                working_dir = self._cache_dir
                filename_prefix = cache_key
                try:
                    # Try loading the previously evaluated result from cache.
                    _logger.debug("Loading result of %s from %s.", cache_key, self._cache_dir)
                    return protocol.load(working_dir, filename_prefix)
                except CacheMiss:
                    _logger.debug("Cache miss.")
                    ret = fn(*args, **kwargs)
                    _logger.debug("Caching result of %s to %s.", cache_key, self._cache_dir)
                    protocol.save(ret, working_dir, filename_prefix)
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
