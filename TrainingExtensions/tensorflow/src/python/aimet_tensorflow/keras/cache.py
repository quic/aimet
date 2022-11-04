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
"""Defines Keras-specific serialization protocols for cache"""
import os

import tensorflow as tf

from aimet_common.cache import SerializationProtocolBase, T, CacheMiss


class KerasModelSerializationProtocol(SerializationProtocolBase[tf.keras.Model]):
    """Serialization protocol for tf.keras.Model objects"""
    def save(self, obj: T, working_dir: str, filename_prefix: str) -> None:
        """
        Save a tf.keras.Model object.

        :param obj: Object to save.
        :param working_dir: Directory to save the file.
        :param filename_prefix: File name prefix.
        :return: TypeError if obj is not a tf.keras.Model
        """
        if not isinstance(obj, tf.keras.Model):
            raise self._type_error(obj, tf.keras.Model)

        file_path = os.path.join(working_dir, filename_prefix)
        obj.save(file_path)

    def load(self, working_dir: str, filename_prefix: str) -> T:
        """
        Load the saved object.

        :param working_dir: Directory to load the file.
        :param filename_prefix: File name prefix.
        :return: Loaded object.
        :raises: Cache miss if the combination of working_dir and
            filename_prefix fails to find a previously saved cache entry.
        """
        file_path = os.path.join(working_dir, filename_prefix)
        if os.path.exists(file_path):
            return tf.keras.models.load_model(file_path)
        raise CacheMiss
