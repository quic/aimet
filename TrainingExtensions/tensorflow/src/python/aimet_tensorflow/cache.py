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

"""Defines TensorFlow-specific serialization protocols for cache"""
import os
import tensorflow as tf

from aimet_common.cache import SerializationProtocolBase, CacheMiss
from aimet_tensorflow.utils.graph_saver import save_model_to_meta, load_model_from_meta

tf.compat.v1.disable_eager_execution()


class TfSessionSerializationProtocol(SerializationProtocolBase[tf.compat.v1.Session]):
    """Serialization protocol for tf.Session objects"""

    def save(self, obj: tf.compat.v1.Session, working_dir: str, filename_prefix: str) -> None:
        """
        Save a tf.Session object.
        :param obj: Object to save.
        :param working_dir: Directory to save the file.
        :param filename_prefix: File name prefix.
        :raises: TypeError if obj is not a tf.Session.
        """
        if type(obj) != tf.compat.v1.Session: # pylint: disable=unidiomatic-typecheck
            raise self._type_error(obj, tf.compat.v1.Session)

        meta_path = os.path.join(working_dir, filename_prefix)
        save_model_to_meta(obj, meta_path)

    def load(self, working_dir: str, filename_prefix: str) -> tf.compat.v1.Session: # pylint: disable=no-self-use
        """
        Load the saved object.
        :param working_dir: Directory to save the file.
        :param filename_prefix: File name prefix.
        :return: Loaded object.
        :raises: Cache miss if the combination of working_dir and
            filename_prefix fails to find a previously saved cache entry.
        """
        meta_path = os.path.join(working_dir, f"{filename_prefix}.meta")
        if os.path.exists(meta_path):
            return load_model_from_meta(meta_path)
        raise CacheMiss
