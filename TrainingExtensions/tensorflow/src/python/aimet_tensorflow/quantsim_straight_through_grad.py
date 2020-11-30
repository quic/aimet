# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" implements straight through graident computation for Quantize Op"""

import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from aimet_tensorflow.utils.constants import QuantizeOpIndices


# by default we will have this registered for Qc Quantize op.
@tf_ops.RegisterGradient("QcQuantize")
def _qc_straight_through_estimator_grad(op, grad):
    # pylint: disable=unused-argument
    """
    straight through estimator logic used to compute gradient for Quantize Op.
    :param op: quantize op
    :param grad: gradient
    :return: gradients computed per input
    """
    x = tf.cast(op.inputs[0], tf.float32)
    encoding_min = tf.cast(op.inputs[int(QuantizeOpIndices.encoding_min)], tf.float32)
    encoding_max = tf.cast(op.inputs[int(QuantizeOpIndices.encoding_max)], tf.float32)
    op_mode = tf.cast(op.inputs[int(QuantizeOpIndices.op_mode)], tf.int8)

    inner_cond = tf.compat.v2.where(tf.less_equal(x, encoding_max),  # condition to check per value
                             1.0,  # execute if true
                             0.0)  # execute if false

    dloss_by_dx = (tf.compat.v2.where(tf.less_equal(encoding_min, x),  # condition to check per value
                               inner_cond,  # execute if true
                               0.0)) * grad

    # Pass through gradient for skipped ops
    dloss_by_dx = tf.cond(tf.equal(op_mode, 3), lambda: grad, lambda: dloss_by_dx)

    return dloss_by_dx, None, None, None, None, None, None
