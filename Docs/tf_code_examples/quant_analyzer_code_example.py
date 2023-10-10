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
# pylint: skip-file

""" Quant Analyzer code example """

# Step 0. Import statements
from typing import Any
import numpy as np
import tensorflow.compat.v1 as tf
from aimet_common.defs import QuantScheme
from aimet_tensorflow.quant_analyzer import QuantAnalyzer, CallbackFunc
# Below import is required just for eval_callback utility
# User can have their own implementation
from Examples.tensorflow.utils.image_net_evaluator import ImageNetEvaluator
# End step 0

# Step 1. Prepare forward pass callback
# NOTE: In the actual use cases, the users should implement this part to serve
#       their own goals if necessary.
def forward_pass_callback(session: tf.compat.v1.Session, _: Any = None) -> None:
    """
    NOTE: This is intended to be the user-defined model calibration function.
    AIMET requires the above signature. So if the user's calibration function does not
    match this signature, please create a simple wrapper around this callback function.

    A callback function for model calibration that simply runs forward passes on the model to
    compute encoding (delta/offset). This callback function should use representative data and should
    be subset of entire train/validation dataset (~1000 images/samples).

    :param model: Tensorflow session.
    :param _: Argument(s) of this callback function. Up to the user to determine the type of this parameter.
    E.g. could be simply an integer representing the number of data samples to use. Or could be a tuple of
    parameters or an object representing something more complex.
    """
    # User action required
    # User should create data loader/iterable using representative dataset and simply run
    # forward passes on the model.
# End step 1

# Step 2. Prepare eval callback
# NOTE: In the actual use cases, the users should implement this part to serve
#       their own goals if necessary.

def eval_callback(session: tf.compat.v1.Session, _: Any = None) -> float:
    """
    NOTE: This is intended to be the user-defined model evaluation function.
    AIMET requires the above signature. So if the user's calibration function does not
    match this signature, please create a simple wrapper around this callback function.

    A callback function for model evaluation that determines model performance. This callback function is
    expected to return scalar value representing the model performance evaluated against entire
    test/evaluation dataset.

    :param model: Tensorflow session.
    :param _: Argument(s) of this callback function. Up to the user to determine the type of this parameter.
    E.g. could be simply an integer representing the number of data samples to use. Or could be a tuple of
    parameters or an object representing something more complex.
    :return: Scalar value representing the model performance.
    """
    # User action required
    # User should create data loader/iterable using entire test/evaluation dataset, perform forward passes on
    # the model and return single scalar value representing the model performance.
    evaluator = ImageNetEvaluator('/path/to/tfrecords dataset', training_inputs=['keras_learning_phase:0'],
                                  data_inputs=['input_1:0'], validation_inputs=['labels:0'],
                                  image_size=224,
                                  batch_size=32,
                                  format_bgr=True)
    evaluator.evaluate(session, iterations=None)
# End step 2

def quant_analyzer_example():


    # Step 3. Create TF session
    # User action required
    # User should create a tf session from the model which has to be analyzed
    session = tf.compat.v1.Session()
    # User has to define start_op_names and output_op_names of model
    start_op_names = ['model_start_op_name']
    output_op_names = ['model_output_op_name']
    # User action required
    # User should pass actual argument(s) of the callback functions.
    forward_pass_callback_fn = CallbackFunc(forward_pass_callback, func_callback_args=None)
    eval_callback_fn = CallbackFunc(eval_callback, func_callback_args=None)
    # End step 3


    # Step 4. Create QuantAnalyzer object
    quant_analyzer = QuantAnalyzer(session, start_op_names=start_op_names, output_op_names=output_op_names,
                                   forward_pass_callback=forward_pass_callback_fn, eval_callback=eval_callback_fn, use_cuda= False)
    # End step 4

    # Step 5 (optional step)
    # Create unlabeled dataset and define num_batches to perform per op mse analysis
    # Create unlabeled dataset and define num_batches to perform per op mse analysis
    # Approximately 256 images/samples are recommended for MSE loss analysis. So, if the dataloader
    # has batch_size of 64, then 4 number of batches leads to 256 images/samples.
    # User action required
    # User should use unlabeled dataloader, so if the dataloader yields labels as well user should use discard them.
    dataset_size = 128
    input_data = np.random.rand(dataset_size, 224, 224, 3)
    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    batch_size = 32
    unlabeled_dataset = dataset.batch(batch_size=batch_size)
    num_batches = 4
    # End step 5

    # Step 6. Run QuantAnalyzer
    quant_analyzer.analyze(quant_scheme=QuantScheme.post_training_tf_enhanced,
                           default_param_bw=8,
                           default_output_bw=8,
                           config_file=None,
                           unlabeled_dataset=unlabeled_dataset,
                           num_batches=num_batches,
                           results_dir="./quant_analyzer_results/")
    # End step 6


if __name__ == '__main__':
    quant_analyzer_example()
