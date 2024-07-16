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

# pylint: skip-file

""" Keras Mixed precision code example to be used for documentation generation. """
# Start of import statements
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import random
import numpy as np

# imports specific to resnet50 pretrained model
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input, decode_predictions

# AIMET imports
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_common.defs import CallbackFunc, QuantizationDataType, QuantScheme

# for batch_norm_fold
from aimet_tensorflow.keras.batch_norm_fold import fold_all_batch_norms

# for mixed precision
from aimet_tensorflow.keras.mixed_precision import choose_mixed_precision, choose_fast_mixed_precision
from aimet_tensorflow.keras.amp.mixed_precision_algo import GreedyMixedPrecisionAlgo

# End of import statements

def center_crop(image):
    """
    Perform the center corp on the images.
    :param image: List of images as tensors which we need to center corp. Expects the image size of 256 x 256.
    :return: Center corped images of size 224 x 224
    """

    img_height = 256
    img_width = 256
    crop_length = 224
    start_x = (img_height - crop_length) // 2
    start_y = (img_width - crop_length) // 2
    cropped_image = image[:,  start_x:(img_width - start_x), start_y:(img_height - start_y), :]
    return cropped_image

def get_eval_func(dataset_dir, batch_size, num_iterations=50000):
    """
    Helper Function returns an evaluation function which performs the forward pass on the specified model
     with given dataset parameters
    :param dataset_dir: Directrory from where the dataset images needs to be loaded.
    :param batch_size: Batch size to be used in dataloader
    :param num_iterations: Optional parameter stating total number of images to be used.
    Default set to 50000, which is size of the validation set of imagenet dataset.
    :return: returns a evaluation function which can be used to evaluate the model's accuracy on the preset dataset.
    """

    def func_wrapper(model, iterations):
        """ Evaluation Function which is return from the parent function. Performs the forward pass on the model with the given dataset and retuerns the acuracy."""

        validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory=dataset_dir,
            labels='inferred',
            label_mode='categorical',
            batch_size=batch_size,
            shuffle=False)

        # If no iterations specified, set to full validation set
        if not iterations:
            iterations = num_iterations
        else:
            iterations = iterations * batch_size
        top1 = 0
        total = 0
        for (img, label) in validation_ds:
            img = center_crop(img)
            x = preprocess_input(img)
            preds = model.predict(x,batch_size = batch_size)
            label = np.where(label)[1]
            label = [validation_ds.class_names[int(i)] for i in label]
            cnt = sum([1 for a, b in zip(label, decode_predictions(preds, top=1)) if str(a) == b[0][0]])
            top1 += cnt
            total += len(label)
            if total >= iterations:
                break
        return top1/total
    return func_wrapper

def get_data_loader_wrapper(dataset_dir, batch_size, is_training=False):
    """
    Helper function which returns a method calling which will give a data loader.
    :param dataset_dir: Directrory from where the dataset images needs to be loaded.
    :param batch_size: Batch size to be used in dataloader
    :param is_training: Default to False. It is used to set the shuffle flag for the data loader.
    :return: Returns a wrapper function which will return a dataloader.
    """
    def dataloader_wrapper():
        dataloader = tf.keras.preprocessing.image_dataset_from_directory(
            directory=dataset_dir,
            labels='inferred',
            label_mode='categorical',
            batch_size=batch_size,
            shuffle = is_training,
            image_size=(256, 256))

        return dataloader.map(lambda x, y: preprocess_input(center_crop(x)))

    return dataloader_wrapper

def get_model():
    """Helper function to return the model"""
    model = ResNet50(
        input_shape=None,
        alpha=1.0,
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=1000)
    return model

def get_quantizated_model(original_model, eval_func):

    quant_scheme = QuantScheme.post_training_tf_enhanced

    sim = QuantizationSimModel(
        model=original_model,
        quant_scheme=quant_scheme,
        rounding_mode="nearest",
        default_output_bw=8,
        default_param_bw=8,
    )

    sim.compute_encodings(eval_func,
                          forward_pass_callback_args=500
                          )
    return sim


def mixed_precision(dataset_dir):
    """
    Sample function which demonstrates the quantization on a Resnet50 model followed by mixed precision
    """
    np.random.seed(1)
    random.seed(1)
    tf.random.set_seed(1)

    batch_size = 32

    # Load the model
    model = get_model()

    # Perform batch norm folding
    _, model = fold_all_batch_norms(model)

    # get the evalutaion function
    # We will use this function to for forward pass callback as well.
    eval_func = get_eval_func(dataset_dir, batch_size)

    # Calculate the Original Model accuracy
    org_top1 = eval_func(model, None)
    print("Original Model Accuracy: ", org_top1)

    # get the quantized model object
    sim = get_quantizated_model(model, eval_func)

    # Set the candidates for the mixed precision algorithm
    # Candidate format given below
    # ((activation bitwidth, activation data type), (param bitwidth, param data type))
    # e.g. ((16, QuantizationDataType.int), (16, QuantizationDataType.int)),
    candidate = [((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
                 ((8, QuantizationDataType.int), (8, QuantizationDataType.int))]

    # The allowed accuracy drop represents the amount of accuracy drop we are accepting
    # to trade for a lower precision, faster model.
    # 0.09 represents we are accepting upto 9% accuracy drop from the baseline.
    allowed_accuracy_drop = 0.09

    eval_callback = CallbackFunc(eval_func, None)
    forward_pass_call_back = CallbackFunc(eval_func, 500)

    # Enable phase-3 (optional)
    # GreedyMixedPrecisionAlgo.ENABLE_CONVERT_OP_REDUCTION = True
    # Note: supported candidates ((8,int), (8,int)) & ((16,int), (8,int))

    # Call the mixed precision wrapper with appropriate parameters
    choose_mixed_precision(sim,  candidate, eval_callback, eval_callback, allowed_accuracy_drop, "./cmp_res", True, forward_pass_call_back )
    print("Mixed Precision Model Accuracy: ", eval_func(sim.model, None))
    sim.export(filename_prefix='mixed_preision_quant_model', path='.')

def fast_mixed_precision(dataset_dir):
    """
    Sample function which demonstrates the quantization on a Resnet50 model followed by mixed precision using AMP 2.0
    """
    np.random.seed(1)
    random.seed(1)
    tf.random.set_seed(1)

    batch_size = 32

    # Load the model
    model = get_model()

    # Perform batch norm folding
    _ = fold_all_batch_norms(model)

    # get the evalutaion function
    # We will use this function to for forward pass callback as well.
    eval_func = get_eval_func(dataset_dir, batch_size)

    # Calculate the Original Model accuracy
    org_top1 = eval_func(model, None)
    print("Original Model Accuracy: ", org_top1)

    # get the quantized model object
    sim = get_quantizated_model(model, eval_func)

    # Set the candidates for the mixed precision algorithm
    # Candidate format given below
    # ((activation bitwidth, activation data type), (param bitwidth, param data type))
    # e.g. ((16, QuantizationDataType.int), (16, QuantizationDataType.int)),
    candidate = [((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
                 ((8, QuantizationDataType.int), (8, QuantizationDataType.int))]

    # The allowed accuracy drop represents the amount of accuracy drop we are accepting
    # to trade for a lower precision, faster model.
    # 0.09 represents we are accepting upto 9% accuracy drop from the baseline.
    allowed_accuracy_drop = 0.09

    data_loader_wrapper = get_data_loader_wrapper(dataset_dir, batch_size)

    eval_callback = CallbackFunc(eval_func, None)
    forward_pass_call_back = CallbackFunc(eval_func, 500)

    # Enable phase-3 (optional)
    # GreedyMixedPrecisionAlgo.ENABLE_CONVERT_OP_REDUCTION = True
    # Note: supported candidates ((8,int), (8,int)) & ((16,int), (8,int))

    # Get the GreedyMixedPrecisionAlgo Object
    choose_fast_mixed_precision(sim,  candidate, data_loader_wrapper, eval_callback, allowed_accuracy_drop, "./cmp_res", True, forward_pass_call_back)
    print("Mixed Precision Model Accuracy: ", eval_func(sim.model, None))
    sim.export(filename_prefix='mixed_preision_quant_model', path='.')


if __name__ == "__main__":
    mixed_precision("/path/to/evaluation/dataset")
