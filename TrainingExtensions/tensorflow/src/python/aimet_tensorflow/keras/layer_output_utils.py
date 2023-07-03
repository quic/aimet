import os
import re
import tensorflow as tf
from aimet_tensorflow.keras.quantsim import QcQuantizeWrapper
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.backend import eager_learning_phase_scope
from collections import OrderedDict
from aimet_common.layer_output_utils import SaveInputOutput
from aimet_common.utils import AimetLogger
import json

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.LayerOutputs)

class LayerOutputUtil:
    """
    This class captures output of every layer of a keras (fp32/quantsim) model, creates a layer-output name to
    layer-output dictionary and saves the per layer outputs
    """
    def __init__(self, model: tf.keras.Model, save_dir: str = f"./KerasLayerOutput"):
        """
        Constructor - It initializes a few things that are required for capturing and naming layer-outputs.
        :param model: Keras (fp32/quantsim) model.
        :param is_quantsim_model: Enable when passing a quantsim keras model.
        """
        self.model = model
        self.layer_output_name_mapper = OrderedDict()
        self.layer_output_name_mapper_path = os.path.join(save_dir, "LayerOutputNameMapper.json")

        # Identify the axis-layout used for representing an image tensor
        axis_layout = 'NHWC' if tf.keras.backend.image_data_format() == 'channels_last' else 'NCHW'

        # Utility to save model inputs and their corresponding layer-outputs
        self.save_inp_out_obj = SaveInputOutput(save_dir, axis_layout=axis_layout)

        logger.info(f"Initialised LayerOutputUtil Class for Keras")

    def _get_layer_name(self, layer):
        if isinstance(layer, QcQuantizeWrapper):
            return layer.original_layer.output.name
        return layer.output.name

    def get_outputs(self, input_batch: tf.Tensor):
        """
        This function captures layer-outputs and renames them as per the AIMET exported model.
        :param input_batch: Batch of inputs for which we want to obtain layer-outputs.
        :return: layer-output name to layer-output batch dict
        """
        layer_name_to_layer_output_dict = OrderedDict()

        pred_func = K.function(inputs=[self.model.layers[0].input],
                               outputs=[layer.output for layer in self.model.layers])

        # run in test mode, i.e. 0 means test
        with eager_learning_phase_scope(value=0):
            output_pred = pred_func(input_batch)

        for layer_idx, layer in enumerate(self.model.layers):
            layer_output_name = self._get_layer_name(layer)

            # Replace all Non-word characters with "_" to make it a valid file name for saving the results
            # For Eg.: "conv2d/BiasAdd:0" gets converted to "conv2d_BiasAdd_0"
            modified_layer_output_name = re.sub(r'\W+', "_", layer_output_name)

            # Storing the actual layer output name to modified layer output name (valid file name to save) in a dict
            if not os.path.exists(self.layer_output_name_mapper_path):
                self.layer_output_name_mapper[layer_output_name] = modified_layer_output_name

            layer_name_to_layer_output_dict[modified_layer_output_name] = output_pred[layer_idx]

        return layer_name_to_layer_output_dict

    def generate_layer_outputs(self, input_batch: tf.Tensor):
        """
        This method captures output of every layer of a keras model & saves the inputs and corresponding layer-outputs to disk.
        This allows layer-output comparison either between original fp32 model and quantization simulated model or quantization
        simulated model and actually quantized model on-target to debug accuracy miss-match issues.

        :param input_batch: Batch of Inputs for which layer output need to be generated
        :return: None
        """

        batch_layer_name_to_layer_output = self.get_outputs(input_batch)
        self.save_inp_out_obj.save(input_batch, batch_layer_name_to_layer_output)

        # Saving the actual layer output name to modified layer output name (valid file name to save) in a json file
        if not os.path.exists(self.layer_output_name_mapper_path):
            json.dump(self.layer_output_name_mapper, open(self.layer_output_name_mapper_path, 'w'), indent=4)

        logger.info(f"Layer Outputs Saved")


