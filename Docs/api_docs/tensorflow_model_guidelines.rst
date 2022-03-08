
===========================
TensorFlow Model Guidelines
===========================

In order to make full use of AIMET features, there are several guidelines users should follow when defining
TensorFlow models.

**If model has BatchNormalization (BN) layers**

If model has BatchNormalization (BN) layers, then user should set it's trainble flag to False and recompile the model
before AIMET usage. This is one of the limitations with TensorFlow 2.x but If you are using TensorFlow 1.x,
then this step is not required::

    ...
    model = Model()
    from aimet_tensorflow.utils.graph import update_keras_bn_ops_trainable_flag
    model = update_keras_bn_ops_trainable_flag(model, load_save_path="./", trainable=False)


.. autofunction:: aimet_tensorflow.utils.graph.update_keras_bn_ops_trainable_flag

**If model has Recurrent (RNN, LSTM etc.) layers**

Recurrent layers (RNN, LSTM) are not supported with TensorFlow 2.x and only supported with TensorFlow 1.x.