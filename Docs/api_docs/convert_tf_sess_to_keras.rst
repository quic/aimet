=============================================
Using AIMET Tensorflow APIs with Keras Models
=============================================

Introduction
============
.. ifconfig:: 'keras' in  included_features

    AIMET Keras API support is currently in development. In the meantime, AIMET Tensorflow APIs can be used with Keras
    models by working on the back-end session of the model. This example code shows a method for how to invoke AIMET on the
    back-end session as well as how to convert the returned session to a Keras model.

.. ifconfig:: 'keras' not in  included_features

    Currently AIMET APIs support Tensorflow sessions. This example code shows a method for how to use AIMET if you have a Keras model by invoking AIMET on the back-end session and converting the returned session to a Keras model.

APIs
====
The method involves performing four steps. The steps are:


**Step 1: Save the session returned by AIMET**

.. autofunction:: aimet_tensorflow.utils.convert_tf_sess_to_keras.save_tf_session_single_gpu

|

**Step 2: Model subclassing to load the corresponding session to Keras model**

.. autofunction:: aimet_tensorflow.utils.convert_tf_sess_to_keras.load_tf_sess_variables_to_keras_single_gpu

|

After these two steps, model can be used for single gpu training. For multi-gpu training, the next two steps needs to be followed:


**Step 3: Saving the Keras model from step 2 to make it compatible with distribution strategy**

.. autofunction:: aimet_tensorflow.utils.convert_tf_sess_to_keras.save_as_tf_module_multi_gpu

|

**Step 4: Model subclassing to load the corresponding Keras model**

.. autofunction:: aimet_tensorflow.utils.convert_tf_sess_to_keras.load_keras_model_multi_gpu

|


Code Example
============

**Required imports**

.. literalinclude:: ../tf_code_examples/converting_tf_session_to_keras.py
    :language: python
    :lines: 40, 51-52

**Steps to convert a TF session found after compression to Keras model**

.. literalinclude:: ../tf_code_examples/converting_tf_session_to_keras.py
    :language: python
    :pyobject: convert_tf_session_to_keras_model

Utility Functions
=================
**Required imports**

.. literalinclude:: ../tf_code_examples/converting_tf_session_to_keras.py
    :language: python
    :lines: 40-49

**Utility function to get session from Keras model**

.. literalinclude:: ../tf_code_examples/converting_tf_session_to_keras.py
    :language: python
    :pyobject: get_sess_from_keras_model

**Utility function to get a compressed session**

.. literalinclude:: ../tf_code_examples/converting_tf_session_to_keras.py
    :language: python
    :pyobject: compress_session

**Utility function for training**

.. literalinclude:: ../tf_code_examples/converting_tf_session_to_keras.py
    :language: python
    :pyobject: train
