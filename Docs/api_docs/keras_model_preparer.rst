:orphan:

.. _api-keras-model-preparer:

==================
Model Preparer API
==================

AIMET Keras ModelPreparer API is used to prepare a Keras model that is not using the Keras Functional or Sequential API.
Specifically, it targets models that have been created using the subclassing feature in Keras. The ModelPreparer API will
convert the subclassing model to a Keras Functional API model. This is required because the AIMET Keras Quantization API
requires a Keras Functional API model as input.

Users are strongly encouraged to use AIMET Keras ModelPreparer API first and then use the returned model as input
to all the AIMET Quantization features. It is manditory to use the AIMET Keras ModelPreparer API if the model is
created using the subclassing feature in Keras, if any of the submodules of the model are created via subclassing, or if
any custom layers that inherit from the Keras Layer class are used in the model.



Top-level API
=============
.. autofunction:: aimet_tensorflow.keras.model_preparer.prepare_model


Code Examples
=============

**Required imports**

.. literalinclude:: ../keras_code_examples/model_preparer_code_example.py
    :language: python
    :start-after: # ModelPreparer Imports
    :end-before: # End ModelPreparer Imports

**Example 1: Model with Two Subclassed Layers**

We begin with a model that has two subclassed layers - `TokenAndPositionEmbedding` and `TransformerBlock`. This model
is taken from the `Transformer text classification example <https://keras.io/examples/nlp/text_classification_with_transformer/>`_.

.. literalinclude:: ../keras_code_examples/model_preparer_code_example.py
    :language: python
    :pyobject: TokenAndPositionEmbedding

.. literalinclude:: ../keras_code_examples/model_preparer_code_example.py
    :language: python
    :pyobject: TransformerBlock

.. literalinclude:: ../keras_code_examples/model_preparer_code_example.py
    :language: python
    :pyobject: get_text_classificaiton_model

Run the model preparer API on the model by passing in the model.

.. literalinclude:: ../keras_code_examples/model_preparer_code_example.py
    :language: python
    :pyobject: model_preparer_two_subclassed_layers

The model preparer API will return a Keras Functional API model. 
We can now use this model as input to the AIMET Keras Quantization API.


**Example 2: Model with Subclassed Layer as First Layer**

.. literalinclude:: ../keras_code_examples/model_preparer_code_example.py
    :language: python
    :pyobject: get_subclass_model_with_functional_layers

Run the model preparer API on the model by passing in the model and an Input Layer. Note that this is an example of when
the model preparer API will require an Input Layer as input.

.. literalinclude:: ../keras_code_examples/model_preparer_code_example.py
    :language: python
    :pyobject: model_preparer_subclassed_model_with_functional_layers

The model preparer API will return a Keras Functional API model. 
We can now use this model as input to the AIMET Keras Quantization API.


Limitations
===========
The AIMET Keras ModelPreparer API has the following limitations:

* The AIMET Keras ModelPreparer API is able to convert subclass layers that have arthmetic experssion in their call function.
  However, this API and Keras, will convert these operations to TFOPLambda layers which are not currently supported by AIMET Keras Quantization API. 
  If possible, it is recommended to have the subclass layers call function resemble the Keras Functional API layers.
    For example, if a subclass layer has two convolution layers in its call function, the call function should look like
    the following::

        def call(self, x, **kwargs):
            x = self.conv_1(x)
            x = self.conv_2(x)
            return x

* If the model starts with a subclassed layer, the AIMET Keras ModelPreparer API will need an Keras Input Layer as input.
  This is becuase the Keras Functional API requires an Input Layer as the first layer in the model. The AIMET Keras ModelPreparer API
  will raise an exception if the model starts with a subclassed layer and an Input Layer is not provided as input.
  