:orphan:

.. _api-torch-model-preparer:

===============================
AIMET PyTorch ModelPreparer API
===============================

AIMET PyTorch ModelPreparer API uses new graph transformation feature available in PyTorch 1.9+ version and
automate model definition changes for example changing functional to module for activation functions, elementwise
functions and unrolling into independent modules when modules are reused. AIMET PyTorch ModelPreparer API requires
minimum PyTorch 1.9 version.

Top-level API
=============
.. autofunction:: aimet_torch.model_preparer.prepare_model

Code Examples
=============

**Required imports**

.. literalinclude:: ../torch_code_examples/model_preparer_code_example.py
   :language: python
   :start-after: # ModelPreparer imports
   :end-before: # End of import statements


**Example 1: Model with Functional relu**

We begin with the following model, which contains two functional relus and relu method inside forward method.

.. literalinclude:: ../torch_code_examples/model_preparer_code_example.py
   :language: python
   :pyobject: ModelWithFunctionalReLU
   :emphasize-lines: 11, 12, 14, 15

Run the model preparer on the model by passing in the model. Make sure that model is in eval() mode.

.. literalinclude:: ../torch_code_examples/model_preparer_code_example.py
   :language: python
   :pyobject: model_preparer_functional_example

After that, we get model_transformed, which is functionally same as original model. User can verify this by comparing
the outputs of both models. model_transformed should have all three functional relus now converted to torch.nn.ReLU
modules.


**Example 2: Model with reused torch.nn.ReLU module**

We begin with the following model, which contains torch.nn.ReLU module which is used at multiple instances inside
model forward function.

.. literalinclude:: ../torch_code_examples/model_preparer_code_example.py
   :language: python
   :pyobject: ModelWithReusedReLU
   :emphasize-lines: 13, 15, 18, 20

Run the model preparer on the model by passing in the model. Make sure that model is in eval() mode.

.. literalinclude:: ../torch_code_examples/model_preparer_code_example.py
   :language: python
   :pyobject: model_preparer_reused_example

After that, we get model_transformed, which is functionally same as original model. User can verify this by comparing
the outputs of both models. model_transformed should have separate torch.nn.Module instances.

**Example 3: Model with elementwise Add**

We begin with the following model, which contains elementwise Add operation inside model forward function.

.. literalinclude:: ../torch_code_examples/model_preparer_code_example.py
   :language: python
   :pyobject: ModelWithElementwiseAddOp
   :emphasize-lines: 10

Run the model preparer on the model by passing in the model. Make sure that model is in eval() mode.

.. literalinclude:: ../torch_code_examples/model_preparer_code_example.py
   :language: python
   :pyobject: model_preparer_elementwise_add_example

After that, we get model_transformed, which is functionally same as original model. User can verify this by comparing
the outputs of both models. model_transformed should have elemenwise_ops.Add module added.