.. _api-torch-model-validator:

========================
PyTorch Model Guidelines
========================

AIMET model dependencies
========================

In order to make full use of AIMET features, there are several guidelines users are encouraged to follow when defining
PyTorch models.

**Model should support conversion to onnx**

The model definition should support conversion to onnx, user could check compatibility of model for onnx conversion as
shown below::

    ...
    model = Model()
    torch.onnx.export(model, <dummy_input>, <onnx_file_name>):

**Define layers as modules instead of using torch.nn.functional equivalents**

When using activation functions and other stateless layers, PyTorch will allow the user to either

- define the layers as modules (instantiated in the constructor and used in the forward pass), or
- use a torch.nn.functional equivalent purely in the forward pass

For AIMET quantization simulation model to add simulation nodes, AIMET requires the former (layers defined as modules).
Changing the model definition to use modules instead of functionals, is mathematically equivalent and does not require
the model to be retrained.

As an example, if the user had::

    def forward(...):
        ...
        x = torch.nn.functional.relu(x)
        ...

Users should instead define their model as::

    def __init__(self,...):
        ...
        self.relu = torch.nn.ReLU()
        ...

    def forward(...):
        ...
        x = self.relu(x)
        ...

This will not be possible in certain cases where operations can only be represented as functionals and not as class
definitions, but should be followed whenever possible.

**Avoid reuse of class defined modules**

Modules defined in the class definition should only be used once. If any modules are being reused, instead define a new
identical module in the class definition.
For example, if the user had::

    def __init__(self,...):
        ...
        self.relu = torch.nn.ReLU()
        ...

    def forward(...):
        ...
        x = self.relu(x)
        ...
        x2 = self.relu(x2)
        ...

Users should instead define their model as::

    def __init__(self,...):
        ...
        self.relu = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        ...

    def forward(...):
        ...
        x = self.relu(x)
        ...
        x2 = self.relu2(x2)
        ...

**Use only torch.Tensor or tuples of torch.Tensors as model/submodule inputs and outputs**

Modules should use tensor or tuples of tensor for inputs and output in order to support conversion of the model to onnx.
For example, if the user had::

    def __init__(self,...):
    ...
    def forward(self, inputs: Dict[str, torch.Tensor]):
        ...
        x = self.conv1(inputs[‘image_rgb’])
        rgb_output = self.relu1(x)
        ...
        x = self.conv2(inputs[‘image_bw'])
        bw_output = self.relu2(x)
        ...
        return { 'rgb': rgb_output, 'bw': bw_output }

Users should instead define their model as::

    def __init__(self,...):
    ...
    def forward(self, image_rgb, image_bw):
        ...
        x = self.conv1(image_rgb)
        rgb_output = self.relu1(x)
        ...
        x = self.conv2(image_bw)
        bw_output = self.relu2(x)
        ...
        return rgb_output, bw_output

Model Validator Utility
=======================
AIMET provides a model validator utility to help check whether AIMET feature can be applied on a Pytorch model. The
model validator currently checks for the following conditions:

- No modules are reused
- Operations have modules associated with them and are not defined as Functionals (excluding a set of known operations)

In this section, we present models failing the validation checks, and show how to run the model validator, as well as
how to fix the models so the validation checks pass.

**Example 1: Model with reused modules**

We begin with the following model, which contains two relu modules sharing the same module instance.

.. literalinclude:: ../torch_code_examples/model_validator_code_example.py
   :language: python
   :pyobject: ModelWithReusedNodes
   :emphasize-lines: 13, 15

Import the model validator:

.. literalinclude:: ../torch_code_examples/model_validator_code_example.py
   :language: python
   :lines: 45

Run the model validator on the model by passing in the model as well as model input:

.. literalinclude:: ../torch_code_examples/model_validator_code_example.py
   :language: python
   :pyobject: validate_example_model

For each validation check run on the model, a logger print will appear::

    Utils - INFO - Running validator check <function validate_for_reused_modules at 0x7f127685a598>

If the validation check finds any issues with the model, the log will contain information for how to resolve the model::

    Utils - WARNING - The following modules are used more than once in the model: ['relu1']
    AIMET features are not designed to work with reused modules. Please redefine your model to use distinct modules for
    each instance.

Finally, at the end of the validation, any failing checks will be logged::

    Utils - INFO - The following validator checks failed:
    Utils - INFO -     <function validate_for_reused_modules at 0x7f127685a598>

In this case, the validate_for_reused_modules check informs that the relu1 module is being used multiple times in the
model. We rewrite the model by defining a separate relu instance for each usage:

.. literalinclude:: ../torch_code_examples/model_validator_code_example.py
   :language: python
   :pyobject: ModelWithoutReusedNodes
   :emphasize-lines: 9, 16

Now, after rerunning the model validator, all checks pass::

    Utils - INFO - Running validator check <function validate_for_reused_modules at 0x7ff577373598>
    Utils - INFO - Running validator check <function validate_for_missing_modules at 0x7ff5703eff28>
    Utils - INFO - All validation checks passed.

**Example 2: Model with functionals**

We start with the following model, which uses a torch linear functional layer in the forward pass:

.. literalinclude:: ../torch_code_examples/model_validator_code_example.py
   :language: python
   :pyobject: ModelWithFunctionalLinear
   :emphasize-lines: 17

Running the model validator shows the validate_for_missing_modules check failing::

    Utils - INFO - Running validator check <function validate_for_missing_modules at 0x7f9dd9bd90d0>
    Utils - WARNING - Ops with missing modules: ['matmul_8']
    This can be due to several reasons:
    1. There is no mapping for the op in ConnectedGraph.op_type_map. Add a mapping for ConnectedGraph to recognize and
    be able to map the op.
    2. The op is defined as a functional in the forward function, instead of as a class module. Redefine the op as a
    class module if possible. Else, check 3.
    3. This op is one that cannot be defined as a class module, but has not been added to ConnectedGraph.functional_ops.
    Add to continue.
    Utils - INFO - The following validator checks failed:
    Utils - INFO - 	<function validate_for_missing_modules at 0x7f9dd9bd90d0>

The check has identified matmul_8 as an operation with a missing pytorch module. In this case, it is due to reason #2
in the log, in which the layer has been defined as a functional in the forward function. We rewrite the model by
defining the layer as a module instead in order to resolve the issue.

.. literalinclude:: ../torch_code_examples/model_validator_code_example.py
   :language: python
   :pyobject: ModelWithoutFunctionalLinear
   :emphasize-lines: 10, 20