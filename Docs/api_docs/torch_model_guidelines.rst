==================
PyTorch Model Guidelines
==================

AIMET model dependencies
========================

In order to make full use of AIMET features, there are several guidelines users are encouraged to follow when defining
PyTorch models.

Define modules using class definitions instead of as functionals in the forward pass
------------------------------------------------------------------------------------
When possible, users should define neural network modules in the __init__ definition of the model, and not in the
forward definition of the model.
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

Avoid reuse of class defined modules
------------------------------------
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
        self.relu2 = torch.nn.ReLU2()
        ...

    def forward(...):
        ...
        x = self.relu(x)
        ...
        x2 = self.relu2(x2)
        ...

Use only torch.Tensor or tuples of torch.Tensors as model/submodule inputs and outputs
--------------------------------------------------------------------------------------
AIMET and Pytorch features being used within AIMET require that model inputs and outputs only contain torch.Tensor or
tuples of torch.Tensors. This applies for both the top level model input and output, as well as inputs and outputs for
all submodules in the model.