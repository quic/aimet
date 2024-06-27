.. _api-torch-blockwise-quantization:

.. currentmodule:: aimet_torch.v2.nn

.. warning::
    This feature is under heavy development and API changes may occur without notice in future versions.

======================
Blockwise Quantization
======================

When performing integer quantization, it is necessary to determine quantization parameters (also known as encodings)
like scale and offset in order to define a quantization grid for mapping floating point values to their quantized integer
counterparts. This process of determining appropriate quantization parameters is known as calibration or computing encodings.

When performing calibration for a particular tensor, one can choose to come up with encodings to cover the whole tensor,
or to split the tensor into sections and compute encodings for each section. Below we describe several ways in which
tensors can be split, along with pros and cons of each:

    - **Per Tensor quantization**: All values in the entire tensor are grouped collectively, and a single set of encodings
      are determined. Benefits include less computation and storage space needed to produce a single set of encodings.
      Drawbacks are that outlier values in the tensor negatively affect the encodings which are used to quantize all other
      values in the tensor.

    - **Per Channel quantization**: Values in the tensor are split into individual channels (typically in the output channels
      dimension). The number of encodings computed for the tensor is equal to the number of channels. The benefit as
      compared to Per Tensor quantization are that outlier values would only influence encodings for the channel the
      outlier resides in, and would not affect encodings for values in other channels.

    - **Blockwise quantization**: Values in the tensor are split into chunks across multiple dimensions. This further
      improves the granularity at which encoding parameters are found, isolating outliers and producing a more optimized
      quantization grid for each block, at the cost of higher computation complexity and more storage used to hold an
      increased number of encodings.

Blockwise quantization is supported as part of the QuantizeDequantize class:

.. autoclass:: aimet_torch.v2.quantization.affine.QuantizeDequantize

Blockwise quantization can be enabled on an individual quantizer basis by instantiating a new QuantizeDequantize object with
the desired settings and replacing an existing quantizer with the new quantizer.

The block_size argument can be used to specify particular block sizes for each dimension of the tensor.
Note that there exists a relationship between the QuantizeDequantize's shape and block_size arguments, along with the shape
of the actual tensor being quantized.

The following rules must apply:

    - If block_size is provided, the length of block_size must match the number of dimensions of the tensor being quantized.
      For example, if the tensor's shape is 4 dimensional, the block size must also be of length 4, specifying one block
      size per dimension.

    - Block sizes for each dimension must evenly divide the size of the tensor in the correpsonding dimension. For example,
      if a tensor's shape is (2, 2, 6, 10), a valid block_size would be (2, 1, 3, 5), since each block size is divisible
      by the tensor's corresponding dimension size.

    - The shape argument of the QuantizeDequantize (not to be confused with the tensor shape) must be the same length of
      block_size, and each element of shape must be equal to the tensor's shape for that dimension divided by the block
      size for that dimension. In other words, the shape argument represents the number of blocks for a particular dimension.
      Thus, number of blocks * block size will lead to the tensor's size for a dimension.

    - For each dimension, a block size value of '-1' is permitted. In such cases, the block size is automatically determined
      based on the tensor's shape in that dimension and the QuantizeDequantize object's shape. This is essentially determining
      the block size for a dimension given the tensor's size along with the number of blocks for that dimension.

Note: While the QuantizeDequantize object supports arbitrary block sizes for experimental purposes, Qualcomm runtime restricts
blockwise quantization to take place with the following constraints:

    - Blockwise quantization must run on weight quantizers only.

    - Block sizes must be set to 1 for the output channel dimension, may take arbitrary values for the input channel
      dimension (it must still be divisible by the input channel tensor shape), and must have block sizes equal to the
      tensor sizes for all other dimensions.

    - Layers with weights running with blockwise quantization must themselves be running with floating-point quantized
      activations.

=======================================
Low Power Blockwise Quantization (LPBQ)
=======================================

Qualcomm runtime supports a version of Blockwise Quantization referred to as Low Power Blockwise Quantization (LPBQ).

In this scheme, blockwise encodings at a lower bitwidth are determined and then adjusted such that they lie on a common
higher bitwidth per channel grid. This allows models to achieve benefits of blockwise quantization while allowing runtimes
to leverage existing per channel kernels in order to run the model.

LPBQ quantization is supported as part of the GroupedBlockQuantizeDequantize class:

.. autoclass:: aimet_torch.v2.quantization.affine.GroupedBlockQuantizeDequantize

In addition to the block_size argument described in the Blockwise Quantization section, LPBQ introduces two new arguments:

    - **decompressed_bw**: The higher bitwidth value for the per channel grid which the lower bitwidth blockwise encodings will
      expand to. Decompressed_bw must be greater than or equal to the bitwidth of the quantizer.

    - **block_grouping**: The block_grouping argument defines the number of blocks for each dimension which will be grouped
      together when expanding the lower bitwidth blockwise encodings to the higher bitwidth per channel encodings.
      The block grouping for a particular dimension must be divisible by the number of blocks for that dimension.

As with block size, a block grouping value of '-1' is valid, and will correspond automatically to the number of blocks for
that dimension.

Note: While the GroupedBlockQuantizeDequantize quantier supports arbitrary block groupings for experimental purposes,
Qualcomm runtime restricts LPBQ to take place with the following constraints:

    - Blockwise quantization must run on weight quantizers only.

    - Block sizes must be set to 1 for the output channel dimension, may take arbitrary values for the input channel
      dimension (it must still be divisible by the input channel tensor shape), and must have block sizes equal to the
      tensor sizes for all other dimensions.

    - Block groupings must be set to '1' for all dimensions, except for the input channels dimension, which should be
      set to the number of blocks for that dimension.


Top Level API
=============

Several top level API functions exist to make it easier to configure blockwise quantization and LPBQ quantization for
a model:

.. autofunction:: aimet_torch.v2.quantsim.config_utils.set_activation_quantizers_to_float

This utility provides a method for configuring certain quantized layers in a model to float quantization. This can be used
in conjunction with :func:'set_blockwise_quantization_for_weights' in order to switch layers running blockwise quantization
into floating-point activation quantization.

Of significance is the second argument in the function, which allows users to specify a subset of layers to switch to
floating-point quantization. Refer to the function docstring for valid types of inputs this argument supports.

Below are examples showing various ways to call this API:

.. code-block:: Python

    import torch
    from aimet_torch.v2.quantsim.config_utils import set_activation_quantizers_to_float

    # Assume 'sim' is a QuantizationSimModel object imported from aimet_torch.v2.quantsim

    # Set activation quantizers of all Conv and Linear layers to float:
    set_activation_quantizers_to_float(sim, [torch.nn.Linear, torch.nn.Conv2d], dtype=torch.float16)

    # Set activation quantizers of specific model layers to float:
    set_activation_quantizers_to_float(sim, [sim.model.conv2, sim.model.linear1], dtype=torch.float16)

    # Set activation quantizers of Conv layers with input channels dim == 16 to float:
    set_activation_quantizers_to_float(sim, lambda module: isinstance(module, torch.nn.Conv2d) and module.weight.shape[1] == 16,
                                       dtype=torch.float16)


.. autofunction:: aimet_torch.v2.quantsim.config_utils.set_blockwise_quantization_for_weights

This utility allows users to configure certain quantized layers in a model to use blockwise quantization with a specified
block_size. For this API, the block_size argument can be a single integer value instead of an array. In this case, all
affected layers would be set to a block size of 1 for the output channels dimension, the specified value for the input
channels dimension, and block size equal to dimension size for all other dimensions.

Note that this allows layers with differing weight shapes (ex. Conv layers with 4d weights vs. Linear layers with 2d weights)
to be handled with a single API call. If an array for block_size is passed in instead, due to the requirement for the
length of the block_size array to match the number of dimensions for a particular layer's weight, the API would need to be
called multiple times for each set of layers with different weight dimensions.

.. autofunction:: aimet_torch.v2.quantsim.config_utils.set_grouped_blockwise_quantization_for_weights

This utility allows users to configure certain quantized layers in a model to use grouped blockwise quantization with a
specified decompressed_bw, block_size, and block_grouping. Similar to :func:'set_blockwise_quantization_for_weights',
block_grouping can be a single value, in which case it will automatically be applied to the input_channel's dimension,
with all other dimensions using a block_grouping value of 1.

Additionally, as different layers may have a different number of blocks for the input channels dimension given the same
block size, a single block_grouping value of '-1' may be used, in which case the input channels dimension will automatically
use a block_grouping value equal to the number of blocks for any affected layer. This effectively allows users to configure
all affected layers to LPBQ quantization with a single API call.

Export
======

Using blockwise quantization results in a larger number of encodings produced as compared to per tensor or per channel
quantization. As a result, a new method of exporting encodings to json has been developed to both reduce the exported
encodings file size as well as reduce the time needed to write exported encodings to the json file.

The following code snippet shows how to export encodings in the new 1.0.0 format:

.. code-block:: Python

    from aimet_common import quantsim

    # Assume 'sim' is a QuantizationSimModel object imported from aimet_torch.v2.quantsim

    # Set encoding_version to 1.0.0
    quantsim.encoding_version = '1.0.0'
    sim.export('./data', 'exported_model', dummy_input)

The 1.0.0 encodings format is supported by Qualcomm runtime and can be used to export Per Tensor, Per Channel, Blockwise,
and LPBQ quantizer encodings. If Blockwise and/or LPBQ quantizers are present in the model, the 1.0.0 format must be
used when exporting encodings for Qualcomm runtime.
