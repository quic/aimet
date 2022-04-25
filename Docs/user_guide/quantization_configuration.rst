.. _ug-quantsim-config:

======================================
Quantization Simulation Configuration
======================================
AIMET allows the configuration of quantizer placement in accordance with a set of rules specified in a json configuration file if provided when the Quantization Simulation API is called.
In the configuration file, quantizers can be turned on and off, and/or configured with asymmetric or symmetric encodings.
The general use case for this file would be for users to match the quantization rules for a particular Runtime they would like to simulate.

The configuration file contains six main sections, in increasing amounts of specificity:

.. image:: ../images/quantsim_config_file.png

Rules defined in a more general section can be overruled by subsequent rules defined in a more specific case.
For example, one may specify in "defaults" for no layers to be quantized, but then turn on quantization for specific layers in the "op_type" section.

It is advised for the user to begin with the default configuration file under

|default-quantsim-config-file|

How to configure individual Configuration File Sections
=======================================================
1. **defaults**:

.. literalinclude:: ../torch_code_examples/quantsim_config_example.py
   :language: python
   :start-after: # defaults start
   :end-before:  # defaults end

The "defaults" section shown above, configures the following:
    - All the Ops' output are quantized
    - All the parameters (Weights and Biases) are quantized
    - Strict Symmetric quantization is disabled.
    - Unsigned symmetric quantization is enabled
    - Per Channel Quantization is disabled.
Based on the Runtime support available to a specific hardware, the user can modify the configuration. In addition, for any Op and parameter the default configuration above could be overridden as shown in the sections below.

2. **params**:

.. literalinclude:: ../torch_code_examples/quantsim_config_example.py
   :language: python
   :start-after: # params start
   :end-before:  # params end


In the "defaults" section, all the params (weights and bias) are configured to be quantized. In this "params" section,
for the Bias param, quantization is disabled.

3. **op_type**:

.. literalinclude:: ../torch_code_examples/quantsim_config_example.py
   :language: python
   :start-after: # op_type start
   :end-before:  # op_type end

The above configuration snippet is not part of the default_config.json file. It is shown here for illustrative purposes.
In this "op_type" section, the default ops configuration in the "defaults" section is overridden for the "Squeeze" Op.
For the "Squeeze" Op, the quantization has been disabled.

4. **supergroups**:

.. literalinclude:: ../torch_code_examples/quantsim_config_example.py
   :language: python
   :start-after: # supergroups start
   :end-before:  # supergroups end

Supergroups are a sequence of Ops which are treated together as single Op by the Runtime. In order to simulate the
Runtime behavior, QuatntSim treats each sequence of Ops configured in this section as a single Op. For example,
the "op_list": ["Conv, "relu"] is a Supergroup. In this case, QuantSim does not quantize the output of the "Conv" Op
but quantizes the output of the "Relu" Op.

5. **model_input**:

.. literalinclude:: ../torch_code_examples/quantsim_config_example.py
   :language: python
   :start-after: # model_input start
   :end-before:  # model_input end

The "model_input" section is used to configure the quantization of the input to the model. In the above example,
the model's input is quantized.

Note:
If you prefer to NOT quantize the input to the model,  remove the line, "is_input_quantized": "True". Do not set it to False.

6. **model_output**:

.. literalinclude:: ../torch_code_examples/quantsim_config_example.py
   :language: python
   :start-after: # model_output start
   :end-before:  # model_output end

The "model_output" section is used to configure the quantization of the output of the model. In the above example,
the model's output is not quantized by leaving the entry blank.