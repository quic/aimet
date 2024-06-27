.. _tutorials-migration-guide:

.. role:: python(code)
   :language: python

Migrate to QuantSim 2.0
===============================

Learn how to migrate your code from QuantSim 1.0 to QuantSim 2.0! 

Migration to QuantSim 2.0 enables access to new features, easier debugging, and simpler code that is easier to extend. This guide provides an overview of the migration process and describes the fundamental differences between the two versions. 

.. note::
    
    Please be advised that QuantSim 2.0 is an experimental feature whose APIs and behaviors are subject to change. 

Changes in QuantSim 2.0
----------------------------

Before migrating, it is important to understand the behavior and API differences between QuantSim 1.0 and QuantSim 2.0. Users can interact with QuantSim through the high level APIs in the same way. Methods like compute_encodings() and export() will remain the same. 

Under the hood, QuantSim 2.0 has a different set of building blocks and properties than QuantSim 1.0, as shown below:

.. image:: ../../images/quantsim2.0.png
  :width: 800


Migration Process
-----------------

The migration process includes the following:

1. Update imports of QuantizationSimModel and other features
2. Change how internal components of QuantizationSimModel are accessed
3. Remove any dependency on deprecated features

Imports
~~~~~~~~~~

To migrate to QuantSim 2.0, your imports should originate from the aimet_torch.v2 namespace and be replaced as shown below. If your code does not directly access lower-level components, no further code change is needed. 

===================== ====================================================== ==================================================================
AIMET Feature         :mod:`aimet_torch`                                     :mod:`aimet_torch.v2`
QuantSim              :class:`aimet_torch.quantsim.QuantizationSimModel`     :class:`aimet_torch.v2.quantsim.quantsim.QuantizationSimModel`
AdaRound              :class:`aimet_torch.adaround.adaround_weight.AdaRound` :class:`aimet_torch.v2.adaround.AdaRound`
Sequential MSE        :class:`aimet_torch.seq_mse.apply_seq_mse`             :class:`aimet_torch.v2.seq_mse.apply_seq_mse`
QuantAnalyzer         :class:`aimet_torch.quant_analyzer.QuantAnalyzer`      :class:`aimet_torch.v2.quant_analyzer.QuantAnalyzer`
AutoQuant             :class:`aimet_torch.auto_quant.AutoQuant`              :class:`aimet_torch.v2.auto_quant.AutoQuant`
===================== ====================================================== ==================================================================

In QuantSim 2.0, all implementation code is ported to Python. Users will no longer need to import from ``aimet_common.libpymo``. Please refer to the table in :ref:`Depracated Features <depracated-features>` to migrate these imports. 

All the other import statements will stay the same, including but not limited to:

* :python:`from aimet_common.defs import QuantScheme`
* :python:`from aimet_torch.cross_layer_equalization import equalize_model`
* :python:`from aimet_torch.model_preparer import prepare_model`


QuantizationSimModel
~~~~~~~~~~~~~~~~~~~~~

---------------------------------------------------
Moving from QuantWrapper to Quantized Modules
---------------------------------------------------

To enable quantization in QuantSim 1.0, modules are wrapped with a QuantizeWrapper. These wrapped modules can be accessed as follows:

>>> from aimet_torch.quantsim import QuantizationSimModel as QuantizationSimModelV1
>>> sim = QuantizationSimModelV1(…)
>>> all_quant_wrappers = sim.quant_wrappers()
>>> for quant_wrapper in sim.quant_wrappers():
    print(quant_wrapper)
StaticGridQuantWrapper(
    (_module_to_wrap): Linear(in_features=100, out_features=200, bias=True)
)
StaticGridQuantWrapper(
    (_module_to_wrap): ReLU()
)

In contrast, QuantSim 2.0 enables quantization through quantized nn.Modules - modules are no longer wrapped but replaced with a quantized version. For example, a nn.Linear would be replaced with QuantizedLinear, nn.Conv2d would be replace by QuantizedConv2d, and so on. The quantized module definitions can be found under `aimet_torch.v2.nn`. These quantized modules can be accessed as follows:

>>> from aimet_torch.v2.quantsim.quantsim import QuantizationSimModel as QuantizationSimModelV2
>>> sim2 = QuantizationSimModelV2(…)
>>> all_q_modules = sim2.qmodules()
>>> for q_module in sim2.qmodules():
    print(q_module)
QuantizedLinear(
    in_features=100, out_features=200, bias=True
    (param_quantizers): ModuleDict(
        (weight): QuantizeDequantize(shape=[1], bitwidth=8, symmetric=True)
        (bias): None
    )
    (input_quantizers): ModuleList(
        (0): QuantizeDequantize(shape=[1], bitwidth=8, symmetric=False)
    )
    (output_quantizers): ModuleList(
        (0): None
    )
)
FakeQuantizedReLU(
    (param_quantizers): ModuleDict()
    (input_quantizers): ModuleList(
        (0): None
    )
    (output_quantizers): ModuleList(
        (0): QuantizeDequantize(shape=[1], bitwidth=8, symmetric=False)
    )
)

For more information on Quantized modules, please refer to the API reference guide :ref:`here<api-torch-quantized-modules>`.

-------------------------------------------------------------------------------
Moving from StaticGrid and LearnedGrid Quantizer to Affine and Float Quantizer
-------------------------------------------------------------------------------

In QuantSim 1.0, we relied on StaticGridQuantizer and LearnedGridQuantizer. For both, floating point quantization could be enabled based on ``QuantizationDataType`` passed in. 

.. code-block:: Python

    from aimet_torch.tensor_quantizer import StaticGridPerChannelQuantizers
    from aimet_common.defs import QuantizationDataType

    fp_quantizer = StaticGridPerChannelQuantizer(data_type = QuantizationDataType.float, ...)
    affine_quantizer = StaticGridPerChannelQuantizer(data_type = QuantizationDataType.int, ...)


However, in QuantSim 2.0, this functionality is separated into an AffineQuantizer and a FloatQuantizer. Users can access these quantizers and related operations under `aimet_torch.v2.quantization`.

.. code-block:: Python

    import aimet_torch.v2.quantization as Q

    affine_q = Q.affine.Quantize(...)
    affine_qdq = Q.affine.QuantizeDequantize(...)
    fp_qdq = Q.float.FloatQuantizeDequantize(...)


From the wrapped module (QuantSim 1.0) or quantized module (QuantSim 2.0), the attributes to access the quantizers remain consistent: ``.input_quantizers`` for input quantizers, ``.output_quantizers`` for output quantizers, and ``.param_quantizers`` for parameter quantizers.

For more information on Quantizers, please refer to the API reference guide :ref:`here<api-torch-quantizers>`.

-----------------------------
Code Examples
-----------------------------
**Setup**

.. code-block:: Python

    # QuantSim 1.0
    from aimet_torch.quantsim import QuantizationSimModel as QuantizationSimModelV1

    sim1 = QuantizationSimModelV1(...)
    wrap_linear = sim1.model.linear

    # QuantSim 2.0
    from aimet_torch.v2.quantsim.quantsim import QuantizationSimModel as QuantizationSimModelV2

    sim2 = QuantizationSimModelV2(...)
    qlinear = sim2.model.linear 


**Case 1: Manually setting common attributes**

*Bitwidth*

.. code-block:: Python

    # QuantSim 1.0
    wrap_linear.param_quantizers['weight'].bitwidth = 4
    wrap_linear.input_quantizers[0].bitwidth = 4
    wrap_linear.output_quantizers[0].bitwidth = 4

    # QuantSim 2.0
    if qlinear.param_quantizers['weight']:
        module.param_quantizers['weight'].bitwidth = 4

    if qlinear.input_quantizers[0]:
        qlinear.input_quantizers[0].bitwidth = 4

    if qlinear.output_quantizers[0]:
        qlinear.output_quantizers[0].bitwidth = 4


*Symmetry*

.. code-block:: Python

    # QuantSim 1.0
    wrap_linear.param_quantizers['weight'].use_symmetric_encodings = True
    wrap_linear.param_quantizers['weight'].is_unsigned_symmetric = False
    wrap_linear.param_quantizers['weight'].use_strict_symmetric = False

    wrap_linear.input_quantizers[0].use_symmetric_encodings = True
    wrap_linear.input_quantizers[0].is_unsigned_symmetric = False
    wrap_linear.input_quantizers[0].use_strict_symmetric = False

    wrap_linear.output_quantizers[0].use_symmetric_encodings = True
    wrap_linear.output_quantizers[0].is_unsigned_symmetric = False
    wrap_linear.output_quantizers[0].use_strict_symmetric = False

    # QuantSim 2.0
    # Notes: simplified into two flags
    if qlinear.param_quantizers['weight']:
        qlinear.param_quantizers['weight'].symmetric = True
        qlinear.param_quantizers['weight'].signed = True

    if qlinear.input_quantizers[0]:
        qlinear.input_quantizers[0].symmetric = True
        qlinear.input_quantizers[0].signed = True

    if qlinear.output_quantizers[0]:
        qlinear.output_quantizers[0].symmetric = True
        qlinear.output_quantizers[0].signed = True

*Encoding Data*

.. code-block:: Python

    # QuantSim 1.0
    import libpymo

    if wrap_linear.param_quantizers['weight'].encoding:
        encoding = libpymo.TfEncoding()
        encoding.max = 1
        encoding.min = -1
        wrap_linear.param_quantizers['weight'].encoding = encoding
    
    if wrap_linear.input_quantizers[0].encoding:
        encoding = libpymo.TfEncoding()
        encoding.max = 1
        encoding.min = -1
        wrap_linear.input_quantizers[0].encoding = encoding
    
    if wrap_linear.output_quantizers[0].encoding:
        encoding = libpymo.TfEncoding()
        encoding.max = 1
        encoding.min = -1
        wrap_linear.output_quantizers[0].encoding = encoding

    # QuantSim 2.0
    # Notes: TfEncoding() is no longer used, encoding min/max are of type torch.nn.Parameter
    if qlinear.param_quantizers['weight']:
        qlinear.param_quantizers['weight'].min.copy_(-1.0) 
        module.param_quantizers['weight'].max.copy_(1.0)

    if qlinear.input_quantizers[0]:
        qlinear.input_quantizers[0].min.copy_(-1.0)
        qlinear.input_quantizers[0].max.copy_(1.0)

    if qlinear.output_quantizers[0]:
        qlinear.output_quantizers[0].min.copy_(-1.0)
        qlinear.output_quantizers[0].max.copy_(1.0)


**Case 2: Enabling and Disabling Quantization**

*Is quantization enabled?*

.. code-block:: Python

    # QuantSim 1.0 
    if wrap_linear.param_quantizers['weight'].enabled:
        pass
    
    # QuantSim 2.0
    # Notes: Quantizers no longer have an 'enabled' attribute. If a quantizer is present, it is enabled
    if qlinear.param_quantizers['weight']:
        pass

*Disabling Quantization*

.. code-block:: Python

    # QuantSim 1.0
    wrap_linear.param_quantizers['weight'].enabled = False

    # QuantSim 2.0
    # Notes: Quantizers can be disabled by setting them to None or using the utility API (_remove_input_quantizers, _remove_output_quantizers, _remove_param_quantizers)
    qlinear._remove_param_quantizers('weight')

*Enabling Quantization*

.. code-block:: Python

    # QuantSim 1.0
    wrap_linear.param_quantizers['weight'].enabled = True

    # QuantSim 2.0
    import aimet_torch.v2.quantization as Q
    qlinear.param_quantizers['weight'] = Q.affine.QuantizeDequantize(...)

*Temporarily disabling Quantization*

.. code-block:: Python

    # QuantSim 1.0
    assert wrap_linear.param_quantizers['weight'].enabled
    wrap_linear.param_quantizers['weight'].enabled = False
    # Run other code here
    wrap_linear.param_quantizers['weight'].enabled = True

    # QuantSim 2.0
    assert qlinear.param_quantizers['weight']
    with qlinear._remove_param_quantizers('weight'):
        assert qlinear.param_quantizers['weight'] is None
        # Run other code here

    assert qlinear.param_quantizers['weight']


**Case 3: Freezing encodings**

.. code-block:: Python
    
    # QuantSim 1.0
    if not wrap_linear.param_quantizers['weight']._is_encoding_frozen:
        wrap_linear.param_quantizers['weight'].freeze_encodings()

    # QuantSim 2.0
    if not qlinear.param_quantizers['weight']._is_encoding_frozen():
        qlinear.param_quantizers['weight']._freeze_encodings()

.. _depracated-features:

Depracated Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Components that are tied to the QuantSim 1.0 design and are no longer needed in QuantSim 2.0 will soon be sunset. Users will currently experience depracation warnings when accessing these APIs and features. 

In QuantSim 2.0, all source code will be implemented in Python to provide easier debugging and improved portability. Thus, invoking any modules defined in C through libpymo will not be supported. 

Below, you can see a list of depracated features and the recommended migration guideline: 


.. list-table:: 
   :widths: 25 25
   :header-rows: 1

   * - Depracated Feature
     - Replacement in V2
   * - libpymo.TensorQuantizer
     - :ref:`AffineQuantizer<api-torch-quantizers>`, :ref:`FloatQuantizer<api-torch-quantizers>`
   * - libpymo.RoundingMode  
     - Set to 'nearest' as default
   * - libpymo.TfEncoding  
     - AffineEncoding, FloatEncoding, VectorEncoding
   * - libpymo.EncodingAnalyzerForPython  
     - :ref:`MinMaxEncodingAnalyzer<api-torch-encoding-analyzer>`, :ref:`SqnrEncodingAnalyzer<api-torch-encoding-analyzer>`, :ref:`PercentileEncodingAnalyzer<api-torch-encoding-analyzer>`
