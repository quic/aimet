.. _tutorials-migration-guide:

Migrate to AIMET v2
===================

Learn how to migrate your code from AIMET 1.0 to AIMET 2.0! 

Migration to AIMET 2.0 enables access to new features, easier debugging, and simpler code that is easier to extend. This guide provides an overview of the migration process and describes the fundamental differences between the two versions. 

Changes in AIMET 2.0
--------------------

Before migrating, it is important to understand the behavior and API differences between AIMET 1.0 and AIMET 2.0. Under the hood, AIMETv2 has a different set of building blocks and properties than v1.  

At a high level, AIMET v2

* QuantizationSimModel is composed of quantized nn.Modules and Quantizers
* Moves all implementation code to Python for easier debugging and portability

Migration Process
-----------------

The migration process includes the following:

1. Update imports of QuantizationSimModel and other features
2. Change how internal components of QuantizationSimModel are accessed
3. Remove any dependency on deprecated features

**Imports**

To migrate to V2, replace your imports as shown below. If your code does not directly access lower-level components, no further code change is needed. 

AIMETv1

.. code-block:: Python

    # Any APIs imported outside aimet_torch.v2 subpackage use AIMET 1.0
    from aimet_torch.quantsim import QuantizationSimModel as QuantizationSimModelV1
    from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
    from aimet_torch.seq_mse import apply_seq_mse
    from aimet_torch.quant_analyzer import QuantAnalyzer
    from aimet_torch.batch_norm_fold import fold_all_batch_norms
    from aimet_torch.auto_quant import AutoQuant

AIMETv2

.. code-block:: Python

    from aimet_torch.v2.quantsim.quantsim import QuantizationSimModel as QuantizationSimModelV2
    from aimet_torch.v2.adaroundimportAdaround, AdaroundParameters
    from aimet_torch.v2.seq_mse import apply_seq_mse
    from aimet_torch.v2.quant_analyzer import QuantAnalyzer
    from aimet_torch.v2.batch_norm_fold import fold_all_batch_norms
    from aimet_torch.v2.auto_quant import AutoQuant

In AIMETv2, all implementation code is ported to Python. Users will no longer need to import from aimet_common.libpymo. Please refer to the table in Step 3 to migrate these imports. 

All the other import statements will stay the same, including but not limited to:

* from aimet_common.defs import QuantScheme
* from aimet_torch.cross_layer_equalization import equalize_model
* from aimet_torch.model_preparer import prepare_model


**QuantizationSimModel**

Users can interact with QuantSim through the higher level APIs in the same way. Methods like compute_encodings() and export() will remain the same. 

Let's delve into the differences when handling the internals of QuantSim.

*Access*

AIMETv1

To enable quantization in AIMET v1, modules are wrapped with a QuantizeWrapper and fake quantization ops are added. 
These wrapped modules and the associated TensorQuantizers can be accessed as follows:

.. code-block:: Python

    sim=QuantizationSimModelV1(…)

    all_quant_wrappers = sim.quant_wrappers()

    all_input_quantizers = [quant_wrapper.input_quantizers for _, quant_wrapper in sim.quant_wrappers()]

    all_output_quantizers = [quant_wrapper.output_quantizers for _, quant_wrapper in sim.quant_wrappers()]

    all_param_quantizers = [quant_wrapper.param_quantizers.values() for _, quant_wrapper in sim.quant_wrappers()]

AIMETv2

In contrast, AIMET v2 enables quantization through quantized nn.Modules and Quantizers. The modules are no longer wrapped but replaced with a quantized version. For example, a nn.Linear would be replaced with QuantizedLinear. 

Regarding quantizers, AIMETv2 has affine and floating point.  Affine quantizers perform integer quantization with an equidistant quantization grid. Floating point quantizers simulate the precision errors of low-bit floating point data types (float16, float8). 

You can learn more enabling quantization for custom modules, quantization on integer kernels, and other details in the AIMETv2 User Guide. 

These quantized modules and the associated Quantizers can be accessed as follows:

.. code-block:: Python

    import aimet_torch.v2 as aimet_v2
    sim2=QuantizationSimModelV2(…)

    all_qmodules=[
        module for module in sim.model.modules()
        if isinstance(module, aimet_v2.nn.BaseQuantizationMixin)]

    all_input_quantizers = [module.input_quantizers for module in all_qmodules]
    all_output_quantizers = [module.output_quantizers for module in all_qmodules]
    all_param_quantizers = [module.param_quantizers for module in all_qmodules]


*Properties*

After accessing the components, we will now explore their inner properties and how to get/set them. 

AIMETv1

.. code-block:: Python

    for _, wrapper in sim.quant_wrappers():

    if wrapper.param_quantizers:
        quantizer = wrapper.param_quantizers['weight']
        
        # 1. Bitwidth
        quantizer.bitwidth = 16
        
        # 2. Encoding Data
        if quantizer.encoding:
            assert type (quantizer.encoding) == libpymo.TfEncoding
            quantizer.encoding.bw = 8
            quantizer.encoding.min = -1
            quantizer.encoding.max = 1
            quantizer.encoding.scale = 1
            quantizer.encoding.offset = -1

        # 3. Symmetry
        quantizer.use_symmetric_encodings = True
        quantizer.use_strict_symmetric = True
        quantizer.is_unsigned_symmetric = False
        
        # 4. V1 attributes
        if_enabled = quantizer.enabled
        round_mode = quantizer.round_mode
        quant_scheme = quantizer.quant_scheme
    
    for quantizer in wrapper.input_quantizers:
        # 1. Bitwidth
        quantizer.bitwidth = 16
        
        # 2. Encoding Data
        if quantizer.encoding:
            #assert type (quantizer.encoding) == libpymo.TfEncoding
            quantizer.encoding.bw = 8
            quantizer.encoding.min = -1
            quantizer.encoding.max = 1
            quantizer.encoding.scale = 1
            quantizer.encoding.offset = -1

        # 3. Symmetry
        quantizer.use_symmetric_encodings = True
        quantizer.use_strict_symmetric = True
        quantizer.is_unsigned_symmetric = False
        
        # 4. V1 attributes
        if_enabled = quantizer.enabled
        round_mode = quantizer.round_mode
        quant_scheme = quantizer.quant_scheme
    
        
    for quantizer in wrapper.output_quantizers:
        # Same flow as wrapper.input_quantizers
        pass


AIMETv2 

.. code-block:: Python

    for module in sim2.model.modules():
        if isinstance(module, aimet_v2.nn.BaseQuantizationMixin):
            if module.param_quantizers:
                if module.param_quantizers['weight']:
                    # 1. Bitwidth
                    module.param_quantizers['weight'].bitwidth = 4
                    
                    # 2. Encoding Data
                    module.param_quantizers['weight'].min = nn.Parameter(torch.tensor([-1.0]))
                    module.param_quantizers['weight'].max = nn.Parameter(torch.tensor([1.0]))

                    encoding = module.param_quantizers['weight'].get_encoding()
                    assert type(encoding) == AffineEncoding

                    # 3. Symmetry
                    module.param_quantizers['weight'].symmetric = True
                    module.param_quantizers['weight'].signed = True

                    # 4. V1 attributes
                    module._remove_param_quantizers('weight') # Equivalent to module.param_quantizers['weight'] = None
            
            if module.input_quantizers[0]:
                # 1. Bitwidth
                module.input_quantizers[0].bitwidth = 4
                    
                # 2. Encoding Data
                module.input_quantizers[0].min = nn.Parameter(torch.tensor([-1.0]))
                module.input_quantizers[0].max = nn.Parameter(torch.tensor([1.0]))

                encoding = module.input_quantizers[0].get_encoding()
                assert type(encoding) == AffineEncoding

                # 3. Symmetry
                module.input_quantizers[0].symmetric = True
                module.input_quantizers[0].signed = True

                # 4. V1 attributes
                module._remove_input_quantizers(0) # Equivalent to module.input_quantizers[0] = None


            if module.output_quantizers[0]:   
                # Same flow as module.input_quantizers for #1-3
                
                # 4. V1 attributes
                module._remove_output_quantizers(0) # Equivalent to module.output_quantizers[0] = None
                pass

For more detail on their differences: 

1. Encoding Data: In AIMET v2, the encoding min/max are now stored as torch.nn.Parameters, which allows users to access these values through the 'parameters' iterator. Encodings are no longer represented by libpymo.TfEncoding but AffineEncoding. 
2. Symmetry: In AIMETv2, we have simplified our design into two flags - symmetric and signed. We support two quantization modes - unsigned asymmetric and signed symmetric. 
3. Miscellaneous Attributes:
   In AIMETv2, quantizers no longer have an 'enabled' attribute. If a quantizer is present, it is enabled and can be disabled by setting it to None. When handling Quantizers, users should check if they are None. 
   
   You can use _remove_input_quantizers, _remove_output_quantizers, and _remove_param_quantizers to remove the respective quantizers as shown below: 
    
   .. code-block:: Python
    
      qlinear = sim2.model.linear 
      with qlinear._remove_input_quantizers(0): # Temporarily removes the 0th input quantizer
         ...
      with qlinear._remove_input_quantizers(): # Temporarily removes all input quantizers
         ...
      qlinear._remove_input_quantizers(0) # Permanently removes the 0th input quantizer
      qlinear._remove_input_quantizers() # Permanently removes all input quantizers
    
   Rounding_mode  and Quant Scheme are no longer an attributes in v2. Rounding_mode will always be set to nearest in v2. 

**Depracated Features**

Components that are tied to the AIMETv1 design and are no longer needed in v2 will soon be sunset. Users will currently experience depracation warnings when accessing these APIs and features. 

In AIMETv2, all source code will be implemented in Python to provide easier debugging and improved portability. Thus, invoking any modules defined in C through libpymo will not be supported. 

Below, you can see a list of depracated features and the recommended migration guideline: 


.. list-table:: 
   :widths: 25 25
   :header-rows: 1

   * - Depracated Feature
     - Replacement in V2
   * - libpymo.TensorQuantizer
     - AffineQuantizer, FloatQuantizer
   * - libpymo.RoundingMode  
     - Set to 'nearest' as default
   * - libpymo.TfEncoding  
     - AffineEncoding, VectorEncoding
   * - libpymo.EncodingAnalyzer  
     - MinMaxEncodingAnalyzer, SqnrEncodingAnalyzer, PercentileEncodingAnalyzer
  