:orphan:

############################
Model Guidelines for PyTorch
############################

To implement the Cross Layer Equalization API,  `aimet_torch.cross_layer_equalization.equalize_model()`,  AIMET creates a computing graph to analyze the sequence of operations in the model.

Certain model constructs prevent AIMET from creating and analyzing the computing graph. The following table lists these potential issues and workarounds.

.. note::

    These restrictions are not applicable if you are using the **Primitive APIs**.

+------------------------+------------------------------+-----------------------------------+
|     Potential Issue    | Description                  |     Workaround                    |
+========================+==============================+===================================+
| ONNX Export            | Use torch.onnx.export()      | If ONNX export fails, rewrite the |
|                        | to export your model.        | specific layer so that ONNX       |
|                        | Make sure ONNX export passes | export passes                     |
+------------------------+------------------------------+-----------------------------------+
| Slicing Operation      |Some models use               | Rewrite the x.view() statement    |
|                        |**torch.tensor.view()** in the| as follows:                       |
|                        |forward function as follows:  | `x = x.view(x.size(0), -1)`       |
|                        |x = x.view(-1, 1024) If       |                                   |
|                        |the view function is written  |                                   |
|                        |this way, it causes an issue  |                                   |
|                        |while creating the            |                                   |
|                        |computing graph               |                                   |
+------------------------+------------------------------+-----------------------------------+
| Bilinear, upsample     |Some models use the upsample  |Set the align_corners parameter to |
| operation              |operation in the forward      |False as follows:                  |
|                        |function as: x=               |x =                                |
|                        |torch.nn.functional.upsample( |torch.nn.functional.upsample(x,    |
|                        |x, size=torch.Size([129,129]) |size=torch.Size([129, 129]),       |
|                        |, mode = 'bilinear',          |mode='bilinear',                   |
|                        |align_corners=True)           |align_corners=False)               |
+------------------------+------------------------------+-----------------------------------+
| Deconvolution operation|The deconvolution operation   | There is no workaround available  |
|                        |is used in the DeepLabV3      | at this time. This issue will be  |
|                        |model. This is not            | addressed in a subsequent AIMET   |
|                        |supported by AIMET.           | release.                          |
+------------------------+------------------------------+-----------------------------------+
