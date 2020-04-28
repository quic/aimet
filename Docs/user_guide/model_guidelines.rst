:orphan:

============================
Model Guidelines for PyTorch
============================

To implement the Cross Layer Equalization API,  aimet_torch.cross_layer_equalization.equalize_model(),  AIMET creates a computing graph to analyze the sequence of Operations in the model.
If your model is defined using certain constructs, it restricts AIMET from successfully creating and analyzing the computing graph. The following table lists the potential issues and workarounds.

Note: These restrictions are not applicable, if you are using the **Primitive APIs**


+------------------------+------------------------------+-----------------------------------+
|     Potential Issue    | Description                  |     Work Around                   |
+========================+==============================+===================================+
| ONNX Export            | Use torch.onnx.export()      | If ONNX export fails, rewrite the |
|                        | to export your model.        | specific layer so that ONNX       |
|                        | Make sure ONNX export passes | export passes                     |
+------------------------+------------------------------+-----------------------------------+
| Slicing Operation      |Some models use               | Rewrite the x.view() statement    |
|                        |**torch.tensor.view()** in the| as follows:                       |
|                        |forward function as follows:  | x = x.view(x.size(0), -1)         |
|                        |x = x.view(-1, 1024)          |                                   |
|                        |If view function is written   |                                   |
|                        |as above, it causes an issue  |                                   |
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
|                        |is used in DeepLabV3 model.   | at this time. This issue will be  |
|                        |This is currently not         | addressed in a subsequent AIMET   |
|                        |supported by AIMET            | release.                          |
+------------------------+------------------------------+-----------------------------------+
