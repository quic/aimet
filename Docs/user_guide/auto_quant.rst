:orphan:

.. _ug-auto-quant:


===============
AIMET AutoQuant
===============

Overview
========

AutoQuant is a unified interface that integrates various post-training quantization techniques provided by AIMET.


Workflow
========

AutoQuant includes 1) batch norm folding, 2) cross-layer equalization, and 3) Adaround.
These techniques are applied in a best-effort manner until the model meets the evaluation goal.
If the model fails to satisfy the evaluation goal, AutoQuant will return the model to which the best combination of the above techniques is applied.

    .. image:: ../images/auto_quant_flowchart.png
