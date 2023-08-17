Dependency Tree
===============

```bash

├── onnx-cpu
│   ├── reqs_deb_common.txt -> ../reqs_deb_common.txt
│   ├── reqs_deb_onnx_common.txt -> ../reqs_deb_torch_common.txt
│   ├── reqs_deb_torch_common.txt -> reqs_deb_onnx_common.txt
│   ├── reqs_pip_common.txt -> ../reqs_pip_common.txt
│   ├── reqs_pip_onnx_common.txt -> ../reqs_pip_torch_common.txt
│   ├── reqs_pip_onnx_cpu.txt
│   ├── reqs_pip_torch_common.txt -> reqs_pip_onnx_common.txt
│   └── reqs_pip_torch_cpu.txt -> reqs_pip_onnx_cpu.txt
├── onnx-gpu
│   ├── reqs_deb_common.txt -> ../reqs_deb_common.txt
│   ├── reqs_deb_onnx_gpu.txt
│   ├── reqs_deb_torch_common.txt
│   ├── reqs_deb_torch_gpu.txt
│   ├── reqs_pip_common.txt -> ../reqs_pip_common.txt
│   ├── reqs_pip_onnx_common.txt -> ../reqs_pip_torch_common.txt
│   ├── reqs_pip_onnx_gpu.txt
│   ├── reqs_pip_torch_common.txt -> reqs_pip_onnx_common.txt
│   └── reqs_pip_torch_gpu.txt -> reqs_pip_onnx_gpu.txt
├── reqs_deb_common.txt
├── reqs_deb_torch_common.txt
├── reqs_pip_common.txt
├── reqs_pip_tf_common_legacy.txt
├── reqs_pip_tf_common.txt
├── reqs_pip_torch_common_legacy.txt
├── reqs_pip_torch_common.txt
├── tf-cpu
│   ├── reqs_deb_common.txt -> ../reqs_deb_common.txt
│   ├── reqs_pip_common.txt -> ../reqs_pip_common.txt
│   ├── reqs_pip_tf_common.txt -> ../reqs_pip_tf_common.txt
│   └── reqs_pip_tf_cpu.txt
├── tf-cpu-tf24
│   ├── reqs_deb_common.txt -> ../reqs_deb_common.txt
│   ├── reqs_pip_common.txt -> ../reqs_pip_common.txt
│   ├── reqs_pip_tf_common.txt -> ../reqs_pip_tf_common_legacy.txt
│   └── reqs_pip_tf_cpu.txt
├── tf-gpu
│   ├── reqs_deb_common.txt -> ../reqs_deb_common.txt
│   ├── reqs_deb_tf_gpu.txt
│   ├── reqs_pip_common.txt -> ../reqs_pip_common.txt
│   ├── reqs_pip_tf_common.txt -> ../reqs_pip_tf_common.txt
│   └── reqs_pip_tf_gpu.txt
├── tf-gpu-tf24
│   ├── reqs_deb_common.txt -> ../reqs_deb_common.txt
│   ├── reqs_deb_tf_gpu.txt
│   ├── reqs_pip_common.txt -> ../reqs_pip_common.txt
│   ├── reqs_pip_tf_common.txt -> ../reqs_pip_tf_common_legacy.txt
│   └── reqs_pip_tf_gpu.txt
├── tf-torch-cpu
│   ├── reqs_deb_common.txt -> ../reqs_deb_common.txt
│   ├── reqs_pip_common.txt -> ../reqs_pip_common.txt
│   ├── reqs_pip_tf_common.txt
│   ├── reqs_pip_tf_cpu.txt
│   ├── reqs_pip_torch_common.txt -> ../reqs_pip_torch_common.txt
│   └── reqs_pip_torch_cpu.txt
├── torch-cpu
│   ├── reqs_deb_common.txt -> ../reqs_deb_common.txt
│   ├── reqs_deb_torch_common.txt -> ../reqs_deb_torch_common.txt
│   ├── reqs_pip_common.txt -> ../reqs_pip_common.txt
│   ├── reqs_pip_torch_common.txt -> ../reqs_pip_torch_common.txt
│   └── reqs_pip_torch_cpu.txt
├── torch-cpu-pt19
│   ├── reqs_deb_common.txt -> ../reqs_deb_common.txt
│   ├── reqs_deb_torch_common.txt -> ../reqs_deb_torch_common.txt
│   ├── reqs_pip_common.txt -> ../reqs_pip_common.txt
│   ├── reqs_pip_torch_common.txt -> ../reqs_pip_torch_common_legacy.txt
│   └── reqs_pip_torch_cpu.txt
├── torch-gpu
│   ├── reqs_deb_common.txt -> ../reqs_deb_common.txt
│   ├── reqs_deb_torch_common.txt -> ../reqs_deb_torch_common.txt
│   ├── reqs_deb_torch_gpu.txt
│   ├── reqs_pip_common.txt -> ../reqs_pip_common.txt
│   ├── reqs_pip_torch_common.txt -> ../reqs_pip_torch_common.txt
│   └── reqs_pip_torch_gpu.txt
└── torch-gpu-pt19
    ├── reqs_deb_common.txt -> ../reqs_deb_common.txt
    ├── reqs_deb_torch_common.txt -> ../reqs_deb_torch_common.txt
    ├── reqs_deb_torch_gpu.txt
    ├── reqs_pip_common.txt -> ../reqs_pip_common.txt
    ├── reqs_pip_torch_common.txt -> ../reqs_pip_torch_common_legacy.txt
    └── reqs_pip_torch_gpu.txt

11 directories, 71 files
```
