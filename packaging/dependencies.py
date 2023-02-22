#!/usr/bin/env python3
#==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
#==============================================================================

import argparse
import functools
import operator
import os
import sys

from collections.abc import Iterable
from typing import List

_CURRENT_MODULE = sys.modules[__name__]


def get_reqs_apt_torch(cuda_version: str = "cpu", tf_verson: str = "", torch_version: str = "") -> List[str]:
    return []


def get_reqs_pip_torch(cuda_version: str = "cpu", tf_verson: str = "", torch_version: str = "") -> List[str]:
    TORCH = {
        "1.9.1": ["torch==1.9.1", "torchvision==0.10.1"],
    }
    cu = "" if cuda_version == "cpu" else f"-cu{cuda_version.replace('.', '')}"
    return [
        f"cumm{cu}==0.2.8",
        f"onnxruntime{'-gpu' if cu else ''}",
        f"spconv{cu}==2.1.20",
        f"--extra-index-url https://download.pytorch.org/whl/{cuda_version.replace('.', '')}"
    ] + TORCH[torch_version]


def get_reqs_apt_tf(cuda_version: str = "cpu", tf_verson: str = "", torch_version: str = ""):
    return []


def get_reqs_pip_tf(cuda_version: str = "cpu", tf_verson: str = "", torch_version: str = ""):
    return [
        "bert-tensorflow",
        "holoviews==1.12.7",
        "numpy==1.19.5",
        "scikit-learn",
        "tensorflow-hub",
        "tensorflow-model-optimization",
    ] + [f"tensorflow-{cuda_version if cuda_version=='cpu' else 'gpu'}=={tf_verson}"]


def get_reqs_pip(cuda_version: str = "cpu", tf_verson: str = "", torch_version: str = ""):
    return [
        "bokeh==1.2.0",
        "dataclasses",
        "h5py==2.10.0",
        "hvplot==0.4.0",
        "Jinja2==3.0.3",
        "jsonschema",
        "osqp",
        "pandas==1.1.5;python_version<'3.7'",
        "pandas==1.4.3;python_version>='3.7'",
        "progressbar2",
        "protobuf==3.19.4;python_version<'3.7'",
        "protobuf==3.20.1;python_version>='3.7'",
        "pybind11",
        "PyYAML",
        "scipy==1.2.1;python_version<'3.7'",
        "scipy==1.8.1;python_version>='3.7'",
        "setuptools",
        "tqdm",
        "transformers",
    ]


def get_reqs_pip_dev(cuda_version: str = "cpu", tf_verson: str = "", torch_version: str = ""):
    return [
        "cmake==3.17",
        "ninja",
        "matplotlib",
        "pybind11",
        "pytest",
        "wheel",
    ]


def get_reqs_apt(cuda_version: str = "cpu", tf_verson: str = "", torch_version: str = ""):
    return [
        "liblapacke",
    ]


def get_reqs_apt_dev(cuda_version: str = "cpu", tf_verson: str = "", torch_version: str = ""):
    return [
        "g++",
        "gcc",
        "git",
        "liblapacke-dev",
        "pkg-config",
        "zlib1g-dev",
    ]


def get_parser():
    parser = argparse.ArgumentParser(description="Get AIMET dependencies for a package manager. ")
    parser.add_argument("pkg_mgr", choices=["apt", "pip"], help="A package manager")
    parser.add_argument("--dev", action="store_true", help="Include dev dependencies")
    parser.add_argument("--cuda", default=os.environ.get("AIMET_CU_VER", "cpu"),
        help="Cuda version, by default will use AIMET_CU_VER environment variable or 'cpu'.")
    parser.add_argument("--tensorflow", default=os.environ.get("AIMET_TF_VER", ""),
        help="Tensorflow version, by default will use AIMET_TF_VER environment variable or ''.")
    parser.add_argument("--torch", default=os.environ.get("AIMET_PT_VER", ""),
        help="Torch version, by default will use AIMET_PT_VER environment variable or ''.")
    return parser


def get_dependencies(argv = None):
    parser = get_parser()
    args = parser.parse_args(argv)
    pkg_mgr = args.pkg_mgr.lower()
    fn_names = {
        f"get_reqs_{pkg_mgr}{'_dev' if args.dev else ''}",
        f"get_reqs_{pkg_mgr}",
        f"get_reqs_{pkg_mgr}{'_torch' if args.torch else ''}",
        f"get_reqs_{pkg_mgr}{'_tf' if args.tensorflow else ''}",
        }
    fn_args = {
        "cuda_version": args.cuda,
        "tf_verson": args.tensorflow,
        "torch_version": args.torch,
    }
    funcs = operator.attrgetter(*fn_names)(_CURRENT_MODULE)
    if not isinstance(funcs, Iterable):
        funcs = [funcs]
    deps = functools.reduce(lambda acc, fn: acc + fn(**fn_args), funcs, [])
    print("\n".join(deps))

if __name__ == '__main__':
    get_dependencies()