#!/usr/bin/env python3
# ==============================================================================
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
# ==============================================================================

import argparse
import functools
import operator
import os
import re
import sys

from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional

_CURRENT_MODULE = sys.modules[__name__]
_SCRIPT_DIR = Path(__file__).parent

_TORCH = {
    "1.9.1": ["torch==1.9.1+{cu}", "torchvision==0.10.1+{cu}"],
    "1.10.0": ["torch==1.10.0+{cu}", "torchvision==0.11.0+{cu}"],
    "1.10.1": ["torch==1.10.1+{cu}", "torchvision==0.11.2+{cu}"],
    "1.11.0": ["torch==1.11.0+{cu}", "torchvision==0.12.0+{cu}"],
    "1.12.0": ["torch==1.12.0+{cu}", "torchvision==0.13.0+{cu}"],
    "1.12.1": ["torch==1.12.1+{cu}", "torchvision==0.13.1+{cu}"],
}


def _get_deps_dir(cuda_version: str, tf_verson: str, torch_version: str) -> Path:
    """Returns a directory where reqs_pip_[framework]*.txt are located."""
    dep_dir = []
    if tf_verson:
        dep_dir.append("tf")
    if torch_version:
        dep_dir.append("torch")
    dep_dir.append("cpu" if cuda_version == "cpu" else "gpu")
    return _SCRIPT_DIR / "dependencies" / "-".join(dep_dir)


def _get_pip_reqs_for_framework(
    cuda_version: str, tf_verson: str, torch_version: str, framework: str
) -> List[str]:
    """Parse reqs_pip_[framework]*.txt files and return requirements.
    Parsing is dumb, fancy syntax of requrements.txt is not supported.
    """
    def _filter_reqs(line: str) -> List[str]:
        if line.startswith("#"):
            return False
        if line.startswith("-"):
            return False
        return True

    reqs = []
    for file in (
        f"reqs_pip_{framework}_common.txt",
        f"reqs_pip_{framework}_{'cpu' if cuda_version == 'cpu' else 'gpu'}.txt",
    ):
        text = (_get_deps_dir(cuda_version, tf_verson, torch_version) / file).read_text(
            encoding="utf8"
        )
        reqs.extend(line for line in text.splitlines() if _filter_reqs(line))
    return reqs


def get_reqs_apt_torch(
    cuda_version: str = "cpu", tf_verson: str = "", torch_version: str = ""
) -> List[str]:
    """Return list of packages which should be installd via APT to use aimet with pytorch"""
    return []


def get_reqs_pip_torch(
    cuda_version: str = "cpu", tf_verson: str = "", torch_version: str = ""
) -> List[str]:
    """Return list of packages which should be installd via PIP to use aimet with pytorch"""
    reqs = _get_pip_reqs_for_framework(cuda_version, tf_verson, torch_version, "torch")
    # Delete torch and torchvision with pinned version
    reqs = filter(lambda r: not (r.startswith("torch==") or r.startswith("torchvision==")), reqs)
    # Almost all cuda packages use `cuXYZ` notation to define a cuda version,
    # Replace it to user defined cuda version
    cu = "cpu" if cuda_version == "cpu" else f"cu{cuda_version.replace('.', '')}"
    if cu != "cpu":
        reqs = map(lambda r: re.sub("cu[0-9]{3}", cu, r), reqs)
    reqs = (
        list(reqs)
        + [r.format(cu=cu) for r in _TORCH[torch_version]]
        + ["--extra-index-url=https://download.pytorch.org/whl/torch/"]
    )
    return reqs


def get_reqs_apt_tf(
    cuda_version: str = "cpu", tf_verson: str = "", torch_version: str = ""
) -> List[str]:
    """Return list of packages which should be installd via APT to use aimet with tensorflow"""
    return []


def get_reqs_pip_tf(
    cuda_version: str = "cpu", tf_verson: str = "", torch_version: str = ""
) -> List[str]:
    """Return list of packages which should be installd via PIP to use aimet with tensorflow"""
    reqs = _get_pip_reqs_for_framework(cuda_version, tf_verson, torch_version, "tf")
    # Delete tensorflow with pinned version
    reqs = filter(
        lambda r: not (r.startswith("tensorflow-cpu==") or r.startswith("tensorflow-gpu==")), reqs
    )
    reqs = list(reqs) + [f"tensorflow-{'cpu' if cuda_version == 'cpu' else 'gpu'}=={tf_verson}"]
    return reqs


def get_reqs_pip(
    cuda_version: str = "cpu", tf_verson: str = "", torch_version: str = ""
) -> List[str]:
    """Return list of packages which should be installd via PIP to use aimet"""
    return (
        (_get_deps_dir(cuda_version, tf_verson, torch_version) / "reqs_pip_common.txt")
        .read_text(encoding="utf8")
        .splitlines()
    )


def get_reqs_pip_dev(
    cuda_version: str = "cpu", tf_verson: str = "", torch_version: str = ""
) -> List[str]:
    """Return list of packages which should be installd via PIP to develop aimet"""
    return get_reqs_pip(cuda_version, tf_verson, torch_version)


def get_reqs_apt(
    cuda_version: str = "cpu", tf_verson: str = "", torch_version: str = ""
) -> List[str]:
    """Return list of packages which should be installd via APT to use aimet"""
    return (
        (_get_deps_dir(cuda_version, tf_verson, torch_version) / "reqs_deb_common.txt")
        .read_text(encoding="utf8")
        .splitlines()
    )


def get_reqs_apt_dev(
    cuda_version: str = "cpu", tf_verson: str = "", torch_version: str = ""
) -> List[str]:
    """Return list of packages which should be installd via APT to develop aimet"""
    return get_reqs_apt(cuda_version, tf_verson, torch_version)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Get AIMET dependencies for a package manager. ")
    parser.add_argument("pkg_mgr", choices=["apt", "pip"], help="A package manager")
    parser.add_argument("--dev", action="store_true", help="Include dev dependencies")
    parser.add_argument(
        "--cuda",
        default=os.environ.get("AIMET_CU_VER", "cpu"),
        help="Cuda version, by default will use AIMET_CU_VER environment variable or 'cpu'.",
    )
    parser.add_argument(
        "--tensorflow",
        default=os.environ.get("AIMET_TF_VER", ""),
        help="Tensorflow version, by default will use AIMET_TF_VER environment variable or ''.",
    )
    parser.add_argument(
        "--torch",
        default=os.environ.get("AIMET_PT_VER", ""),
        help="Torch version, by default will use AIMET_PT_VER environment variable or ''.",
    )
    return parser


def get_dependencies(argv: Optional[List[str]] = None) -> None:
    """Function calls another function to get list of dependencies based on cli arguments.
    All functions must follow the same naming convention: get_reqs_[apt|pip][_dev|_torch|_tf].
    """
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
