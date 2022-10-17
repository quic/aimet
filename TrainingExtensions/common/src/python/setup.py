#==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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

import os
import shlex
import subprocess
import sys
from pathlib import Path
from shutil import copy

from setuptools import Distribution, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext

CURRENT_DIR = Path(__file__).parent.resolve()
PACKAGING_DIR = CURRENT_DIR / ".." / ".." / ".." / ".." / "packaging"

def str2bool(str_):
    TRUE_VALS = {"true", "yes", "y", "on", "1"}
    FALSE_VALS = {"false", "no", "n", "off", "0"}
    if str_.lower() in TRUE_VALS:
        return True
    elif str_.lower() in FALSE_VALS:
        return False
    else:
        raise RuntimeError(f"Unknown boolean value '{str_}' (known values are: {TRUE_VALS | FALSE_VALS}")

ENABLE_CUDA = str2bool(os.environ.get("ENABLE_CUDA", "False"))
ENABLE_TORCH = str2bool(os.environ.get("ENABLE_TORCH", "True"))
ENABLE_TENSORFLOW = str2bool(os.environ.get("ENABLE_TENSORFLOW", "True"))

PKG_FILES = [
    "dependencies/reqs_pip_common.txt",
    "dependencies/reqs_deb_common.txt",
    "INSTALL.txt",
    "envsetup.sh",
    # "LICENSE.pdf",
]

AIMET_COMMON_VERSION = os.environ.get("SW_VERSION")
if AIMET_COMMON_VERSION is None:
    AIMET_COMMON_VERSION = (PACKAGING_DIR / "version.txt").read_text().strip()

AIMET_COMMON_URL = subprocess.run(
        shlex.split("git config --get remote.origin.url"), check=True,
        cwd=CURRENT_DIR, stdout=subprocess.PIPE, encoding="utf8",
    ).stdout + f"/releases/download/{AIMET_COMMON_VERSION}"

class BuildExtensionCommand(build_ext):
    def run(self):
        super().run()
        # Create dest directories
        dst_dir = Path(self.get_ext_fullpath("dummy")).parent / "aimet_common"
        dst_dir.mkdir(parents=True, exist_ok=True)
        (dst_dir / "bin").mkdir(parents=True, exist_ok=True)

        # Check if prebuilt C++ part is available
        whl_prep_dir = os.environ.get("WHL_PREP_DIR")
        if whl_prep_dir is None:
            src_dir = CURRENT_DIR
            bld_dir = Path(self.build_temp).resolve()
            whl_prep_dir = bld_dir / "whlprep"
            cmake_args = [
                f"-DPython3_ROOT_DIR={os.path.dirname(sys.executable)}",
                f"-DWHL_PREP_DIR={whl_prep_dir}",
                f"-DENABLE_CUDA={ENABLE_CUDA}",
                f"-DENABLE_TORCH={ENABLE_TORCH}",
                f"-DENABLE_TENSORFLOW={ENABLE_TENSORFLOW}",
                f"-DWHL_EDITABLE_MODE={self.inplace}",
            ]
            subprocess.run(["cmake", "-B", bld_dir, "-S",  src_dir] + cmake_args,
                check=True, stdout=sys.stdout, stderr=sys.stderr, encoding="utf8",
                )
            subprocess.run(["cmake", "--build",  bld_dir, "-j", "-t", "whl_prep_aimet_common"],
                check=True, stdout=sys.stdout, stderr=sys.stderr, encoding="utf8",
                )
            if ENABLE_TORCH:
                subprocess.run(
                    ["cmake", "--build",  bld_dir, "-j", "-t", "whl_prep_aimet_torch"],
                    check=True, stdout=sys.stdout, stderr=sys.stderr, encoding="utf8",
                    )
            if ENABLE_TENSORFLOW:
                subprocess.run(
                    ["cmake", "--build",  bld_dir, "-j", "-t", "whl_prep_aimet_tensorflow"],
                    check=True, stdout=sys.stdout, stderr=sys.stderr, encoding="utf8",
                    )
        # Copy C++ part into wheel package
        subprocess.run(
            shlex.split(f"cp -Prv {whl_prep_dir}/aimet_common/. {dst_dir}"),
            check=True, stdout=sys.stdout, stderr=sys.stderr, encoding="utf8",
            )
        if ENABLE_TORCH:
            subprocess.run(
                shlex.split(f"cp -Prv {whl_prep_dir}/aimet_torch/. {dst_dir}"),
                check=True, stdout=sys.stdout, stderr=sys.stderr, encoding="utf8",
                )
        if ENABLE_TENSORFLOW:
            subprocess.run(
                shlex.split(f"cp -Prv {whl_prep_dir}/aimet_tensorflow/. {dst_dir}"),
                check=True, stdout=sys.stdout, stderr=sys.stderr, encoding="utf8",
                )
        subprocess.run(
            shlex.split(f"cp -Lrv {' '.join(str(PACKAGING_DIR / f) for f in PKG_FILES)} {dst_dir / 'bin'}"),
            check=True, stdout=sys.stdout, stderr=sys.stderr, encoding="utf8",
            )



class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(
    author_email="aimet.os@quicinc.com",
    author="Qualcomm Innovation Center, Inc.",
    cmdclass={"build_ext": BuildExtensionCommand, },
    description="AIMET Common Package",
    distclass = BinaryDistribution,
    install_requires=(PACKAGING_DIR / "dependencies"/ "reqs_pip_common.txt").read_text().splitlines(),
    license="NOTICE.txt",
    long_description=(PACKAGING_DIR / "README.txt").read_text(),
    name="AimetCommon",
    package_data={"": ["*.json"], },
    package_dir={"": ".", },
    packages=find_namespace_packages(where=".", exclude=["build", "x86_64-linux-gnu"]),
    platforms="x86",
    python_requires=">=3.6",
    url=AIMET_COMMON_URL,
    version=AIMET_COMMON_VERSION,
)