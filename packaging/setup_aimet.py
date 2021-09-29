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

""" Package generation file for top-level aimet package """

import sys
import os.path
from setuptools import setup
from packaging_common import bdist_wheel_aimet, get_dependency_wheel
import setup_cfg # pylint: disable=import-error

package_url_base = setup_cfg.remote_url + "/releases/download/" + str(setup_cfg.version)

dependency_list = []
torch_dep_whl = get_dependency_wheel("AimetTorch")
if torch_dep_whl is not None:
    torch_dep_whl_url = package_url_base + "/" + torch_dep_whl
    dependency_list.append(torch_dep_whl_url)

tf_dep_whl = get_dependency_wheel("AimetTensorflow")
if tf_dep_whl is not None:
    tf_dep_whl_url = package_url_base + "/" + tf_dep_whl
    dependency_list.append(tf_dep_whl_url)

if "--gpu" in sys.argv:
    # There is NO common GPU dependency list, so just ignore the option if it was passed in
    sys.argv.remove("--gpu")


setup(
    name='Aimet',
    version=str(setup_cfg.version),
    author='Qualcomm Innovation Center, Inc.',
    author_email='aimet.os@quicinc.com',
    url=package_url_base,
    license='NOTICE.txt',
    description='AIMET',
    long_description=open('README.txt').read(),
    install_requires=[],
    dependency_links=dependency_list,
    zip_safe=True,
    platforms='x86',
    python_requires='>=3.6',
    cmdclass={
        'bdist_wheel': bdist_wheel_aimet,
    },
)
