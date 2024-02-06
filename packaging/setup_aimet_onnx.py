#==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Package generation file for aimet onnx package """

import sys
import setup_cfg # pylint: disable=import-error

from packaging_common import bdist_wheel_aimet, get_package_list, get_pip_dep_packages_list, \
    get_required_package_data, get_all_dependency_urls
from setuptools import setup

package_name = "aimet_onnx"
common_package_name ="aimet_common"
package_url_base = setup_cfg.remote_url + "/releases/download/"+str(setup_cfg.version)


# Obtain list of package and sub-packages (including common dependency ones)
packages_found = get_package_list(package_name)

# Obtain list of dependencies for current package
install_requires_list = get_pip_dep_packages_list(package_name)
# Obtain non-python artifacts for current package
required_package_data = get_required_package_data(package_name)
# Obtain non-PyPi dependency URLs (if any) for current package
dependency_url_list = get_all_dependency_urls(package_name)

# Obtain non-python artifacts for common package
required_common_package_data = get_required_package_data(common_package_name)
# Obtain non-PyPi dependency URLs (if any) for common package
dependency_url_list.extend(get_all_dependency_urls(common_package_name))

if "--gpu" in sys.argv:
    sys.argv.remove("--gpu")


setup(
    name='AimetOnnx',
    version=str(setup_cfg.version),
    author='Qualcomm Innovation Center, Inc.',
    author_email='aimet.os@quicinc.com',
    packages=packages_found,
    url=package_url_base,
    license='NOTICE.txt',
    description='AIMET Onnx Package',
    long_description=open('README.txt').read(),
    package_data={package_name:required_package_data, common_package_name:required_common_package_data},
    install_requires=install_requires_list,
    dependency_links=dependency_url_list,
    include_package_data=True,
    zip_safe=True,
    platforms='x86',
    python_requires='>=3.8',
    cmdclass={
        'bdist_wheel': bdist_wheel_aimet,
    },
)
