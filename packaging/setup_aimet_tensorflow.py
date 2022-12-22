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

""" Package generation file for aimet tensorflow package """

import os
import sys
from setuptools import setup, find_packages, find_namespace_packages
from packaging_common import bdist_wheel_aimet, get_dependency_packages, get_dependency_urls, get_dependency_wheel
import setup_cfg # pylint: disable=import-error

package_name = "aimet_tensorflow"
package_url_base = setup_cfg.remote_url + "/releases/download/"+str(setup_cfg.version)

dependency_url_list = []
common_dep_whl = get_dependency_wheel("AimetCommon")
if common_dep_whl is not None:
    common_dep_whl_url = package_url_base + "/" + common_dep_whl
    dependency_url_list.append(common_dep_whl_url)
else:
    sys.exit("Could not find dependency wheel file for package: %s" % package_name)

# Obtain package contents; exclude build and certain other files including those from other packages
packages_found = find_packages() + \
    find_namespace_packages(exclude=['*bin', 'pyenv3*', 'build', 'dist', '*bin', '*x86*', '*aimet_common*', '*aimet_torch*'])

# Create common dependency list
package_dependency_files = ['reqs_pip_tf_common.txt']
install_requires_list = get_dependency_packages(package_name, 'reqs_pip_tf_common.txt')
if "--gpu" in sys.argv:
    # Create Tensorflow GPU dependency list
    package_dependency_files.extend(['reqs_pip_tf_gpu.txt', 'reqs_deb_tf_gpu.txt'])
    install_requires_list.extend(get_dependency_packages(package_name, 'reqs_pip_tf_gpu.txt'))
    dependency_url_list.extend(get_dependency_urls(package_name, 'reqs_pip_tf_gpu.txt'))
    sys.argv.remove("--gpu")
else:
    # Create Tensorflow CPU dependency list
    package_dependency_files.extend(['reqs_pip_tf_cpu.txt'])
    install_requires_list.extend(get_dependency_packages(package_name, 'reqs_pip_tf_cpu.txt'))
    dependency_url_list.extend(get_dependency_urls(package_name, 'reqs_pip_tf_cpu.txt'))

# Loop over package artifacts folder
required_package_data = ['acceptance_tests/*.*']
for path, _, filenames in os.walk(package_name):
    required_package_data += [os.path.join(path, filename) for filename in filenames if
                              filename.endswith(tuple(package_dependency_files))]

required_package_data = ['/'.join(files.split('/')[1:]) for files in required_package_data]
#TODO For some reason, we need to explicitly add HTML files from subfolders like this
required_package_data += ['*/*.html']

setup(
    name='AimetTensorflow',
    version=str(setup_cfg.version),
    author='Qualcomm Innovation Center, Inc.',
    author_email='aimet.os@quicinc.com',
    packages=packages_found,
    url=package_url_base,
    license='NOTICE.txt',
    description='AIMET TensorFlow Package',
    long_description=open('README.txt').read(),
    package_data={package_name:required_package_data},
    install_requires=install_requires_list,
    dependency_links=dependency_url_list,
    include_package_data=True,
    zip_safe=True,
    platforms='x86',
    python_requires='>=3.6',
    cmdclass={
        'bdist_wheel': bdist_wheel_aimet,
    },
)
