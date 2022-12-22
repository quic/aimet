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

""" Package generation file for aimet common package """

import os
import sys
from setuptools import setup, find_packages, find_namespace_packages
from packaging_common import bdist_wheel_aimet
import setup_cfg # pylint: disable=import-error

package_name = "aimet_common"
package_bin_dir = package_name + "/bin"
package_url_base = setup_cfg.remote_url + "/releases/download/"+str(setup_cfg.version)

# Obtain package contents; exclude build and other files
packages_found = find_packages() + \
    find_namespace_packages(exclude=['pyenv3*', 'build', 'dist', '*bin', '*x86*', '*aimet_tensorflow*', '*aimet_torch*'])

# Create common dependency list
install_requires_list = [open(package_bin_dir + '/reqs_pip_common.txt').read()]
package_dependency_files = ['reqs_pip_common.txt', 'reqs_deb_common.txt', 'INSTALL.txt', 'envsetup.sh', 'LICENSE.pdf']
if "--gpu" in sys.argv:
    # There is NO common GPU dependency list, so just ignore the option if it was passed in
    sys.argv.remove("--gpu")

# Loop over package artifacts folder
required_package_data = []
for path, _, filenames in os.walk(package_name):
    # Create package contents' list
    required_package_data += [os.path.join(path, filename) for filename in filenames if
                              filename.endswith('.json') or
                              filename.endswith('.so') or
                              path.startswith(package_name + '/x86_64-linux-gnu') or
                              filename.endswith(tuple(package_dependency_files))]
required_package_data = ['/'.join(files.split('/')[1:]) for files in required_package_data]
#TODO For some reason, we need to explicitly add HTML files from subfolders like this
required_package_data += ['*/*.html']

setup(
    name='AimetCommon',
    version=str(setup_cfg.version),
    author='Qualcomm Innovation Center, Inc.',
    author_email='aimet.os@quicinc.com',
    packages=packages_found,
    url=package_url_base,
    license='NOTICE.txt',
    description='AIMET Common Package',
    long_description=open('README.txt').read(),
    package_data={package_name:required_package_data},
    install_requires=install_requires_list,
    zip_safe=True,
    platforms='x86',
    python_requires='>=3.6',
    cmdclass={
        'bdist_wheel': bdist_wheel_aimet,
    },
)
