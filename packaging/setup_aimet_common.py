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

""" Package generation file for aimet common package """

import os
import setup_cfg # pylint: disable=import-error
from setuptools import setup, find_packages, find_namespace_packages

package_url_base = setup_cfg.remote_url + "/releases/download/"+str(setup_cfg.version)

# get all packages , discard build files etc.
packages_found = find_packages() + find_namespace_packages(exclude=['pyenv3*', 'build', 'dist', '*bin', '*x86*', '*aimet_tensorflow*', '*aimet_torch*'])

required_package_data=[]

package_dependency_files = ['requirements.txt', 'packages_common.txt', 'packages_gpu.txt', 'INSTALL.txt', 'envsetup.sh', 'LICENSE.pdf']

for path, _, filenames in os.walk('aimet_common'):
    required_package_data += [os.path.join(path, filename) for filename in filenames if 
    filename.endswith('.json') or path.startswith('aimet_common/x86_64-linux-gnu') or 
    filename.endswith(tuple(package_dependency_files))]
required_package_data = ['/'.join(files.split('/')[1:]) for files in required_package_data]

setup(
    name='AimetCommon',
    version=str(setup_cfg.version),
    author='Qualcomm Innovation Center, Inc.',
    author_email='aimet@noreply.github.com',
    packages=packages_found,
    url=package_url_base,
    license='NOTICE.txt',
    description='AIMET',
    long_description=open('README.txt').read(),
    package_data={'aimet_common':required_package_data},
    install_requires=[open('requirements.txt').read()],
    zip_safe=True,
    platforms='x86',
    python_requires='>=3.6',
)
