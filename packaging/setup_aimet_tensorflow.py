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
import setup_cfg # pylint: disable=import-error

package_name = "aimet_tensorflow"

def prepend_bin_path(dependency_list_file):
    '''
    Append the common path to the dependency file
    '''
    package_bin_dir = package_name + "/bin"
    full_path = package_bin_dir + '/' + dependency_list_file
    return full_path


def get_dependency_packages(dependency_list_file):
    '''
    Read the dependency file and return a list of packages
    '''

    dependency_file_full_path = prepend_bin_path(dependency_list_file)
    dependency_list_array = open(dependency_file_full_path).read().splitlines()

    dependency_packages_list = []
    for dependency_line in dependency_list_array:
        if dependency_line.strip().startswith('#'):
            # Skip the commented out lines
            continue
        # Pick up the package and version (<package>==<ver> i.e. first column of file)
        dependency_packages_list.append(dependency_line.lstrip().split()[0])

    return dependency_packages_list

def get_dependency_urls(dependency_list_file):
    '''
    Read the dependency file and return a list of package source URLs
    '''

    url_delimiter = '-f '

    dependency_file_full_path = prepend_bin_path(dependency_list_file)
    dependency_list_array = open(dependency_file_full_path).read().splitlines()

    dependency_urls_list = []
    for dependency_line in dependency_list_array:
        if dependency_line.strip().startswith('#'):
            # Skip the commented out lines
            continue

        # Extract any package custom URLs from the requirements file
        # Ex. torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
        url_delimiter = '-f '
        # Split each line using the URL option as delimiter
        dependency_line_split_list = dependency_line.split(url_delimiter)
        # The split list will have at least 2 elements if a URL was present
        if len(dependency_line_split_list) > 1:
            # If the URL exists, remove all whitespaces
            dependency_url = dependency_line_split_list[1].strip()
            # Add to the list only if not already present
            if dependency_url not in dependency_urls_list:
                dependency_urls_list.append(dependency_url)

    return dependency_urls_list


package_url_base = setup_cfg.remote_url + "/releases/download/"+str(setup_cfg.version)
common_dep_whl = package_url_base + "/AimetCommon-" + str(setup_cfg.version) + "-py3-none-any.whl"
dependency_url_list = [common_dep_whl]

# Obtain package contents; exclude build and other files
packages_found = find_packages() + find_namespace_packages(exclude=['*bin', 'pyenv3*', 'build', 'dist', '*bin', '*x86*'])

# Create common dependency list
package_dependency_files = ['reqs_pip_tf_common.txt']
install_requires_list = get_dependency_packages('reqs_pip_tf_common.txt')
if "--gpu" in sys.argv:
    # Create Tensorflow GPU dependency list
    package_dependency_files.extend(['bin/reqs_pip_tf_gpu.txt', 'bin/reqs_deb_tf_gpu.txt'])
    install_requires_list.extend(get_dependency_packages('reqs_pip_tf_gpu.txt'))
    dependency_url_list.extend(get_dependency_urls('reqs_pip_tf_gpu.txt'))
    sys.argv.remove("--gpu")
else:
    # Create Tensorflow CPU dependency list
    package_dependency_files.extend(['bin/reqs_pip_tf_cpu.txt'])
    install_requires_list.extend(get_dependency_packages('reqs_pip_tf_cpu.txt'))
    dependency_url_list.extend(get_dependency_urls('reqs_pip_tf_cpu.txt'))

# Loop over package artifacts folder
required_package_data = ['acceptance_tests/*.*']
for path, _, filenames in os.walk(package_name):
    required_package_data += [os.path.join(path, filename) for filename in filenames if
                              filename.endswith(tuple(package_dependency_files))]
required_package_data = ['/'.join(files.split('/')[1:]) for files in required_package_data]


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
)
