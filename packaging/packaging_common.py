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

""" Common tools for Package generation """

import os
import sys
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
import setup_cfg # pylint: disable=import-error


class bdist_wheel_aimet(_bdist_wheel):
    '''
    Override the wheel package distribution class to create a "platform" wheel
    (with a specific version of python, ABI, and supported platform architecture).
    '''
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        # Mark this as NOT a pure python package (i.e. as a platform wheel package)
        self.root_is_pure = False


def get_dependency_wheel(wheel_starts_with):
    '''
    '''
    for _, _, file_list in os.walk("."):
        for file in file_list:
            if file.startswith(wheel_starts_with + "-" + str(setup_cfg.version)) and file.endswith("whl"):
                return file

    return "None"


def prepend_bin_path(package_name, dependency_list_file):
    '''
    Append the common path to the dependency file
    '''
    package_bin_dir = package_name + "/bin"
    full_path = package_bin_dir + '/' + dependency_list_file
    return full_path


def get_dependency_packages(package_name, dependency_list_file):
    '''
    Read the dependency file and return a list of packages
    '''

    dependency_file_full_path = prepend_bin_path(package_name, dependency_list_file)
    dependency_list_array = open(dependency_file_full_path).read().splitlines()

    dependency_packages_list = []
    for dependency_line in dependency_list_array:
        if dependency_line.strip().startswith('#'):
            # Skip the commented out lines
            continue
        # Pick up the package and version (<package>==<ver> i.e. first column of file)
        dependency_packages_list.append(dependency_line.lstrip().split()[0])

    return dependency_packages_list


def get_dependency_urls(package_name, dependency_list_file):
    '''
    Read the dependency file and return a list of package source URLs
    '''

    url_delimiter = '-f '

    dependency_file_full_path = prepend_bin_path(package_name, dependency_list_file)
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
