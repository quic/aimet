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

""" Common tools for Package generation """

import os
import sys
import setup_cfg # pylint: disable=import-error

from pathlib import Path
from setuptools import find_packages, find_namespace_packages
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


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
    ''' Search for and return the dependency wheel file '''
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
    full_path = os.path.join(package_bin_dir, dependency_list_file)

    return full_path


def get_dependency_packages(dependency_list_file, aimet_package_name = None):
    '''
    Read the dependency file and return a list of packages
    '''

    if aimet_package_name is None:
        dependency_list_file_fullpath = dependency_list_file
    else:
        dependency_list_file_fullpath = prepend_bin_path(aimet_package_name, dependency_list_file)

    dependency_list_array = open(dependency_list_file_fullpath).read().splitlines()

    dependency_packages_list = []
    for dependency_line in dependency_list_array:
        if dependency_line.strip().startswith('#'):
            # Skip the commented out lines
            continue
        # Pick up the package and version (<package>==<ver> i.e. first column of file)
        dependency_packages_list.append(dependency_line.lstrip().split()[0])

    return dependency_packages_list


def get_dependency_urls(dependency_file_full_path):
    '''
    Read the dependency file and return a list of package source URLs
    '''

    url_delimiter = '-f '

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


def get_package_list(aimet_package_name):
    '''
    Obtain package contents.
    Exclude build and certain other files including those from other packages
    '''

    # Exclude build and other unnecessary files / folders
    exclusion_list = ['pyenv3*', 'build', 'dist', '*bin', '*x86*']
    # Exclude all aimet package folders MINUS the package in question
    # NOTE: aimet_common will always be included
    exclusion_list += ['*aimet_tensorflow*', '*aimet_torch*', '*aimet_onnx*']
    # Remove package currently being created from exclusion list
    exclusion_list.remove("*" + aimet_package_name + "*")

    packages_found = find_packages() + find_namespace_packages(exclude=exclusion_list)

    return packages_found


def get_pip_dep_files(aimet_package_name):
    ''' Get list of pip dependency list files for a given aimet package name '''

    package_bin_dir = aimet_package_name + "/bin"

    pip_deps_reqs_list = []
    for filename in Path(package_bin_dir).glob("reqs_pip*.txt"):
        pip_deps_reqs_list.append(filename)

    return pip_deps_reqs_list


def get_all_dep_files(aimet_package_name):
    ''' Get list of ALL dependency files for a given aimet package name '''

    package_bin_dir = aimet_package_name + "/bin"

    package_deps_list = []
    for filename in Path(package_bin_dir).glob("*.*"):
        package_deps_list.append(filename)

    return package_deps_list


def get_pip_dep_packages_list(aimet_package_name):
    ''' Get list of python dependency packages for a given aimet package '''

    reqs_pip_dep_files = get_pip_dep_files(aimet_package_name)

    pip_dep_packages_list = []
    for reqs_pip_dep_file in reqs_pip_dep_files:
        pip_dep_list_single = get_dependency_packages(reqs_pip_dep_file)
        pip_dep_packages_list.extend(pip_dep_list_single)

    return pip_dep_packages_list


def get_required_package_data(aimet_package_name):
    ''' Get list of python dependency packages for a given aimet package '''

    # Obtain a list of app dependency files with full paths
    package_dependency_files_posixpath = get_all_dep_files(aimet_package_name)
    # Convert list from Posix paths to strings
    package_dependency_files = [str(pkg_dep_file) for pkg_dep_file in package_dependency_files_posixpath]

    # Loop over package artifacts folder and create package contents' list
    required_package_data = ['acceptance_tests/*.*']
    for path, _, filenames in os.walk(aimet_package_name):        
        # Loop over each individual asset
        for filename in filenames:
            filename_with_path = os.path.join(path, filename)
            if filename.endswith('.json') or \
                filename.endswith('.so') or \
                path.startswith(aimet_package_name + '/x86_64-linux-gnu') or \
                filename_with_path in package_dependency_files:
                
                required_package_data.append(filename_with_path)

    required_package_data = ['/'.join(files.split('/')[1:]) for files in required_package_data]

    #TODO For some reason, we need to explicitly add HTML and XML files from subfolders like this
    required_package_data += ['*/*/*.html']
    required_package_data += ['*/*.xml']

    return required_package_data


def get_all_dependency_urls(aimet_package_name):
    '''
    Read ALL dependency files and return the complete list of package source URLs
    '''

    # Get the list of pip dependency list files
    reqs_pip_dep_files = get_pip_dep_files(aimet_package_name)

    # Loop over the list and build a URL list
    dependency_url_list = []
    for reqs_pip_dep_file in reqs_pip_dep_files:
        dependency_url_list.extend(get_dependency_urls(reqs_pip_dep_file))

    return dependency_url_list
