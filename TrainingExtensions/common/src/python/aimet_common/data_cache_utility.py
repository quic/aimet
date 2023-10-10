# -*- mode: python -*-
# =============================================================================
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
# =============================================================================

"""Utility functions for test/train data cache implementation"""

import os
import shutil
from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


def is_cache_env_set():

    """
    check if Cache Environment variable is set or not
    :param: None
    :return: TRUE in case DEPENDENCY_DATA_PATH environment variable is set, False otherwise
    """
    return 'DEPENDENCY_DATA_PATH' in os.environ

def is_mnist_cache_present():

    """
    check if MNIST data is present in the cache
    :param: None
    :return: None
    DEPENDENCY_DATA_PATH is an environment variable configured by the user,
    in case, user wants to reuse the MNIST data rather than
    downloading the data again.
    """
    mnist_cache_present = False

    if 'DEPENDENCY_DATA_PATH' in os.environ:
        logger.info("Dependency data path was set to %s", os.environ.get('DEPENDENCY_DATA_PATH'))
        mnist_cache_folder = os.path.join(os.environ.get('DEPENDENCY_DATA_PATH'), 'MNIST', 'processed', 'training.pt')
        mnist_cache_present = os.path.exists(mnist_cache_folder)
        logger.info("mnist_cache_folder = %s, mnist_cache_present = %s", mnist_cache_folder, mnist_cache_present)
    else:
        logger.info("Dependency data path env variable was NOT set")

    return mnist_cache_present

def copy_mnist_to_cache():

    """
    Copy MNIST data is present in the cache copy to build folder
    :param: None
    :return: None
    """

    mnist_cache_folder = os.path.join(os.environ.get('DEPENDENCY_DATA_PATH'), 'MNIST')

    logger.info("Copying MNIST to Cache location")
    src_mnist_dir = '../data/MNIST'
    shutil.copytree(src_mnist_dir, mnist_cache_folder)


def copy_cache_mnist_to_local_build():

    """
    if MNIST data is present in the cache, copy the data locally in build folder
    :param: None
    :return: None

    DEPENDENCY_DATA_PATH is an environment variable configured by the user,
    in case, user wants to reuse the MNIST data rather than
    downloading the data again.
    """

    if is_cache_env_set():
        mnist_cache_folder = os.path.join(os.environ.get('DEPENDENCY_DATA_PATH'), 'MNIST')
        dst_dir = '../data/MNIST'
        logger.info("Downloading MNIST data from %s to %s if needed", os.environ.get('DEPENDENCY_DATA_PATH'), dst_dir)

        # Verify whether the data already exists (check existance of one known file)
        if not os.path.exists(os.path.join(dst_dir, 'processed', 'training.pt')):
            logger.info("MNIST Cache is set but data is not present, copying to local build")
            shutil.rmtree(dst_dir, ignore_errors=True)
            shutil.copytree(mnist_cache_folder, dst_dir)

    else:
        logger.info("Dependency data path env variable was NOT set")
