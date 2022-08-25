#!/usr/bin/python

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

# move xml files generated from unit tests to target directory
# so that XUnitBuilder plugin from Jenkins can find them


import sys
import os
import shutil

search_root = os.path.join(sys.argv[1],"build")
unit_test_dir = os.path.join(sys.argv[1],"unit_test_results")
shutil.rmtree(unit_test_dir, ignore_errors=True)

if not os.path.exists(unit_test_dir):
    os.makedirs(unit_test_dir)

for dirpath, dirs, files in os.walk(search_root, onerror=None, followlinks=False):
    output_file = None

    if "py_test_output.xml" in files:
        output_file = "py_test_output.xml"
    elif "cpp_test_output.xml" in files:
        output_file = "cpp_test_output.xml"

    if output_file is not None:
        src_file = os.path.abspath(os.path.join(dirpath, output_file))
        dst_file = os.path.join(unit_test_dir, src_file.replace('/', '_'))
        shutil.copy2(src_file, dst_file)

        if not os.path.exists(dst_file):
            print("Copying Unit test results to report directory Failed. Destination path %s does not exist."
                         % dst_file)

