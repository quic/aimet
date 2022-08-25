#!/usr/bin/python3

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

# move xml files generated from acceptance tests to target directory
# so that XUnitBuilder plugin from Jenkins can find them

import sys
import os
import shutil

search_root = os.path.join(sys.argv[1],"build","NightlyTests")
acceptance_test_dir = os.path.join(sys.argv[1],"acceptance_test_results")
shutil.rmtree(acceptance_test_dir, ignore_errors=True)

if not os.path.exists(acceptance_test_dir):
    os.makedirs(acceptance_test_dir)

for dirpath, dirs, files in os.walk(search_root, onerror=None, followlinks=False):
    output_files = []
    output_files = [f for f in files if os.path.splitext(f)[1] == ".xml"]

    for output_file in output_files:
        src_file = os.path.abspath(os.path.join(dirpath, output_file))
        dst_file = os.path.join(acceptance_test_dir, src_file.replace('/', '_'))
        shutil.copy2(src_file, dst_file)

        if not os.path.exists(dst_file):
            print("Copying Acceptance test results to report directory Failed. Destination path %s does not exist."
                         % dst_file)

