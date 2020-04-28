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

"""
Utility script to strip copyright notice text from AIMET files
Expects all files passed to it to have @@-COPYRIGHT-START-@@ and @@-COPYRIGHT-END-@@ markers
Changes the files in-place
Intent is to invoke it something similar to: ag -l "@@-COPYRIGHT-START-@@" | xargs -n 1 copyright-strip.py
"""

import sys
import os
from enum import IntEnum

copyright_start_text = '@@-COPYRIGHT-START-@@'
copyright_end_text = '@@-COPYRIGHT-END-@@'


class State(IntEnum):
    BEFORE_COPYRIGHT_START = 1
    IN_COPYRIGHT_BLOCK = 2
    AFTER_COPYRIGHT_END = 3


if __name__ == '__main__':
    if len(sys.argv) != 2:
        script_name = os.path.basename(sys.argv[0])
        print('Usage: {} <name of file>'.format(script_name))
        sys.exit(1)

    filename = sys.argv[1]

    with open(filename, "r") as file:
        lines = file.readlines()

    with open(filename, "w") as new_file:

        state = State(State.BEFORE_COPYRIGHT_START)
        for line in lines:

            if state == State.BEFORE_COPYRIGHT_START:
                if copyright_start_text in line:
                    state = State.IN_COPYRIGHT_BLOCK

                new_file.write(line)

            elif state == State.IN_COPYRIGHT_BLOCK:
                if copyright_end_text in line:
                    state = State.AFTER_COPYRIGHT_END
                    new_file.write(line)

            elif state == State.AFTER_COPYRIGHT_END:
                new_file.write(line)

            else:
                raise NotImplementedError
