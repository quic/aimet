# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
"""Creates reStructuredText file containing text, code, and outputs from properly annotated python script

'Proper' annotations use a similar style to sphinx-gallery annotated python files, following these rules:

1) File starts with a docstring containing reStructuredText
2) All subsequent reStructuredText segments begin with 20 or more `#` characters followed by the exact rst in comments

Args:
    --source: .py file to run and convert
    --output: .rst file output

"""


import io
import re
import argparse
import contextlib

# Lazy regex matching anything inside first set of double """ or '''
DOCSTRING_PATTERN = re.compile('("""(.|\n)*?""")|(\'\'\'(.|\n)*?\'\'\')')

# 20 or more '#' symbols followed by any number of '# <rst text>` lines
RST_TEXT_PATTERN = re.compile("#" * 20 + ".*" + "\n" "((#.*)\n)*")

START_CODE = "\n.. code-block:: Python\n\n"
START_OUTPUT = "\n\n.. rst-class:: script-output\n\n  .. code-block:: none\n\n"

CODE_INDENT = " " * 4
OUTPUT_INDENT = " " * 6


namespace = {}


def process_suite(suite):
    suite = suite.strip()
    if not suite:
        return ""

    # Send output of suite to string
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(suite, namespace)

    output = output.getvalue()

    suite_lines = [f"{CODE_INDENT}{line}" for line in suite.split("\n")]
    all_lines = [START_CODE] + suite_lines

    if output:
        output_lines = [f"{OUTPUT_INDENT}{line}" for line in output.split("\n")]
        all_lines += [START_OUTPUT] + output_lines

    return "\n".join(all_lines) + "\n\n\n"


def process_rst(rst):

    def remove_pound(string):
        return string[1:] if len(string) == 1 else string[2:]

    # Remove the rst indicator line
    rst_lines = rst.split("\n")[1:]

    return "\n".join(remove_pound(line) for line in rst_lines) + "\n\n"



def convert(args):

    output_str = ""

    with open(args.source, "r") as source:
        source_code = source.read()

    docstring = re.search(DOCSTRING_PATTERN, source_code)

    if docstring:
        output_str += docstring.group().strip("'\"")
        source_code = source_code[docstring.span()[1]:]

    while source_code:
        next_text = re.search(RST_TEXT_PATTERN, source_code)

        if next_text:
            suite = source_code[:next_text.span()[0]]
            rst_text = next_text.group()
            source_code = source_code[next_text.span()[1]:]
        else:
            suite = source_code
            rst_text = ""
            source_code = ""

        suite_processed = process_suite(suite)
        rst_processed = process_rst(rst_text)

        output_str += suite_processed + rst_processed

    with open(args.output, "w") as out_file:
        out_file.write(output_str)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--output")

    args = parser.parse_args()
    convert(args)
