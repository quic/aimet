# =============================================================================
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
# =============================================================================

# CMake file to upload the AIMET wheel packages

set(build_packaging_dir "${CMAKE_BINARY_DIR}/packaging")
file(GLOB wheel_files "${build_packaging_dir}/dist/*.whl")
message(STATUS "*** Found wheel files: ${wheel_files}")

# Check the pip config file and set it to a default value (if not set)
if(PIP_CONFIG_FILE STREQUAL "None")
    set(PIP_CONFIG_FILE "~/.pypirc")
    message(WARNING "PIP_CONFIG_FILE was not specified. Setting it to ${PIP_CONFIG_FILE}.")
else()
    message(STATUS "PIP_CONFIG_FILE already set to ${PIP_CONFIG_FILE}.")
endif()

# Check whether the pip index was specified (must be present within the pip config file)
if(PIP_INDEX STREQUAL "None")
    message(FATAL_ERROR "PIP_INDEX was not set. Please cmake -DPIP_INDEX=<pip_index_value>.")
endif()

# Set the pip package upload command argument string
set(pip_upload_cmd_args " upload --verbose --repository ${PIP_INDEX} --config-file ${PIP_CONFIG_FILE} ")

# Check the certificate path and append to the command root string if present
if(PIP_CERT_FILE STREQUAL "None")
    message(WARNING "PIP_CERT_FILE was not specified. Not using that option with twine command.")
else()
    set(pip_upload_cmd_args "${pip_upload_cmd_args} --cert ${PIP_CERT_FILE}")
endif()

# Loop over the package array list to select the wheel files to be uploaded
foreach(wheel_file ${wheel_files})
    # Pre-pend the twine command and add the wheel file to be uploaded at the end
    set(pip_upload_cmd twine "${pip_upload_cmd_args} ${wheel_file}")
    message(STATUS "Package upload command: ${pip_upload_cmd}")

    # execute the command to upload the wheel files.
    execute_process(COMMAND ${pip_upload_cmd} WORKING_DIRECTORY ${build_packaging_dir} OUTPUT_VARIABLE output_var ERROR_VARIABLE error_var RESULT_VARIABLE result_var)
    if(result_var EQUAL "1")
        message( FATAL_ERROR "twine upload failed")
    endif()

    message(WARNING "Package upload MAY not have completed. Please check destination and upload manually if needed.")
endforeach()
