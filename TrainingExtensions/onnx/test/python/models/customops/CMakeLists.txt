#==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

find_package(Torch REQUIRED)
set(CMAKE_CXX_STANDARD 14)

add_library(torch_custom_add SHARED torch_custom_add.cpp)
target_compile_features(torch_custom_add PRIVATE cxx_std_14)
target_link_libraries(torch_custom_add "${TORCH_LIBRARIES}")
set_target_properties(torch_custom_add PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts/aimet_common/customops"
        )

# This block attempts to find the version number of the installed onnxruntime_headers
find_path(ONNXRUNTIME_VER "VERSION_NUMBER" PATH_SUFFIXES onnxruntime_headers)
file(READ ${ONNXRUNTIME_VER}/VERSION_NUMBER VERSION_NUMBER)
string(REGEX REPLACE "([^0123456789.])" "" VERSION_NUMBER ${VERSION_NUMBER})
message(STATUS "Found onnxruntime_header version: ${VERSION_NUMBER}")

# Only build custom op targets if onnxruntime_headers are from >= v1.15.1
if(VERSION_NUMBER VERSION_GREATER_EQUAL "1.15.1")
    find_path(ONNXRUNTIME_INC "onnxruntime_cxx_api.h" PATH_SUFFIXES onnxruntime_headers/include)
    find_library(ONNXRUNTIME_LIBRARY NAMES libonnxruntime.so PATH_SUFFIXES /onnxruntime_headers/lib)

    add_library(onnx_custom_add SHARED onnx_custom_add.cpp onnx_custom_add.h)
    target_compile_features(onnx_custom_add PRIVATE cxx_std_14)
    target_include_directories(onnx_custom_add PRIVATE ${ONNXRUNTIME_INC})
    target_link_libraries(onnx_custom_add PUBLIC ${ONNXRUNTIME_LIBRARY})
    set_target_properties(onnx_custom_add PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts/aimet_common/customops"
            )
else()
    message("onnxruntime headers must be >=v1.15.1 to build onnx customop targets. Skipping onnx custom op targets.")
endif()
