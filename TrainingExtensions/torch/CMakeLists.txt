#==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018, Qualcomm Innovation Center, Inc. All rights reserved.
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
cmake_minimum_required(VERSION 3.5)

find_program(PYTHON "python3")


set(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/src/setup.py")
set(OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp"
      "${CMAKE_CURRENT_BINARY_DIR}/build"
      )
set(DEPS "src/AimetTensorQuantizer.cpp")

get_target_property(DlQuantizationIncludes MoDlQuantization INCLUDE_DIRECTORIES)
string(REPLACE ";" "','" DlQuantizationIncludePaths "${DlQuantizationIncludes}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/setup.py.in ${SETUP_PY})

add_custom_command(OUTPUT ${OUTPUT}
      COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH="${CMAKE_BINARY_DIR}/artifacts"
              ${PYTHON} ${SETUP_PY} install  --install-base=${CMAKE_BINARY_DIR}/artifacts
                                             --install-purelib=${CMAKE_BINARY_DIR}/artifacts
                                             --install-platlib=${CMAKE_BINARY_DIR}/artifacts
                                             --install-scripts=${CMAKE_BINARY_DIR}/artifacts
                                             --install-data=${CMAKE_BINARY_DIR}/artifacts
                                             --install-headers=${CMAKE_BINARY_DIR}/artifacts
      COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
      DEPENDS ${DEPS}
      )

add_custom_target(TorchCppOps ALL
      DEPENDS ${OUTPUT}
      SOURCES
      src/AimetTensorQuantizer.cpp
      )

set(FILE_PATH "${CMAKE_BINARY_DIR}/artifacts/aimet_tensor_quantizer*.egg/AimetTensorQuantizer*.so")

add_custom_target(BuildTorchExtensions ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/artifacts/aimet_torch
    COMMAND ${CMAKE_COMMAND} -E copy ${FILE_PATH} ${CMAKE_BINARY_DIR}/artifacts/aimet_torch)

add_dependencies(TorchCppOps PyModelOptimizations)
add_dependencies(BuildTorchExtensions TorchCppOps)

add_subdirectory(src/python)
add_subdirectory(test)

install(DIRECTORY ${CMAKE_BINARY_DIR}/artifacts/aimet_torch/
      DESTINATION ${AIMET_INSTALL_DIR}/lib/python/aimet_torch
      FILES_MATCHING PATTERN "AimetTensorQuantizer*.so"
      )
