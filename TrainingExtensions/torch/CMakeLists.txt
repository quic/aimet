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

find_package(Python3 COMPONENTS Interpreter)


set(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/src/setup.py")
set(OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp"
      "${CMAKE_CURRENT_BINARY_DIR}/build"
      )
set(DEPS "src/AimetTensorQuantizer.cpp")

get_target_property(DlQuantizationIncludes MoDlQuantization INCLUDE_DIRECTORIES)
string(REPLACE ";" "','" DlQuantizationIncludePaths "${DlQuantizationIncludes}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/setup.py.in ${SETUP_PY})

if (ENABLE_CUDA)
    set(ENABLE_CUDA_PYTORCH 1)
else()
    set(ENABLE_CUDA_PYTORCH 0)
endif()

# Set EXTRA_LINK_ARGS to specify RPATH to not use LD_LIBRARY_PATH at runtime.
# Use an absolute path RPATH because the SO file is located in the build directory. It is not
# possible to use $ORIGIN to point to folder with torch libs.
# Later, when a SO is copied into a wheel, RPATH will be changed to use relative paths ($ORIGIN)
add_custom_command(OUTPUT ${OUTPUT}
      COMMAND ${CMAKE_COMMAND}
              -E env
                  PYTHONPATH="${CMAKE_BINARY_DIR}/artifacts"
                  ENABLE_CUDA_PYTORCH=${ENABLE_CUDA_PYTORCH}
                  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
                  EXTRA_LINK_ARGS="-Wl,-rpath;-Wl,$<TARGET_FILE_DIR:PyModelOptimizations>;-Wl,-rpath;-Wl,${TORCH_DIR}/lib;-Wl,--enable-new-dtags"
              ${Python3_EXECUTABLE} ${SETUP_PY} install  --install-base=${CMAKE_BINARY_DIR}/artifacts
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
    COMMAND ${CMAKE_COMMAND} -E copy ${FILE_PATH} ${CMAKE_BINARY_DIR}/artifacts/aimet_common)

add_dependencies(TorchCppOps PyModelOptimizations)
add_dependencies(BuildTorchExtensions TorchCppOps)

add_subdirectory(src/python)

if (ENABLE_ONNX)
    message(STATUS "ONNX is enabled, skipping Torch Unit test")
else(ENABLE_ONNX)
    if (ENABLE_TESTS)
        add_subdirectory(test)
    endif()
endif(ENABLE_ONNX)

install(DIRECTORY ${CMAKE_BINARY_DIR}/artifacts/aimet_common/
      DESTINATION ${AIMET_INSTALL_DIR}/lib/python/aimet_common
      FILES_MATCHING PATTERN "AimetTensorQuantizer*.so"
      )

add_custom_target(whl_prep_cp_torch
      COMMAND ${CMAKE_COMMAND} -E make_directory ${WHL_PREP_AIMET_TORCH_DIR}
      COMMAND find ${CMAKE_BINARY_DIR}/artifacts -path \"*/aimet_tensor_quantizer*\" -name \"AimetTensorQuantizer*.so\"
              -exec sh -c '$$0 -f $$1 ${WHL_PREP_AIMET_TORCH_DIR}/$$\(basename $$1\)' cp {} \\\;
      COMMAND find ${WHL_PREP_AIMET_TORCH_DIR} -name 'AimetTensorQuantizer*.so'
              -exec ${PATCHELF_EXE} --set-rpath '$$ORIGIN:$$ORIGIN/../torch/lib' {} \\\;
      DEPENDS
          TorchCppOps
)
add_custom_target(whl_prep_ln_torch
      COMMAND ${CMAKE_COMMAND} -E make_directory ${WHL_PREP_AIMET_TORCH_DIR}
      COMMAND find ${CMAKE_BINARY_DIR}/artifacts -path \"*/aimet_tensor_quantizer*\" -name \"AimetTensorQuantizer*.so\"
              -exec sh -c '$$0 -f $$1 ${WHL_PREP_AIMET_TORCH_DIR}/$$\(basename $$1\)' "ln -s" {} \\\;
      DEPENDS
          TorchCppOps
)

add_dependencies(whl_prep_cp whl_prep_cp_torch)
add_dependencies(whl_prep_ln whl_prep_ln_torch)
whl_add_whl_action_target(torch)
