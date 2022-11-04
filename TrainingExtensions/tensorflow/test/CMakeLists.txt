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

if (ENABLE_CUDA)
    set(CUDA_FLAG "not blah")
else (ENABLE_CUDA)
    set(CUDA_FLAG "not cuda")
endif (ENABLE_CUDA)

if (TF_VERSION MATCHES "^1.15")
    set(TF_VERSION_FLAG "not tf2")
else ()
    set(TF_VERSION_FLAG "not tf1")
endif ()

string(APPEND MARKER_FLAG ${CUDA_FLAG} " and " ${TF_VERSION_FLAG})

add_test(TfNonEagerTrainingExtensionPythonTest
        ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python/non_eager -m ${MARKER_FLAG} --junitxml=${CMAKE_CURRENT_BINARY_DIR}/py_test_non_eager_output.xml
        )

set_property(TEST TfNonEagerTrainingExtensionPythonTest
      APPEND PROPERTY ENVIRONMENT "${AIMET_PYTHONPATH}")

if(DEFINED ENV{DEPENDENCY_DATA_PATH})
set_property(TEST TfNonEagerTrainingExtensionPythonTest
      APPEND PROPERTY ENVIRONMENT "${DEPENDENCY_DATA_PATH}")
endif(DEFINED ENV{DEPENDENCY_DATA_PATH})

if (NOT TF_VERSION MATCHES "^1.15")
    add_test(TfEagerTrainingExtensionTest
            ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python/eager -m ${MARKER_FLAG} --junitxml=${CMAKE_CURRENT_BINARY_DIR}/py_test_eager_output.xml
            )

    set_property(TEST TfEagerTrainingExtensionTest
          APPEND PROPERTY ENVIRONMENT "${AIMET_PYTHONPATH}")

    if(DEFINED ENV{DEPENDENCY_DATA_PATH})
    set_property(TEST TfEagerTrainingExtensionTest
          APPEND PROPERTY ENVIRONMENT "${DEPENDENCY_DATA_PATH}")
    endif(DEFINED ENV{DEPENDENCY_DATA_PATH})

endif ()

add_executable(TfTrainingExtensionCppTest
    TestTfTensorOps.cpp)

target_link_libraries(TfTrainingExtensionCppTest
        gtest_main
        TensorFlow
        TrainingExtensionsTf
        )

add_test(TfTrainingExtensionCppTest
        TfTrainingExtensionCppTest --gtest_output=xml:cpp_test_output.xml)
