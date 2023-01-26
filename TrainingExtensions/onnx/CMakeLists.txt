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

find_package(Python3 COMPONENTS Interpreter)

add_subdirectory(src/python)
add_subdirectory(test)


add_library(OnnxCppOps SHARED
        src/QcQuantizeOp.h
        src/AimetOpUtils.h
        src/AimetOpUtils.cpp
        src/QcQuantizeInfo.cpp
        src/QcQuantizeInfo.h
        src/QcQuantizeOp.cpp)

set(ONNXRUNTIME_VER "1.13.1" CACHE STRING "ONNX Runtime version")
set(ONNXRUNTIME_URL "v${ONNXRUNTIME_VER}/onnxruntime-linux-x64-${ONNXRUNTIME_VER}.tgz")

FetchContent_Declare(
        onnxruntime
        URL https://github.com/microsoft/onnxruntime/releases/download/${ONNXRUNTIME_URL}
)

FetchContent_makeAvailable(onnxruntime)
set(ONNXRUNTIME_INCLUDE_DIR ${onnxruntime_SOURCE_DIR}/include)
set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/lib)


target_include_directories(OnnxCppOps PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${ONNXRUNTIME_LIB_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
        )

if (ENABLE_CUDA)
target_link_libraries(OnnxCppOps PUBLIC
        MoDlQuantization
        MoDlEqualization
        MoDlQuantizationCuda
        MoDlCompression

        CUDA::cublas
        ${Python_LIBRARIES}
        )
else (ENABLE_CUDA)
    target_link_libraries(OnnxCppOps PUBLIC
            MoDlQuantization
            MoDlEqualization
            MoDlCompression

            ${Python_LIBRARIES}
            )
endif (ENABLE_CUDA)


set_target_properties(OnnxCppOps PROPERTIES
        OUTPUT_NAME "aimet_ort_ops"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts/aimet_onnx"
        )


Python3_add_library(quant_info SHARED
        src/QcQuantizeInfo.h
        src/QcQuantizeInfo.cpp)


if (ENABLE_CUDA)
    target_link_libraries(quant_info
            PUBLIC
            stdc++
            MoDlQuantization
            MoDlEqualization
            MoDlQuantizationCuda
            MoDlCompression
            PyModelOptimizations

            CUDA::cublas
            ${OPENCV_LINK_LIBRARIES}
            z
            )

else (ENABLE_CUDA)
    target_link_libraries(quant_info
            PUBLIC
            stdc++
            MoDlQuantization
            MoDlEqualization
            MoDlCompression
            PyModelOptimizations

            ${OPENCV_LINK_LIBRARIES}
            z
            )

endif (ENABLE_CUDA)


set_target_properties(quant_info
        PROPERTIES
        OUTPUT_NAME "quant_info"
        SUFFIX ".${Python3_SOABI}.so"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts/aimet_onnx/"
        )

install(TARGETS quant_info
        LIBRARY DESTINATION ${AIMET_INSTALL_DIR}/lib/python/aimet_onnx
        )
