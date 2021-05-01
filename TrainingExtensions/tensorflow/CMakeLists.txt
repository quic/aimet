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

add_library(TrainingExtensionsTf SHARED
        src/QcQuantizeStaticOp.hpp
        src/QcQuantizeStaticOp.cpp
        src/QcQuantizeOpDeprecated.cpp
        src/QcQuantizeOpDeprecated.hpp
        src/AimetOpUtils.h
        src/AimetOpUtils.cpp
        src/QcQuantizeRecurrentParamOp.hpp
        src/QcQuantizeRecurrentParamOp.cpp
        src/QcQuantizeOp.hpp
        src/QcQuantizeOp.cpp
        )

target_include_directories(TrainingExtensionsTf PRIVATE
      ${TENSORFLOW_INCLUDES}
      ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
      )

if (ENABLE_CUDA)
    target_compile_options(TrainingExtensionsTf
            PRIVATE
            -DGOOGLE_CUDA=1
            )

    target_link_libraries(TrainingExtensionsTf PUBLIC
            MoDlCompression
            MoDlQuantization
            MoDlQuantizationCuda
            TensorFlow
            TrainingExtensionsTfCuda
            CUDA::cublas
        )

else()
    target_link_libraries(TrainingExtensionsTf PUBLIC
            MoDlCompression
            MoDlQuantization
            TensorFlow
        )

endif()

set_target_properties(TrainingExtensionsTf 
    PROPERTIES
        SUFFIX ".so"
        OUTPUT_NAME "aimet_tf_ops"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts"
    )

get_target_property(dirs MoDlQuantization INCLUDE_DIRECTORIES)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common/include ${TENSORFLOW_INCLUDES}/external/nsync/public ${dirs})

get_target_property(dirs MoDlQuantization INCLUDE_DIRECTORIES)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common/include ${dirs})

if (ENABLE_CUDA)
    add_library(TrainingExtensionsTfCuda
            src/QcQuantizeOpGpuDeprecated.cu
            src/AimetOpUtilsGpu.cu
            )

    target_compile_options(TrainingExtensionsTfCuda
            PRIVATE
            -Xcompiler -fPIC -DGLIBCXX_USE_CXX11_ABI=0
            -DGOOGLE_CUDA=1
            -I ${TENSORFLOW_INCLUDES} --expt-relaxed-constexpr
            )

endif(ENABLE_CUDA)

install(FILES $<TARGET_FILE:TrainingExtensionsTf>
      DESTINATION ${AIMET_INSTALL_DIR}/lib/x86_64-linux-gnu
      )

add_subdirectory(src/python)
add_subdirectory(test)


