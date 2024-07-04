#=============================================================================
#
#  @@-COPYRIGHT-START-@@
#  
#  Copyright (c) 2018-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
#
#=============================================================================

# INPUTS:
#
# OUTPUTS:
# - TF_VERSION
# - TF_LIB_DIR
# - TF_LIB_FILE
# - PYWRAP_TF_INTERNAL

function(set_tensorflow_version)
    if (NOT ${Python3_FOUND})
        message(FATAL_ERROR "Need Python3 executable to determine TensorFlow version.")
    endif()

    # Get Tensorflow version
    execute_process(COMMAND "${Python3_EXECUTABLE}" "-c" "import tensorflow as tf; print(tf.__version__)"
            OUTPUT_VARIABLE TF_VERSION_
            OUTPUT_STRIP_TRAILING_WHITESPACE)

    message(STATUS "Found TensorFlow version: ${TF_VERSION_}")
    set(TF_VERSION ${TF_VERSION_} PARENT_SCOPE)
endfunction()

function(set_tensorflow_library_path)
    if (NOT ${Python3_FOUND})
        message(FATAL_ERROR "Need Python3 executable to determine TensorFlow library path.")
    endif()

    # Get location of TensorFlow library
    execute_process(COMMAND "${Python3_EXECUTABLE}" "-c" "import tensorflow as tf; print(tf.sysconfig.get_lib())"
            OUTPUT_VARIABLE TF_LIB_DIR_
            OUTPUT_STRIP_TRAILING_WHITESPACE)

    message(STATUS "Found TensorFlow library path: ${TF_LIB_DIR_}")
    set(TF_LIB_DIR ${TF_LIB_DIR_} PARENT_SCOPE)
endfunction()

macro(add_library_tensorflow TF_LIB_DIR)
    # Find the TensorFlow library file
    find_library(TF_LIB_FILE
            NAMES libtensorflow_framework.so.1 libtensorflow_framework.so.2
            HINTS ${TF_LIB_DIR})

    if(NOT TF_LIB_FILE)
        message(FATAL_ERROR "TensorFlow library NOT found.")
    endif()

    add_library(TensorFlow SHARED IMPORTED)

    set_target_properties(TensorFlow PROPERTIES
        IMPORTED_LOCATION "${TF_LIB_FILE}"
        INTERFACE_INCLUDE_DIRECTORIES "${TF_LIB_DIR}/include"
        )
endmacro()

macro(add_library_pywrap_tensorflow_internal TF_LIB_DIR)
    # Find the _pywrap_tensorflow_internal.so library. Used for custom ops.
    find_library(PYWRAP_TF_INTERNAL
            NAMES _pywrap_tensorflow_internal.so
            HINTS ${TF_LIB_DIR}/python/)

    if(NOT PYWRAP_TF_INTERNAL)
        message(FATAL_ERROR "_pywrap_tensorflow_internal library NOT found.")
    endif()

    add_library(PyWrapTensorFlowInternal SHARED IMPORTED)

    set_target_properties(PyWrapTensorFlowInternal PROPERTIES
        IMPORTED_LOCATION "${PYWRAP_TF_INTERNAL}"
        )
endmacro()
