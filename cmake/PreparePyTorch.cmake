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
# - TORCH_VERSION
# - TORCH_CMAKE_PREFIX_PATH
# - TORCH_DIR
# - CMAKE_CUDA_ARCHITECTURES
# - TORCH_CUDA_ARCH_LIST

function(set_torch_version)
    if (NOT ${Python3_FOUND})
        message(FATAL_ERROR "Need Python3 executable to determine PyTorch version.")
    endif()

    execute_process(COMMAND "${Python3_EXECUTABLE}" "-c" "import torch; print(torch.__version__)"
            OUTPUT_VARIABLE TORCH_VERSION_
            OUTPUT_STRIP_TRAILING_WHITESPACE)

    message(STATUS "Found Torch version: ${TORCH_VERSION_}")
    set(TORCH_VERSION ${TORCH_VERSION_} PARENT_SCOPE)
endfunction()

function(set_torch_cmake_prefix_path)
    if (NOT ${Python3_FOUND})
        message(FATAL_ERROR "Need Python3 executable to determine PyTorch CMake prefix path.")
    endif()

    execute_process(COMMAND ${Python3_EXECUTABLE} "-c" "import torch;print(torch.utils.cmake_prefix_path)"
                    RESULT_VARIABLE TORCH_NOT_FOUND
                    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    )

    # Append PyTorch CMake directory to CMAKE_PREFIX_PATH
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${TORCH_CMAKE_PREFIX_PATH}" PARENT_SCOPE)

    # Side-effect: set TORCH_DIR variable (to be used somewhere later)
    get_filename_component(TORCH_DIR_ ${TORCH_CMAKE_PREFIX_PATH}/../../ ABSOLUTE)
    set(TORCH_DIR ${TORCH_DIR_} PARENT_SCOPE)
    message(STATUS "Set TORCH_DIR = ${TORCH_DIR_}")
endfunction()

function(check_torch_cxx_abi_compatibility)
    if (NOT ${Python3_FOUND})
        message(FATAL_ERROR "Need Python3 executable to determine PyTorch version.")
    endif()

    execute_process(COMMAND "${Python3_EXECUTABLE}" "-c" "import torch.utils.cpp_extension;print(torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version(\"g++\")[0])"
            OUTPUT_VARIABLE TORCH_IS_ABI_COMPATIBLE_
            OUTPUT_STRIP_TRAILING_WHITESPACE)

    message(STATUS "TORCH_IS_ABI_COMPATIBLE_ = ${TORCH_IS_ABI_COMPATIBLE_}")
    if (NOT TORCH_IS_ABI_COMPATIBLE_)
        message(FATAL_ERROR "Torch/C++ compiler ABI incompatibility detected.")
    endif()
endfunction()

# ----

macro(update_torch_cuda_arch_list)
    if (NOT ${Python3_FOUND})
        message(FATAL_ERROR "Need Python3 executable to determine PyTorch supported CUDA architectures.")
    endif()

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 72)
    endif()
    message(STATUS "** Initial CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES} **")

    # Update CMAKE_CUDA_ARCHITECTURES with the supported architectures that this pytorch version was
    # compiled for:
    #   1. Remove sm_ prefixes from the CUDA architecture names.
    #   2. Change python list into a CMake list.
    execute_process(COMMAND ${Python3_EXECUTABLE} "-c" "import torch; print(';'.join(arch.split('_')[1] for arch in torch.cuda.get_arch_list()))"
                    RESULT_VARIABLE TORCH_NOT_FOUND
                    OUTPUT_VARIABLE CMAKE_CUDA_ARCHITECTURES
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    )
    message(STATUS "** Updated CMAKE_CUDA_ARCHITECTURES to ${CMAKE_CUDA_ARCHITECTURES} **")

    # We remove certain architectures that are not supported
    set(UNSUPPORTED_CUDA_ARCHITECTURES_TORCH 90)
    list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES ${UNSUPPORTED_CUDA_ARCHITECTURES_TORCH})
    message(STATUS "** Removed unsupported archs (${UNSUPPORTED_CUDA_ARCHITECTURES_TORCH})")
    message(STATUS "** Now CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

    # Set torch cuda architecture list variable
    # Convert to the proper format (Reference: https://stackoverflow.com/a/74962874)
    #   - Insert "." between the digits of the architecture version (ex. 50 --> 5.0)
    #   - Repleace semi-colons in list with spaces
    set(TORCH_CUDA_ARCH_LIST ${CMAKE_CUDA_ARCHITECTURES})
    list(TRANSFORM TORCH_CUDA_ARCH_LIST REPLACE "([0-9])([0-9])" "\\1.\\2")
    string(REPLACE ";" " " TORCH_CUDA_ARCH_LIST "${TORCH_CUDA_ARCH_LIST}")
    message(STATUS "** Updated TORCH_CUDA_ARCH_LIST to ${TORCH_CUDA_ARCH_LIST} **")
endmacro()
