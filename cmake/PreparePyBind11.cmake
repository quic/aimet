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

macro(add_library_pybind11)
    if (NOT ${Python3_FOUND})
        message(FATAL_ERROR "Need Python3 executable to determine pybind11 include path.")
    endif()

    execute_process(COMMAND ${Python3_EXECUTABLE} "-c" "import pybind11;print(pybind11.get_include())"
                    RESULT_VARIABLE PYBIND11_NOT_FOUND
                    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR_
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    )

    # If we enable PyTorch builds then use the pybind11 headers that are part of the torch pip install
    # So we don't have a version mismatch - between PyTorch custom C++ op code and PyMO
    find_path(PYBIND11_HEADER "pybind11.h"
            PATHS ${TORCH_INCLUDE_DIRS} ${PYBIND11_INCLUDE_DIR_}
            PATH_SUFFIXES "pybind11"
            REQUIRED
            NO_DEFAULT_PATH
            )

    get_filename_component(PYBIND11_INCLUDE_DIR ${PYBIND11_HEADER} DIRECTORY)

    if (NOT PYBIND11_INCLUDE_DIR)
        message(FATAL_ERROR "Could not find pybind11.")
    endif()

    add_library(pybind11 SHARED IMPORTED)

    set_target_properties(pybind11 PROPERTIES
            IMPORTED_LOCATION ${Python3_LIBRARIES}
            INTERFACE_INCLUDE_DIRECTORIES "${PYBIND11_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES Python3::Module
            )
endmacro()
