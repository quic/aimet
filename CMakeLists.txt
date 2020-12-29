#=============================================================================
#
#  @@-COPYRIGHT-START-@@
#  
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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
cmake_minimum_required(VERSION 3.5)

project(aimet)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -ggdb -fPIC -D_GLIBCXX_USE_CXX11_ABI=0")
set(AIMET_INSTALL_DIR ${CMAKE_BINARY_DIR}/staging/universal)

set(AIMET_PYTHONPATH
      "PYTHONPATH=${CMAKE_BINARY_DIR}/artifacts:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/torch/src/python/:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/tensorflow/src/python/:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/common/src/python/" CACHE STRING "python path")

set(AIMET_LD_LIBRARY_PATH
      "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/artifacts:$ENV{LD_LIBRARY_PATH}")

# Set the software version from version.txt file (if not already set)
if(NOT DEFINED SW_VERSION)
  file(STRINGS "packaging/version.txt" SW_VERSION)
  message(STATUS "Set SW_VERSION = ${SW_VERSION} from ${CMAKE_CURRENT_SOURCE_DIR}/packaging/version.txt")
else()
  message(STATUS "SW_VERSION already set to ${SW_VERSION}.")
endif()

find_package(Threads)
# Disabling openMP since statically linked OpenMP in PyTorch (v1.5.0) seems to conflict with aimet usage
#find_package(OpenMP REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# -------------------------------
# Conditional build for CUDA
# -------------------------------
if (NOT (DEFINED ENABLE_CUDA))
  message("Compiling with CUDA not explicitly disabled. Enabling implicitly")
  set(ENABLE_CUDA ON CACHE BOOL "")

endif(NOT (DEFINED ENABLE_CUDA))

if (ENABLE_CUDA)
  message("Compiling with CUDA enabled")
  enable_language(CUDA)
  set(CUDA_NVCC_FLAGS
        -O3 -std=c++11 -I /usr/local -I /usr/local/cuda-10.0/include -x cu -Xcompiler -fPIC -D_GLIBCXX_USE_CXX11_ABI=0

        -gencode arch=compute_35,code=sm_35
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_52,code=sm_52
        -gencode arch=compute_60,code=sm_60
        -gencode arch=compute_61,code=sm_61
        -gencode arch=compute_61,code=compute_61
        -gencode arch=compute_70,code=compute_70
     )

else(ENABLE_CUDA)
  message("Compiling with CUDA disabled")

endif(ENABLE_CUDA)

set(LAPACKE_LIBRARY "/usr/lib/x86_64-linux-gnu/liblapacke.so.3.7.1")

find_package(OpenCV REQUIRED)
set(OPENCV_LIBRARY /usr/local/lib/libopencv_core.a)

# ----------------------------------
# Conditional build for PyTorch
# ----------------------------------

if (NOT (DEFINED ENABLE_TORCH))
  message("AIMET Torch build not explicitly disabled. Enabling implicitly")
  set(ENABLE_TORCH ON CACHE BOOL "")

endif(NOT (DEFINED ENABLE_TORCH))

if (ENABLE_TORCH)
  # If we enable PyTorch builds then use the pybind11 headers that are part of the torch pip install
  # So we don't have a version mismatch - between PyTorch custom C++ op code and PyMO
  list(APPEND PYBIND11_INC_DIRS "/usr/local/lib/python3.6/dist-packages/torch/include")
  list(APPEND PYBIND11_INC_DIRS "/usr/include/python3.6/")

else (ENABLE_TORCH)
  list(APPEND PYBIND11_INC_DIRS "/usr/local/lib/python3.6/dist-packages/pybind11/include/")
  list(APPEND PYBIND11_INC_DIRS "/usr/include/python3.6/")

  message("AIMET Torch build disabled")

endif (ENABLE_TORCH)

# ----------------------------------
# PyBind11
# ----------------------------------

add_library(PYBIND11 SHARED IMPORTED)

set_target_properties(PYBIND11 PROPERTIES
        IMPORTED_LOCATION "/usr/lib/x86_64-linux-gnu/libpython3.6m.so"
        INTERFACE_INCLUDE_DIRECTORIES "${PYBIND11_INC_DIRS}"
        )

# ----------------------------------
# Conditional build for TensorFlow
# ----------------------------------

if (NOT (DEFINED ENABLE_TENSORFLOW))
  message("AIMET TensorFlow build not explicitly disabled. Enabling implicitly")
  set(ENABLE_TENSORFLOW ON CACHE BOOL "")

endif(NOT (DEFINED ENABLE_TENSORFLOW))

if (ENABLE_TENSORFLOW)
  list(APPEND TENSORFLOW_INCLUDES "/usr/local/lib/python3.6/dist-packages/tensorflow_core/include")
  add_library(TensorFlow SHARED IMPORTED)
  set_target_properties(TensorFlow PROPERTIES
        IMPORTED_LOCATION
          "/usr/local/lib/python3.6/dist-packages/tensorflow_core/libtensorflow_framework.so.1"
        INTERFACE_INCLUDE_DIRECTORIES
          "${TENSORFLOW_INCLUDES}"
        )

else (ENABLE_TENSORFLOW)
  message("AIMET TensorFlow build disabled")

endif (ENABLE_TENSORFLOW)

# -------------------------------
# Generate pip packages
# -------------------------------
# Set the packaging path (if not already set)
if(NOT DEFINED AIMET_PACKAGE_PATH)
  set(AIMET_PACKAGE_PATH "\"${AIMET_INSTALL_DIR}\"")
  message(STATUS "Set AIMET_PACKAGE_PATH = ${AIMET_PACKAGE_PATH}")
endif(NOT DEFINED AIMET_PACKAGE_PATH)

execute_process(COMMAND git config --get remote.origin.url WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE remote_url)
if(NOT remote_url STREQUAL "")
  message(STATUS "Repo Remote URL = ${remote_url}")
  string(REGEX REPLACE "\n$" "" remote_url "${remote_url}")
  # Remove the ".git" suffix from the remote repo URL
  string(REGEX REPLACE "\\.[^.]*$" "" remote_url ${remote_url})

  set(remote_url_cfg "\"${remote_url}\"")
  set(sw_version_cfg "\"${SW_VERSION}\"")

  file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/packaging/setup_cfg.py" "version=${sw_version_cfg}\n")
  file(APPEND "${CMAKE_CURRENT_SOURCE_DIR}/packaging/setup_cfg.py" "remote_url=${remote_url_cfg}")

  add_custom_target(packageaimet
    # Run the install target first
    COMMAND "${CMAKE_COMMAND}" --build . --target install
    # Now run the packaging target to generate wheel files
    COMMAND ${CMAKE_COMMAND} -DAIMET_PACKAGE_PATH=${AIMET_PACKAGE_PATH} -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/packaging/package_aimet.cmake
  )
else()
  message(STATUS "Repo Remote URL is blank. Skipping packageaimet target")
endif()

enable_testing()

add_subdirectory(ModelOptimizations)
add_subdirectory(TrainingExtensions)
add_subdirectory(NightlyTests)
add_subdirectory(ThirdParty)
add_subdirectory(Docs)
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Benchmarks")
  add_subdirectory(Benchmarks)
endif()

# -------------------------------
# Packaging - deprecated
# -------------------------------
set(CPACK_PACKAGE_NAME "aimet")
set(CPACK_PACKAGE_VERSION_MAJOR 0)
set(CPACK_PACKAGE_VERSION_MINOR 1)
set(CPACK_PACKAGE_VERSION_PATCH 1)
set(CPACK_SYSTEM_NAME "build-0.0.0.0")
set(CPACK_GENERATOR "TGZ")
set(CPACK_INSTALLED_DIRECTORIES ${AIMET_INSTALL_DIR} .)
include(CPack)
