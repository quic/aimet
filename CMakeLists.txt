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

if(NOT DEFINED AIMET_PYTHONPATH)
  set(AIMET_PYTHONPATH "PYTHONPATH=${CMAKE_BINARY_DIR}/artifacts" CACHE STRING "python path")
endif()
set(AIMET_PYTHONPATH "${AIMET_PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/common/src/python")

if(NOT DEFINED AIMET_LD_LIBRARY_PATH)
  set(AIMET_LD_LIBRARY_PATH "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/artifacts:$ENV{LD_LIBRARY_PATH}")
  message(STATUS "Set ${AIMET_LD_LIBRARY_PATH} in ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

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

  find_package(CUDAToolkit)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 72)
  endif()

  enable_language(CUDA)

else(ENABLE_CUDA)
  message("Compiling with CUDA disabled")

endif(ENABLE_CUDA)

set(LAPACKE_LIBRARY "/usr/lib/x86_64-linux-gnu/liblapacke.so.3.7.1")

find_package(OpenCV REQUIRED)
set(OPENCV_LIBRARY /usr/local/lib/libopencv_core.a)

# ----------------------------------
# Conditional build for PyTorch
# ----------------------------------

find_package(Python COMPONENTS Interpreter Development)
set(PYTHON_DIST_PACKAGES "/usr/local/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/dist-packages/")
message("Found python: ${Python_FOUND}, at ${Python_LIBRARIES}")
EXECUTE_PROCESS( COMMAND "uname" "-m" OUTPUT_VARIABLE ARCHITECTURE OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "python" "-c" "import sys; print(sys.abiflags)" OUTPUT_VARIABLE ABI_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)

if (NOT (DEFINED ENABLE_TORCH))
  message("AIMET Torch build not explicitly disabled. Enabling implicitly")
  set(ENABLE_TORCH ON CACHE BOOL "")

endif(NOT (DEFINED ENABLE_TORCH))

if (ENABLE_TORCH)
  set(AIMET_PYTHONPATH "${AIMET_PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/torch/src/python:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/onnx/src/python")

  # If we enable PyTorch builds then use the pybind11 headers that are part of the torch pip install
  # So we don't have a version mismatch - between PyTorch custom C++ op code and PyMO
  list(APPEND PYBIND11_INC_DIRS "${PYTHON_DIST_PACKAGES}/torch/include")
  list(APPEND PYBIND11_INC_DIRS ${Python_INCLUDE_DIRS})

else (ENABLE_TORCH)
  list(APPEND PYBIND11_INC_DIRS "${PYTHON_DIST_PACKAGES}/pybind11/include/")
  list(APPEND PYBIND11_INC_DIRS ${Python_INCLUDE_DIRS})

  message("AIMET Torch build disabled")

endif (ENABLE_TORCH)

# ----------------------------------
# PyBind11
# ----------------------------------

add_library(PYBIND11 SHARED IMPORTED)

set_target_properties(PYBIND11 PROPERTIES
        IMPORTED_LOCATION ${Python_LIBRARIES}
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
  if(NOT DEFINED TF_LIB_DIR)
    # Get location of TensorFlow library
    execute_process(COMMAND "python" "-c" "import tensorflow as tf; print(tf.sysconfig.get_lib())" OUTPUT_VARIABLE TF_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    # Get Tensorflow version
    execute_process(COMMAND "python" "-c" "import tensorflow as tf; print(tf.__version__)" OUTPUT_VARIABLE TF_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Found TensorFlow version: ${TF_VERSION} TensorFlow library path: ${TF_LIB_DIR}")
  else(TF_LIB_DIR)
    message(STATUS "TensorFlow version and TensorFlow library already set.")
  endif(NOT DEFINED TF_LIB_DIR)

  # Find the TensorFlow library file
  find_library(TF_LIB_FILE NAMES libtensorflow_framework.so.1 libtensorflow_framework.so.2 HINTS ${TF_LIB_DIR})
  if(NOT TF_LIB_FILE)
    message(FATAL_ERROR "TensorFlow library NOT found at ${TF_LIB_DIR}")
  endif()

  # find the _pywrap_tensorflow_internal.so library. Used for custom ops
  find_library(PYWRAP_TF_INTERNAL NAMES _pywrap_tensorflow_internal.so HINTS ${TF_LIB_DIR}/python/)

  set(AIMET_PYTHONPATH "${AIMET_PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/tensorflow/src/python")
  list(APPEND TENSORFLOW_INCLUDES "${TF_LIB_DIR}/include")
  add_library(TensorFlow SHARED IMPORTED)
  set_target_properties(TensorFlow PROPERTIES
        IMPORTED_LOCATION
          "${TF_LIB_FILE}"
        INTERFACE_INCLUDE_DIRECTORIES
          "${TENSORFLOW_INCLUDES}"
        )

  add_library(PyWrapTensorFlowInternal SHARED IMPORTED)
  set_target_properties(PyWrapTensorFlowInternal PROPERTIES
        IMPORTED_LOCATION "${PYWRAP_TF_INTERNAL}"
        )

else (ENABLE_TENSORFLOW)
  message("AIMET TensorFlow build disabled")

endif (ENABLE_TENSORFLOW)

# Export PYTHONPATH to the parent cmake scope (if present)
get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
  set(AIMET_PYTHONPATH "${AIMET_PYTHONPATH}" PARENT_SCOPE)
else()
  message(STATUS "Set ${AIMET_PYTHONPATH} in ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

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
    COMMAND ${CMAKE_COMMAND} -DAIMET_PACKAGE_PATH=${AIMET_PACKAGE_PATH} -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DENABLE_CUDA=${ENABLE_CUDA} -DENABLE_TENSORFLOW=${ENABLE_TENSORFLOW} -DENABLE_TORCH=${ENABLE_TORCH} -P ${CMAKE_CURRENT_SOURCE_DIR}/packaging/package_aimet.cmake
  )
else()
  message(STATUS "Repo Remote URL is blank. Skipping packageaimet target")
endif()

enable_testing()

add_subdirectory(ModelOptimizations)
add_subdirectory(TrainingExtensions)
add_subdirectory(Examples)
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
