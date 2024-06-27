#=============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
cmake_minimum_required(VERSION 3.19)

project(aimet)

# Project-global settings
set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard whose features are requested to build this target.")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Boolean describing whether the value of CXX_STANDARD is a requirement.")
set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "Boolean specifying whether compiler specific extensions are requested.")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(AIMET_INSTALL_DIR ${CMAKE_BINARY_DIR}/staging/universal)

if(NOT DEFINED AIMET_PYTHONPATH)
    set(AIMET_PYTHONPATH "PYTHONPATH=${CMAKE_BINARY_DIR}/artifacts" CACHE STRING "python path")
endif()
set(AIMET_PYTHONPATH "${AIMET_PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/common/src/python")

set(WHL_EDITABLE_MODE OFF CACHE BOOL "Enable editable mode, wheels would have symlinks to C++ part instead of copies")
set(WHL_PREP_DIR "${CMAKE_BINARY_DIR}/whlprep" CACHE STRING "A path to store extra files which should be included in the wheels")
set(WHL_PREP_AIMET_COMMON_DIR "${WHL_PREP_DIR}/aimet_common" CACHE STRING "A path to store extra files which should be included in the aimet_common wheel")
set(WHL_PREP_AIMET_TORCH_DIR "${WHL_PREP_DIR}/aimet_torch" CACHE STRING "A path to store extra files which should be included in the aimet_torch wheel")
set(WHL_PREP_AIMET_TENSORFLOW_DIR "${WHL_PREP_DIR}/aimet_tensorflow" CACHE STRING "A path to store extra files which should be included in the aimet_tensorflow wheel")
set(WHL_PREP_AIMET_ONNX_DIR "${WHL_PREP_DIR}/aimet_onnx" CACHE STRING "A path to store extra files which should be included in the aimet_onnx wheel")
mark_as_advanced(WHL_PREP_DIR WHL_PREP_AIMET_COMMON_DIR WHL_PREP_AIMET_TORCH_DIR WHL_PREP_AIMET_TENSORFLOW_DIR WHL_PREP_AIMET_ONNX_DIR)

# Set the software version from version.txt file (if not already set)
if(NOT DEFINED SW_VERSION)
    file(STRINGS "packaging/version.txt" SW_VERSION)
    message(STATUS "Set SW_VERSION = ${SW_VERSION} from ${CMAKE_CURRENT_SOURCE_DIR}/packaging/version.txt")
else()
    message(STATUS "SW_VERSION already set to ${SW_VERSION}.")
endif()

# Set default CMake options
option(ENABLE_CUDA "Enable use of CUDA" ON)
option(ENABLE_TORCH "Enable AIMET-Torch build" ON)
option(ENABLE_ONNX "Enable AIMET-ONNX build" OFF)
option(ENABLE_TENSORFLOW "Enable AIMET-TensorFlow build" OFF)
option(ENABLE_TESTS "Enable building tests" ON)

message(STATUS "AIMET build configuration:")
if (ENABLE_CUDA)
    message(STATUS "** CUDA support enabled.")
else()
    message(STATUS "** CUDA support disabled.")
endif()

if (ENABLE_TORCH)
    message(STATUS "** Building AIMET-Torch enabled.")
else()
    message(STATUS "** Building AIMET-Torch disabled.")
endif()

if (ENABLE_ONNX)
    message (STATUS "** Building AIMET-ONNX enabled.")
else()
    message(STATUS "** Building AIMET-ONNX disabled.")
endif()

if (ENABLE_TENSORFLOW)
    message(STATUS "** Building AIMET-TensorFlow enabled.")
else()
    message(STATUS "** Building AIMET-TensorFlow disabled.")
endif()

if (ENABLE_TESTS)
    message(STATUS "** Building AIMET tests enabled.")
else()
    message(STATUS "** Building AIMET tests disabled.")
endif()

if (ENABLE_TORCH AND ENABLE_TENSORFLOW)
    # Remove this once it's proven feasible.
    message(FATAL_ERROR "AIMET Torch and TensorFlow support currently cannot be built at the same time.")
endif()

if (ENABLE_TORCH)
    message(NOTICE "PyTorch is still pre-built with the pre-C++11 ABI.")
    message(NOTICE "Adding to CXXFLAGS: -D_GLIBCXX_USE_CXX11_ABI=0 -DPYBIND11_BUILD_ABI=\\\"_cxxabi1011\\\"")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0 -DPYBIND11_BUILD_ABI=\\\"_cxxabi1011\\\"")
    #message(NOTICE "Adding to CXXFLAGS: -DPYBIND11_BUILD_ABI=\\\"_cxxabi1011\\\"")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPYBIND11_BUILD_ABI=\\\"_cxxabi1011\\\"")
endif()

if (ENABLE_ONNX AND NOT ENABLE_TORCH)
    message(NOTICE "**** Enabling Torch as it is required by ONNX")
    set(ENABLE_TORCH ON CACHE BOOL "" FORCE)
endif()

# -------------------------------
# Centralized dependency handling
# -------------------------------

# System thread library. Seems to be only needed by DlCompression tests. May be optional - not sure.
find_package(Threads)

# Python libraries
find_package(Python3 COMPONENTS Interpreter Development)
message("Found Python3: ${Python3_FOUND}, at ${Python3_LIBRARIES}")

find_package(PkgConfig)
pkg_search_module(LAPACKE REQUIRED lapacke)

#find_package(OpenCV REQUIRED)
pkg_search_module(OPENCV opencv) # in case opencv will not be found, it would be built form source

if (ENABLE_TORCH)
    include(cmake/PreparePyTorch.cmake)
    find_package(Torch REQUIRED)
endif()

if (ENABLE_ONNX)
    include(cmake/PrepareONNX.cmake)
    find_package(Eigen3 REQUIRED)
    find_package(Torch REQUIRED)
endif()

if (ENABLE_TENSORFLOW)
    include(cmake/PrepareTensorFlow.cmake)
endif()

include(cmake/PreparePyBind11.cmake)

include(cmake/ThirdPartyDependencies.cmake)

# -------------------------------
# Conditional build for CUDA
# -------------------------------
set(CUDA_VER_STRING "cpu")

if (ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Truncate the string for use in version string ex. 11.6.124 --> cu116
    string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.*[0-9]*" "cu\\1\\2" CUDA_VER_STRING ${CUDAToolkit_VERSION})
    message(STATUS "Found CUDA toolkit version ${CUDAToolkit_VERSION}, using ${CUDA_VER_STRING}")
endif(ENABLE_CUDA)

if (ENABLE_TORCH)
  if (NOT ENABLE_ONNX)
    set(FMWORK_VERSION ${TORCH_VERSION})
  endif()
  set(AIMET_PYTHONPATH "${AIMET_PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/torch/src/python")
endif (ENABLE_TORCH)

if (ENABLE_ONNX)
  set(FMWORK_VERSION ${ONNX_VERSION})
  set(AIMET_PYTHONPATH "${AIMET_PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/onnx/src/python")
endif (ENABLE_ONNX)

if (ENABLE_TENSORFLOW)
  set(FMWORK_VERSION ${TF_VERSION})
  set(AIMET_PYTHONPATH "${AIMET_PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/tensorflow/src/python")
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

set(remote_url_cmake_opt "-DREMOTE_URL=\"\"")

if (PIP_INDEX EQUAL "reporelease")
  execute_process(COMMAND git config --get remote.origin.url WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE remote_url)

  if(NOT remote_url STREQUAL "")
    string(REGEX REPLACE "\n$" "" remote_url "${remote_url}")
    # Remove the ".git" suffix from the remote repo URL
    string(REGEX REPLACE "\\.[^.]*$" "" remote_url ${remote_url})
    string(REGEX REPLACE ".*@" "" remote_post ${remote_url})
    set(remote_url "https://${remote_post}")
    message(STATUS "Repo Remote URL = ${remote_url}")
    set(remote_url_cfg "\"${remote_url}\"")
    set(sw_version_cfg "\"${SW_VERSION}\"")

    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/packaging/setup_cfg.py" "version=${sw_version_cfg}\n")
    file(APPEND "${CMAKE_CURRENT_SOURCE_DIR}/packaging/setup_cfg.py" "remote_url=${remote_url_cfg}")

    set(remote_url_cmake_opt "-DREMOTE_URL=${remote_url}")
  else()
    message(FATAL_ERROR "Repo Remote URL is blank. Unable to create AIMET wheel package")
  endif()
endif()

add_custom_target(packageaimet
  # Run the install target first
  COMMAND "${CMAKE_COMMAND}" --build "${PROJECT_BINARY_DIR}" --target install

  # Now run the packaging target to generate wheel files
  COMMAND ${CMAKE_COMMAND} -DSW_VERSION=${SW_VERSION} -DCUDA_VER_STRING=${CUDA_VER_STRING} -DPIP_INDEX=${PIP_INDEX} ${remote_url_cmake_opt} -DPYTHON3_EXECUTABLE=${Python3_EXECUTABLE} -DAIMET_PACKAGE_PATH=${AIMET_PACKAGE_PATH} -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DENABLE_CUDA=${ENABLE_CUDA} -DENABLE_TENSORFLOW=${ENABLE_TENSORFLOW} -DENABLE_TORCH=${ENABLE_TORCH} -DENABLE_ONNX=${ENABLE_ONNX} -DPATCHELF_EXE=${PATCHELF_EXE} -DFMWORK_VERSION=${FMWORK_VERSION} -P ${CMAKE_CURRENT_SOURCE_DIR}/packaging/package_aimet.cmake
)


if (ENABLE_TESTS)
    enable_testing()
    add_subdirectory(NightlyTests)
endif()

add_subdirectory(ModelOptimizations)
add_subdirectory(TrainingExtensions)
add_subdirectory(Examples)
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

# -------------------------------
# Upload pip packages
# -------------------------------

# Check the pip config file path and set it to a default value (if not set)
if(NOT DEFINED PIP_CONFIG_FILE)
  set(PIP_CONFIG_FILE "None")
endif()

# Check the pip index name and set it to a default value (if not set)
if(NOT DEFINED PIP_INDEX)
  set(PIP_INDEX "None")
endif()

# Check the pip certificate path and set it to a default value (if not set)
if(NOT DEFINED PIP_CERT_FILE)
  set(PIP_CERT_FILE "None")
endif()

#TODO For some reason, this package upload target does NOT work as expected and needs to be debugged
add_custom_target(upload
    # Now run the packaging target to upload the pip package
    COMMAND ${CMAKE_COMMAND} -DPIP_CONFIG_FILE=${PIP_CONFIG_FILE} -DPIP_INDEX=${PIP_INDEX} -DPIP_CERT_FILE=${PIP_CERT_FILE} -P ${CMAKE_CURRENT_SOURCE_DIR}/packaging/upload_aimet.cmake
    DEPENDS packageaimet
)
