#=============================================================================
#
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
#
#=============================================================================

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
include(FetchContent)

if (EXISTS googletest/googletest-release-1.8.0)
    add_subdirectory(googletest/googletest-release-1.8.0 EXCLUDE_FROM_ALL)
else ()
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.10.0
    )
    FetchContent_GetProperties(googletest)
    if (NOT googletest_POPULATED)
        FetchContent_Populate(googletest)
        add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif ()
endif ()

if (NOT OPENCV_FOUND)
  # Aimet requires opencv which might not be installed since it is not part of neither
  # reqs_dep_*.txt nor reqs_pip_*.txt. Download and compile opencv for user only if
  # opencv is not found.
  set(BUILD_LIST        "core" CACHE INTERNAL "[OpenCV] ")
  set(WITH_LAPACK       ON  CACHE INTERNAL "[OpenCV] ")
  set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "[OpenCV] ")
  set(BUILD_EXAMPLES    OFF CACHE INTERNAL "[OpenCV] ")
  set(BUILD_ITT         OFF CACHE INTERNAL "[OpenCV] ")
  set(BUILD_JAVA        OFF CACHE INTERNAL "[OpenCV] ")
  set(BUILD_opencv_apps OFF CACHE INTERNAL "[OpenCV] ")
  set(BUILD_opencv_python2  OFF CACHE INTERNAL "[OpenCV] ")
  set(BUILD_opencv_python3  OFF CACHE INTERNAL "[OpenCV] ")
  set(BUILD_PERF_TESTS  OFF CACHE INTERNAL "[OpenCV] ")
  set(BUILD_TESTS       OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_CUDA         OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_FFMPEG       OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_GTK          OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_IPP          OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_JPEG         OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_OPENEXR      OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_OPENJPEG     OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_PNG          OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_TIFF         OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_V4L          OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_WEBP         OFF CACHE INTERNAL "[OpenCV] ")

  FetchContent_Declare(
    opencv
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG        4.6.0
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(opencv)
  set(OPENCV_LINK_LIBRARIES opencv_core PARENT_SCOPE)
  get_target_property(OPENCV_INCLUDE_DIRS opencv_core INCLUDE_DIRECTORIES)
  set(OPENCV_INCLUDE_DIRS ${OPENCV_INCLUDE_DIRS} PARENT_SCOPE)
endif()

if (ENABLE_TORCH)
  # Aimet ONNX extension requires headers for onnxruntime which might not be installed
  # since they are not part of neither reqs_dep_*.txt nor reqs_pip_*.txt.
  # Download and install onnxruntime headers for user only if onnxruntime package is installed
  # and onnxruntime_cxx_api.h can't be found in /opt/onnxruntime
  execute_process(
    COMMAND ${Python3_EXECUTABLE} "-c" "import onnxruntime; print(onnxruntime.__version__)"
      RESULT_VARIABLE ONNXRUNTIME_NOT_FOUND
      OUTPUT_VARIABLE ONNXRUNTIME_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  file(GLOB_RECURSE ONNXRUNTIME_HEADER_PATH  "/opt/onnxruntime/*onnxruntime_cxx_api.h")
  if (ONNXRUNTIME_NOT_FOUND EQUAL 0 AND NOT ONNXRUNTIME_HEADER_PATH)
    FetchContent_Declare(
      onnxruntime_headers
      URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz
    )
    FetchContent_MakeAvailable(onnxruntime_headers)
    set(CMAKE_INCLUDE_PATH "${CMAKE_INCLUDE_PATH};${onnxruntime_headers_SOURCE_DIR}/include" PARENT_SCOPE)
  endif()
endif()