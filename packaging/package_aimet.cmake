# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
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
# =============================================================================

cmake_minimum_required(VERSION 3.5)


#copying NOTICE, README, INSTALL files to build folder for Aimet whl package creation
configure_file("${SOURCE_DIR}/packaging/NOTICE.txt" "${CMAKE_BINARY_DIR}/packaging/NOTICE.txt" COPYONLY)
configure_file("${SOURCE_DIR}/packaging/README.txt" "${CMAKE_BINARY_DIR}/packaging/README.txt" COPYONLY)
configure_file("${SOURCE_DIR}/packaging/requirements.txt" "${CMAKE_BINARY_DIR}/packaging/requirements.txt" COPYONLY)
configure_file("${SOURCE_DIR}/packaging/INSTALL.txt" "${CMAKE_BINARY_DIR}/packaging/INSTALL.txt" COPYONLY)
configure_file("${SOURCE_DIR}/packaging/setup_cfg.py" "${CMAKE_BINARY_DIR}/packaging/setup_cfg.py" COPYONLY)

# Setup the package array list
set(package_name_list aimet_common)
if(ENABLE_TENSORFLOW)
  list(APPEND package_name_list aimet_tensorflow)
endif()
if(ENABLE_TORCH)
  list(APPEND package_name_list aimet_torch)
endif()
list(APPEND package_name_list aimet)

#to create whl packages, copying set up files, and related code to build directory.
#these copied files would be input to the setuptools

foreach(package ${package_name_list})

  configure_file("${SOURCE_DIR}/packaging/setup_${package}.py" "${CMAKE_BINARY_DIR}/packaging/setup.py" COPYONLY)
  
  if("${package}" STREQUAL "aimet_common")
    #MANIFEST file is different for different packages 
    file(WRITE "${CMAKE_BINARY_DIR}/packaging/MANIFEST.ini" "graft ${package}/x86_64-linux-gnu")
    file(COPY ${AIMET_PACKAGE_PATH}/lib/python/${package} DESTINATION ${CMAKE_BINARY_DIR}/packaging/)
    file(COPY ${AIMET_PACKAGE_PATH}/lib/x86_64-linux-gnu/ DESTINATION ${CMAKE_BINARY_DIR}/packaging/${package}/x86_64-linux-gnu/)

    # Copy over dependency installation files
    configure_file("${SOURCE_DIR}/packaging/requirements.txt" "${CMAKE_BINARY_DIR}/packaging/${package}/bin/" COPYONLY)
    configure_file("${SOURCE_DIR}/packaging/packages_common.txt" "${CMAKE_BINARY_DIR}/packaging/${package}/bin/" COPYONLY)
    configure_file("${SOURCE_DIR}/packaging/packages_gpu.txt" "${CMAKE_BINARY_DIR}/packaging/${package}/bin/" COPYONLY)
    configure_file("${SOURCE_DIR}/packaging/INSTALL.txt" "${CMAKE_BINARY_DIR}/packaging/${package}/bin/" COPYONLY)
    configure_file("${SOURCE_DIR}/packaging/envsetup.sh" "${CMAKE_BINARY_DIR}/packaging/${package}/bin/" COPYONLY)
  elseif("${package}" STREQUAL "aimet")
    file(WRITE "${CMAKE_BINARY_DIR}/packaging/MANIFEST.ini" "include README.txt NOTICE.txt")
  else()
    file(WRITE "${CMAKE_BINARY_DIR}/packaging/MANIFEST.ini" "graft ${package}/acceptance_tests")
    file(COPY ${AIMET_PACKAGE_PATH}/lib/python/${package} DESTINATION ${CMAKE_BINARY_DIR}/packaging/)
  endif()
  #creating the whl packages. 
  execute_process(COMMAND python3 setup.py sdist bdist_wheel WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/packaging OUTPUT_VARIABLE output_var)

endforeach()
