#=============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

find_program(PUGIXML_EXE pugixml
             PATHS ${CMAKE_BINARY_DIR}/_deps/pugixml-src/bin)
if (PUGIXML_EXE)
    message(STATUS "Found pugixml in '${PUGIXML_EXE}'")
    add_subdirectory(${CMAKE_BINARY_DIR}/_deps/pugixml-src/pugixml-1.13 ${CMAKE_BINARY_DIR}/_deps/pugixml-src/pugixml-1.13 EXCLUDE_FROM_ALL)
elseif (EXISTS $ENV{DEPENDENCY_DATA_PATH}/pugixml.zip)
    message(STATUS "Setting up pugixml from internal cache")
    file(ARCHIVE_EXTRACT INPUT $ENV{DEPENDENCY_DATA_PATH}/pugixml.zip
    DESTINATION ${CMAKE_BINARY_DIR}/_deps/pugixml-src/)
    add_subdirectory(${CMAKE_BINARY_DIR}/_deps/pugixml-src/pugixml-1.13 ${CMAKE_BINARY_DIR}/_deps/pugixml-src/pugixml-1.13 EXCLUDE_FROM_ALL)
elseif (NOT PUGIXML_FOUND)
    FetchContent_Declare(
        pugixml
        GIT_REPOSITORY https://github.com/zeux/pugixml.git
        GIT_TAG        v1.13
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(pugixml)
endif()