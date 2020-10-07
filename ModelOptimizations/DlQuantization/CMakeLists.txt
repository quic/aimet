#==============================================================================
#  @@-COPYRIGHT-START-@@
#  
#  Copyright (c) 2016-2017, Qualcomm Innovation Center, Inc. All rights reserved.
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

add_library(MoDlQuantization STATIC

      src/IQuantizationAlgorithm.hpp
      src/MainQuantizationClass.cpp
      src/MainQuantizationClass.hpp
      src/math_functions.cpp
      src/math_functions.hpp
      src/QuantizerFactory.cpp
      src/TfEnhancedQuantizer.cpp
      src/TfEnhancedQuantizer.hpp
      src/TfQuantizer.cpp
      src/TfQuantizer.hpp
      src/trim_functions.cpp
      src/trim_functions.hpp

      src/TfEnhancedEncodingAnalyzer.cpp
      src/TfEnhancedEncodingAnalyzer.h
      src/TfEncodingAnalyzer.h
      src/TfEncodingAnalyzer.cpp
      src/TensorQuantizationSim.h
      src/TensorQuantizationSim.cpp
      src/EncodingAnalyzerForPython.cpp
      src/TensorQuantizationSimForPython.cpp
      src/TensorQuantizer.cpp

      include/DlQuantization/IQuantizer.hpp
      include/DlQuantization/Quantization.hpp
      include/DlQuantization/QuantizerFactory.hpp
      include/DlQuantization/IQuantizationEncodingAnalyzer.hpp
      include/DlQuantization/ITensorQuantizationSim.h
      include/DlQuantization/EncodingAnalyzerForPython.h
      include/DlQuantization/TensorQuantizationSimForPython.h
      include/DlQuantization/TensorQuantizer.h)

if (ENABLE_CUDA)
target_compile_options(MoDlQuantization
      PRIVATE
         -DGPU_QUANTIZATION_ENABLED=1
      )
endif (ENABLE_CUDA)

target_include_directories(MoDlQuantization
      PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PYBIND11_INC_DIRS}
      )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if (ENABLE_CUDA)
add_library(MoDlQuantizationCuda

      src/math_functions.cu
      src/cuda_util.cu
      src/trim_functions.cu
      src/cuda_util.hpp
      src/trim_functions.cuh
      )

target_compile_options(MoDlQuantizationCuda
      PRIVATE
         -Xcompiler -fPIC)
endif (ENABLE_CUDA)

add_subdirectory(test)
