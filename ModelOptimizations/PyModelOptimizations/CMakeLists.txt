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
cmake_minimum_required(VERSION 3.5)

add_library(PyModelOptimizations SHARED
      PyModelOptimizations.cpp)

if (ENABLE_CUDA)
    target_link_libraries(PyModelOptimizations
          PUBLIC
             stdc++
             PYBIND11
             MoDlQuantization
             MoDlEqualization
             MoDlQuantizationCuda
             MoDlCompression

            CUDA::cublas
             ${LAPACKE_LIBRARY}
             ${OPENCV_LIBRARY}
             z
          )

else (ENABLE_CUDA)
    target_link_libraries(PyModelOptimizations
            PUBLIC
            stdc++
            PYBIND11
            MoDlQuantization
            MoDlEqualization
            MoDlCompression

            ${LAPACKE_LIBRARY}
            ${OPENCV_LIBRARY}
            z
            )

endif (ENABLE_CUDA)

set(PYMO_HEADERS
    "../DlCompression/include/DlCompression/ISVD.hpp"
    "../DlEqualization/include/DlEqualization/BatchNormFoldForPython.h"
    "../DlEqualization/include/DlEqualization/BiasCorrectionForPython.h"
    "../DlEqualization/include/DlEqualization/CrossLayerScaling.h"
    "../DlEqualization/include/DlEqualization/HighBiasFoldForPython.h"
    "../DlEqualization/include/DlEqualization/def.h"
    "../DlQuantization/include/DlQuantization/EncodingAnalyzerForPython.h"
    "../DlQuantization/include/DlQuantization/IQuantizationEncodingAnalyzer.hpp"
    "../DlQuantization/include/DlQuantization/IQuantizer.hpp"
    "../DlQuantization/include/DlQuantization/Quantization.hpp"
    "../DlQuantization/include/DlQuantization/QuantizerFactory.hpp"
    "../DlQuantization/include/DlQuantization/TensorQuantizationSimForPython.h"
    "../DlQuantization/include/DlQuantization/TensorQuantizer.h"
    "../DlQuantization/include/DlQuantization/TensorQuantizerOpFacade.h"
)

set_target_properties(PyModelOptimizations
      PROPERTIES
         OUTPUT_NAME "pymo"
         LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts"
         PUBLIC_HEADER "${PYMO_HEADERS}"
      )

install(TARGETS PyModelOptimizations
        LIBRARY DESTINATION ${AIMET_INSTALL_DIR}/lib/x86_64-linux-gnu
        PUBLIC_HEADER DESTINATION ${AIMET_INSTALL_DIR}/lib/x86_64-linux-gnu/include
)
