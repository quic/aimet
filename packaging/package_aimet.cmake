# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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

# CMake file to generate AIMET packages


set(src_packaging_dir "${SOURCE_DIR}/packaging")
set(src_deps_dir "${src_packaging_dir}/dependencies")
set(build_packaging_dir "${CMAKE_BINARY_DIR}/packaging")

# First delete the existing packaging directory if it exists
file(REMOVE_RECURSE ${build_packaging_dir})

# set varinat name
if(DEFINED ENV{AIMET_VARIANT})
  set(variant_name $ENV{AIMET_VARIANT})
else()
  if(ENABLE_TORCH AND ENABLE_TENSORFLOW)
    set(variant_name "tf-torch")
  elseif(ENABLE_TORCH)
    set(variant_name "torch")
  elseif(ENABLE_TENSORFLOW)
    set(variant_name "tf")
  else()
    set(variant_name "tf-torch")
  endif()

  if(ENABLE_CUDA)
    set(variant_name ${variant_name}-gpu)
  else()
    set(variant_name ${variant_name}-cpu)
  endif()
endif()

# Copy NOTICE, README, INSTALL, etc files to build folder for package creation
configure_file("${src_packaging_dir}/NOTICE.txt" "${build_packaging_dir}/NOTICE.txt" COPYONLY)
configure_file("${src_packaging_dir}/README.txt" "${build_packaging_dir}/README.txt" COPYONLY)
configure_file("${src_packaging_dir}/INSTALL.txt" "${build_packaging_dir}/INSTALL.txt" COPYONLY)
configure_file("${src_packaging_dir}/setup_cfg.py" "${build_packaging_dir}/setup_cfg.py" COPYONLY)
configure_file("${src_packaging_dir}/packaging_common.py" "${build_packaging_dir}/packaging_common.py" COPYONLY)

# Common dependencies
set(deps_name_list_aimet_common "reqs_deb_common.txt" "reqs_pip_common.txt")

# Initialize package array list with AIMET common package
set(package_name_list aimet_common)

# Set a GPU flag for use by setup scripts
set(CUDA_OPTION "")
if(ENABLE_CUDA)
  set(CUDA_OPTION "--gpu")
endif()

# Setup Tensorflow package dependencies if required
if(ENABLE_TENSORFLOW)
  # Add AIMET Tensorflow package to package array list
  list(APPEND package_name_list aimet_tensorflow)

  # Tensorflow dependencies that are common to CPU and GPU
  set(deps_name_list_aimet_tensorflow "reqs_pip_tf_common.txt")

  if(ENABLE_CUDA)
    # Tensorflow GPU dependencies
    list(APPEND deps_name_list_aimet_tensorflow "reqs_deb_tf_gpu.txt" "reqs_pip_tf_gpu.txt")
  else()
    # Tensorflow CPU dependencies
    list(APPEND deps_name_list_aimet_tensorflow "reqs_pip_tf_cpu.txt")
  endif()
endif()

# Setup Torch package dependencies if required
if(ENABLE_TORCH)
  # Add AIMET Torch package to package array list
  list(APPEND package_name_list aimet_torch)

  # Torch dependencies that are common to CPU and GPU
  set(deps_name_list_aimet_torch "reqs_pip_torch_common.txt")

  if(ENABLE_CUDA)
    # Torch GPU dependencies
    list(APPEND deps_name_list_aimet_torch "reqs_deb_torch_gpu.txt" "reqs_pip_torch_gpu.txt")
  else()
    # Torch CPU dependencies
    list(APPEND deps_name_list_aimet_torch "reqs_pip_torch_cpu.txt")
  endif()
endif()

# Finally, add AIMET "top-level" package to package array list
list(APPEND package_name_list aimet)


# Loop over the package array list to generate wheel files
foreach(package ${package_name_list})

  # Rename the package setup script
  configure_file("${src_packaging_dir}/setup_${package}.py" "${build_packaging_dir}/setup.py" COPYONLY)
  # Location of the package folder
  set(package_dir "${build_packaging_dir}/${package}")
  # Location of the dependency subfolder within package
  set(package_deps_dir "${package_dir}/bin")

  # Loop over the dependency list for this package and copy to dependency location
  file(MAKE_DIRECTORY "${package_deps_dir}")
  foreach(dependency_file ${deps_name_list_${package}})
    message(VERBOSE "*** Copied ${src_deps_dir}/${dependency_file} to ${package_deps_dir}/ ***")
    configure_file("${src_deps_dir}/${variant_name}/${dependency_file}" "${package_deps_dir}/" COPYONLY)
  endforeach()

  file(WRITE "${build_packaging_dir}/MANIFEST.ini" "")
  if("${package}" STREQUAL "aimet_common")
    # NOTE: MANIFEST file is different for different packages
    file(APPEND "${build_packaging_dir}/MANIFEST.ini" "graft ${package}/x86_64-linux-gnu \n")
    file(APPEND "${build_packaging_dir}/MANIFEST.ini" "include ${package}/*/*.html \n")
    # Populate the python code
    file(COPY ${AIMET_PACKAGE_PATH}/lib/python/${package} DESTINATION ${build_packaging_dir}/)
    # Populate the C++ libraries
    file(COPY ${AIMET_PACKAGE_PATH}/lib/x86_64-linux-gnu/ DESTINATION ${package_dir}/x86_64-linux-gnu/)

    # Copy over dependency installation files
    if(EXISTS "${src_packaging_dir}/LICENSE.pdf")
      configure_file("${src_packaging_dir}/LICENSE.pdf" "${package_deps_dir}/" COPYONLY)
    endif()
    configure_file("${src_packaging_dir}/INSTALL.txt" "${package_deps_dir}/" COPYONLY)
    configure_file("${src_packaging_dir}/envsetup.sh" "${package_deps_dir}/" COPYONLY)
  elseif("${package}" STREQUAL "aimet")
    # Populate top-level AIMET package contents (manifest file)
    file(APPEND "${build_packaging_dir}/MANIFEST.ini" "include README.txt NOTICE.txt \n")
  else()
    # Populate AIMET Tensorflow or Torch package contents (manifest and python code)
    file(APPEND "${build_packaging_dir}/MANIFEST.ini" "graft ${package}/acceptance_tests \n")
    file(APPEND "${build_packaging_dir}/MANIFEST.ini" "include ${package}/*/*.html \n")
    file(COPY ${AIMET_PACKAGE_PATH}/lib/python/${package} DESTINATION ${build_packaging_dir}/)
  endif()
  # Update RPATH to relative paths ($ORIGIN) to not bother by a path where python is installed
  # Linux loader would be able to resolve dependencies without LD_LIBRARY_PATH
  execute_process(
      COMMAND find ${build_packaging_dir} -name "AimetTensorQuantizer*.so" -exec ${PATCHELF_EXE} --set-rpath $ORIGIN:$ORIGIN/../torch/lib {} \;
  )
  execute_process(
      COMMAND find ${build_packaging_dir} -name "libaimet_tf_ops*.so" -exec ${PATCHELF_EXE} --set-rpath $ORIGIN:$ORIGIN/../../tensorflow:$ORIGIN/../../tensorflow/python {} \;
  )
  # Invoke the setup tools script to create the wheel packages.
  execute_process(COMMAND python3 setup.py sdist bdist_wheel ${CUDA_OPTION} WORKING_DIRECTORY ${build_packaging_dir} OUTPUT_VARIABLE output_var)

  # Rename and keep a copy of the manifest file for debugging/reference purposes
  configure_file("${build_packaging_dir}/MANIFEST.ini" "${build_packaging_dir}/MANIFEST.ini.${package}" COPYONLY)

endforeach()
