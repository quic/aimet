#!/bin/bash
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

###############################################################################
## This is a script to build and run tests on AIMET code.
## This script must be run from within the AIMET top-level folder.
## For help and usage information: buildntest.sh -h
###############################################################################

# enable exit on error.
set -e

workspaceFolder=`pwd`
workspaceFolder=`readlink -f ${workspaceFolder}`
outputRootFolder=$workspaceFolder
scriptPath=`readlink -f $(dirname "$0")`
options_string=""
EXIT_CODE=0


interactive_mode=0
dry_run=0
loading_symbol="..."
USER_MOUNT_DIRS=""

usage() {
  echo -e "\nThis is a script to build and run tests on AIMET code."
  echo -e "This script must be executed from within the AIMET repo's top-level folder."
  echo -e "NOTE: This script will build and start a docker container.\n"
  
  echo "${0} [-o <output_folder>]"
  echo "    -b --> build the code"
  echo "    -u --> run unit tests"
  echo "    -v --> run code violation checks (using pylint tool)"
  echo "    -g --> run code coverage checks (using pycov tool)"
  echo "    -s --> run static analysis (using clang-tidy tool)"
  echo "    -a --> run acceptance tests (Warning: This will take a long time to complete!)"
  echo "    -o --> optional output folder. Default is current directory"
  echo "    -i --> just build and start the docker in interactive mode (shell prompt)"
  echo "    -m --> mount the volumes (comma-separated list of paths)"
  echo "    -n --> dry run mode (just display the docker command)"
}


while getopts "o:abcim:nghsuv" opt;
   do
      case $opt in
         a)
             options_string+=" -a"
             ;;
         b)
             options_string+=" -b"
             ;;
         g)
             options_string+=" -g"
             ;;
         u)
             options_string+=" -u"
             ;;
         v)
             options_string+=" -v"
             ;;
         s)
             options_string+=" -s"
             ;;
         h)
             usage
             exit 0
             ;;
         i)
             interactive_mode=1
             ;;
         m)
             USER_MOUNT_DIRS=$OPTARG
             ;;
         n)
             dry_run=1
             loading_symbol=":"
             ;;
         o)
             outputRootFolder=$OPTARG
             ;;
         :)
             echo "Option -$OPTARG requires an argument" >&2
             exit 1;;
         ?)
             echo "Unknown arg $opt"
             usage
             exit 1
             ;;
      esac
done

if [ ${dry_run} -eq 0 ]; then
	set -x
fi



timestamp=$(date +%Y-%m-%d_%H-%M-%S)


if [ ! -d "../aimet" ] && [ ! -d "../aimet-main" ]; then
   echo -e "ERROR: Not in the correct directory!"
   exit 3
fi

echo -e "Building docker image${loading_symbol} \n"
docker_image_name="aimet-dev-docker:latest"
DOCKER_BUILD_CMD="docker build -t ${docker_image_name} -f ${scriptPath}/Jenkins/Dockerfile ."
if [ $interactive_mode -eq 1 ] && [ $dry_run -eq 1 ]; then
	echo ${DOCKER_BUILD_CMD}
	echo
else
	eval ${DOCKER_BUILD_CMD}
fi

if [[ -z "${BUILD_NUMBER}" ]]; then
     results_path=${outputRootFolder}/buildntest_results/$timestamp
     docker_container_name=aimet-dev_${USER}_${timestamp}
else
     results_path=${outputRootFolder}/buildntest_results
     docker_container_name=aimet-dev_${USER}
fi
rm -rf {results_path} | true
mkdir -p ${results_path}

# Kill any previous running containers by the same name
docker ps | grep ${docker_container_name} && docker kill ${docker_container_name} || true

# Add data dependency path as additional volume mount if it exists
if [ -n "${DEPENDENCY_DATA_PATH}" ]; then
   docker_add_vol_mount+=${DEPENDENCY_DATA_PATH}
else
   # If it does not exist, then just add the path of the current script since we cannot leave it 
   # empty
   docker_add_vol_mount+=${scriptPath}
fi

#Check if and which version of nvidia docker is present
set +e
DOCKER_RUN_PREFIX="docker run"
dpkg -s nvidia-container-toolkit > /dev/null 2>&1
NVIDIA_CONTAINER_TOOKIT_RC=$?
dpkg -s nvidia-docker > /dev/null 2>&1
NVIDIA_DOCKER_RC=$?
set -e

if [ $NVIDIA_CONTAINER_TOOKIT_RC -eq 0 ]
then
    DOCKER_RUN_PREFIX="docker run --gpus all"
elif [ $NVIDIA_DOCKER_RC -eq 0 ]
then
    DOCKER_RUN_PREFIX="nvidia-docker run"
else
    echo "WARNING: No nvidia support detected! Unit tests might fail due to GPU dependencies."
fi

echo -e "Starting docker container${loading_symbol} \n"
DOCKER_RUN_CMD="${DOCKER_RUN_PREFIX} --rm --name=$docker_container_name -e DISPLAY=:0 \
				-e DEPENDENCY_DATA_PATH=${DEPENDENCY_DATA_PATH} \
				-u $(id -u ${USER}):$(id -g ${USER}) \
				-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
				-v /tmp/.X11-unix:/tmp/.X11-unix \
				-v ${workspaceFolder}:${workspaceFolder} \
				-v ${outputRootFolder}:${outputRootFolder} \
				-v ${docker_add_vol_mount}:${docker_add_vol_mount} \
				-v /etc/localtime:/etc/localtime:ro \
				-v /etc/timezone:/etc/timezone:ro --network=host --ulimit core=-1 \
				-w ${workspaceFolder} \
				--ipc=host --shm-size=8G"

#check if HOME variable is set
if [[ -v HOME ]]; then
	DOCKER_RUN_CMD="${DOCKER_RUN_CMD} -v ${HOME}:${HOME}"
fi

if [ $interactive_mode -eq 1 ]; then
	for user_dir in ${USER_MOUNT_DIRS}; do
		DOCKER_RUN_CMD="${DOCKER_RUN_CMD} -v ${user_dir}:${user_dir}"
	done
	DOCKER_RUN_CMD="${DOCKER_RUN_CMD} -it --hostname aimet-dev ${docker_image_name}"
	if [ $dry_run -eq 1 ]; then
		echo ${DOCKER_RUN_CMD}
		echo
	else
		eval ${DOCKER_RUN_CMD}
	fi
else
	DOCKER_RUN_CMD="${DOCKER_RUN_CMD} --entrypoint=${scriptPath}/dobuildntest.sh \
	${docker_image_name} ${options_string} -w ${workspaceFolder} \
	-o ${results_path} | tee ${results_path}/full_log.txt"
	eval ${DOCKER_RUN_CMD}

	# Capture the status of the docker command prior to the tee pipe
	EXIT_CODE=${PIPESTATUS[0]}

	if [ ${EXIT_CODE} -ne 0 ]; then
	    echo -e "Docker execution of stage failed!"
	elif [ ! -f "${results_path}/summary.txt" ]; then
	    echo -e "Failed to launch any build or test stages!"
	    EXIT_CODE=3
	elif grep -q FAIL "${results_path}/summary.txt"; then
	    echo -e "One or more stages failed!"
	    EXIT_CODE=3
	fi

	exit $EXIT_CODE
fi
