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
## This is a script to build and run tests AIMET code.
## This script must be run from within the AIMET top-level folder
###############################################################################

# verbose mode
# set -x

# enable exit on error.
set -e

run_prep=1
run_clean=0
run_build=0
run_unit_tests=0
run_code_violation=0
run_code_coverage=0
run_static_analysis=0
run_acceptance_tests=0

EXIT_CODE=0

# Array to store python source file paths
declare -a PYTHON_SRC_PATHS=()
# Variable to store python source file paths
PYTHON_SRC_MODULE_PATHS=""
# PYTHONPATH variable
PYTHONPATH_VALUE=""

# Array to store python source file paths for code coverage
declare -a PYCOV_SRC_PATHS=()
# Array to store python test file paths for code coverage
declare -a PYCOV_TEST_PATHS=()

workspaceFolder=`pwd`
outputFolder=

function pre_exit {
    # Capture the exit code
    EXIT_CODE=$?

    if [ -z "$outputFolder" ]; then
        outputFolder=$workspaceFolder/buildntest_results
    fi

    summaryFile=${outputFolder}/summary.txt

    if [[ -f ${summaryFile} ]]; then
       echo -e "----------------------------------------------------------------------------------------------------------\n" | tee -a ${summaryFile}
       echo -e "\nResults are in location:\n${outputFolder}\n" | tee -a ${summaryFile}
       cat ${summaryFile}
       if grep -q FAIL "${summaryFile}"; then
           EXIT_CODE=3
       fi
    fi

    # Return the exit code
    exit ${EXIT_CODE}
}
trap pre_exit EXIT

function check_stage() {
    RESULT=$1
    STAGE=$2
    if [ "$3" ]; then
        EXIT_ON_FAIL=$3
    fi

    if [ $RESULT -eq 0 ]; then
        echo -e "Stage $STAGE \t\t PASS " | tee -a ${outputFolder}/summary.txt
    else
        echo -e "Stage $STAGE \t\t FAILED " | tee -a ${outputFolder}/summary.txt
        if [ $EXIT_ON_FAIL == "true" ]; then
            echo -e "\n ABORTED " | tee -a ${outputFolder}/summary.txt
            exit 3
        fi
    fi
}

usage() {
  echo -e "\nThis is a script to build and run tests on AIMET code."
  echo -e "This script must be executed from within the AIMET repo's top-level folder."
  echo -e "NOTE: This script must be executed within the docker container (or in a machine with all dependencies installed). It will NOT start a docker container.\n"
  echo "${0} [-o <output_folder>]"
  echo "    -b --> build the code"
  echo "    -u --> run unit tests"
  echo "    -v --> run code violation checks (using pylint tool)"
  echo "    -g --> run code coverage checks (using pycov tool)"
  echo "    -s --> run static analysis (using clang-tidy tool)"
  echo "    -a --> run acceptance tests (Warning: This will take a long time to complete!)"  
  echo "    -o --> optional output folder. Default is current directory"
  echo "    -w --> path to AIMET workspace. Default is current directory"
}

while getopts "o:w:abcghsuv" opt;
   do
      case $opt in
         a)
             run_acceptance_tests=1
             ;;
         b)
             run_build=1
             ;;
         c)
             run_clean=1
             ;;
         g)
             run_code_coverage=1
             ;;
         u)
             run_unit_tests=1
             ;;
         v)
             run_code_violation=1
             ;;
         s)
             run_static_analysis=1
             ;;
         h)
             usage
             exit 0
             ;;
         o)
             outputFolder=$OPTARG
             ;;
         w)
             workspaceFolder=$OPTARG
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

# If no modes are enabled by user, then enable all modes by default
if [ $run_clean -eq 0 ] && [ $run_acceptance_tests -eq 0 ] && [ $run_build -eq 0 ] && \
    [ $run_unit_tests -eq 0 ] && [ $run_code_violation -eq 0 ] && [ $run_code_coverage -eq 0 ] && \
    [ $run_static_analysis -eq 0 ]; then
    run_prep=1
    run_clean=1
    run_build=1
    run_unit_tests=1
    run_code_violation=1
    run_code_coverage=1
    run_static_analysis=1
    run_acceptance_tests=0
fi

if [[ -z "${workspaceFolder}" ]]; then
    usage
    echo -e "ERROR: Workspace directory was not specified!"
    exit 3
fi

echo "Starting AIMET build and test..."
workspaceFolder=`readlink -f ${workspaceFolder}`
buildFolder=$workspaceFolder/build
artifactsFolder=$buildFolder/artifacts
AIMET_TORCH_HOME=${buildFolder}/torch_pretrain_data

# Sanity check to verify whether we're running form the correct repo location
if [[ ! -d "${workspaceFolder}/TrainingExtensions" ]] && [[ ! -d "${workspaceFolder}/aimet/TrainingExtensions" ]]; then
   echo -e "ERROR: Not in AIMET directory!"
   exit 3
fi

if [[ -d "${workspaceFolder}/Jenkins" ]]; then
    toolsFolder=${workspaceFolder}/Jenkins
elif [[ -d "${workspaceFolder}/aimet/Jenkins" ]]; then
    toolsFolder=${workspaceFolder}/aimet/Jenkins
fi

if [ $run_clean -eq 1 ]; then
    echo -e "\n********** Stage: Clean **********\n"
    if [ -d ${buildFolder} ]; then
        rm -rf ${buildFolder}/* | true
    fi
fi

if [ -z "$outputFolder" ]; then
    outputFolder=$workspaceFolder/build/results
fi
mkdir -p ${outputFolder}
if [ ! -f "${outputFolder}/summary.txt" ]; then
    touch ${outputFolder}/summary.txt
fi
if ! grep -q "AIMET Build and Test Summary" "${outputFolder}/summary.txt"; then
    echo -e "\n----------------------------------------------------------------------------------------------------------" | tee -a ${outputFolder}/summary.txt
    echo -e "\t\t AIMET Build and Test Summary " | tee -a ${outputFolder}/summary.txt
    echo -e "----------------------------------------------------------------------------------------------------------" | tee -a ${outputFolder}/summary.txt
fi

if [ $run_prep -eq 1 ]; then
    echo -e "\n********** Stage 1: Preparation **********\n"
    cd $workspaceFolder

    # Download the checkpoint files if they don't already exist
    #NOTE: We needed this due to some intermittant issues downloading via torchvision
    ## mkdir -p ${AIMET_TORCH_HOME}/checkpoints
    ## wget -N https://download.pytorch.org/models/resnet18-5c106cde.pth -P ${AIMET_TORCH_HOME}/checkpoints
    ## wget -N https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth -P ${AIMET_TORCH_HOME}/checkpoints

    # Clone the google test repo if not already present
    if [ ! -d $workspaceFolder/ThirdParty/googletest/googletest-release-1.8.0 ]; then
        mkdir -p $workspaceFolder/ThirdParty/googletest
        pushd $workspaceFolder/ThirdParty/googletest
        git clone https://github.com/google/googletest.git -b release-1.8.0 googletest-release-1.8.0
        popd
        check_stage $? "Preparation" "true"
    fi

    # Array of python src file path endings
    declare -a python_src_path_endings=("TrainingExtensions/tensorflow/src/python/aimet_tensorflow"
        "TrainingExtensions/torch/src/python/aimet_torch"
        "TrainingExtensions/common/src/python/aimet_common")

    # Populate an array of python src paths for use in later stages
    for python_src_path_ending in "${python_src_path_endings[@]}"; do
    	# Find all paths 
        PYTHON_SRC_PATHS+=($(find . -path "*$python_src_path_ending" -exec readlink -f {} \;))
    done

    # Populate the PYTHONPATH env variable value for use in later stages
    PYTHONPATH_VALUE+=$artifactsFolder
    for python_src_path in "${PYTHON_SRC_PATHS[@]}"; do
	    # Append the parent of each python src path to PYTHONPATH (separated by colon)
        python_src_path_parent=$(readlink -f ${python_src_path}/..)
        PYTHONPATH_VALUE+=":"
        PYTHONPATH_VALUE+=${python_src_path_parent}
	    # Append the same path also to a string (separated by space)
        PYTHON_SRC_MODULE_PATHS+=python_src_path_parent
        PYTHON_SRC_MODULE_PATHS+=" "
    done
    echo "PYTHONPATH value = $PYTHONPATH_VALUE"

    # list of path endings of interest for code coverage and their corresponding test folders
    pycov_dir_endings=("TrainingExtensions/tensorflow/src/python:TrainingExtensions/tensorflow/test"
                   "TrainingExtensions/torch/src/python:TrainingExtensions/torch/test" 
                   "TrainingExtensions/common/src/python:TrainingExtensions/common/test")
    # Loop over the directory endings
    for pycov_dir_ending in "${pycov_dir_endings[@]}"; do
        pycov_src_path_ending=${pycov_dir_ending%%:*}
        pycov_test_path_ending=${pycov_dir_ending#*:}
        # Find all absolute src and test folders ending in the endings of interest
        PYCOV_SRC_PATHS+=($(find . -path "*$pycov_src_path_ending" -exec readlink -f {} \; | grep -v build))
        PYCOV_TEST_PATHS+=($(find . -path "*$pycov_test_path_ending" -exec readlink -f {} \; | grep -v build))
    done

    # Just display all the code coverage paths for debugging purposes
    for ((index=0;index<${#PYCOV_SRC_PATHS[@]};++index)); do
        pycov_src_path=${PYCOV_SRC_PATHS[index]}
        pycov_test_path=${PYCOV_TEST_PATHS[index]}
        echo -e "pycov_src_path = $pycov_src_path, pycov_test_path = $pycov_test_path"
    done
fi

if [ $run_build -eq 1 ]; then
    echo -e "\n********** Stage 2: Build **********\n"

    mkdir -p build
    cd build

    extra_opts=""
    if [ -n "$SW_VERSION" ]; then
        extra_opts+="-DSW_VERSION=${SW_VERSION}"
    fi
    # Do not exit on failure by default from this point forward
    set +e
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ${extra_opts} ..

    make -j 8
    check_stage $? "Build" "true"

    echo -e "\n********** Stage 2a: Generate Docs **********\n"
    make doc
    check_stage $? "Generate Doc" "true"

    #TODO We temporarily disable this install step
    ## echo -e "\n********** Stage 2b: Install **********\n"
    ## make install
    ## check_stage $? "Install" "true"

    #TODO We temporarily disable this pip package generation step
    ## echo -e "\n********** Stage 2c: Package **********\n"
    ## make packageaimet
    ## check_stage $? "Package" "true"
fi

if [ $run_unit_tests -eq 1 ]; then
    echo -e "\n********** Stage 3: Unit tests **********\n"
    cd $workspaceFolder/build
    set +e
    unit_test_cmd=""
    if [[ -z ${DEPENDENCY_DATA_PATH} ]]; then
        echo -e "DEPENDENCY_DATA_PATH was NOT set"
    else
        echo -e "DEPENDENCY_DATA_PATH was set to ${DEPENDENCY_DATA_PATH}"
        unit_test_cmd+="export DEPENDENCY_DATA_PATH=${DEPENDENCY_DATA_PATH} && "
    fi
    unit_test_cmd+="export TORCH_HOME=${AIMET_TORCH_HOME} && ctest --verbose"
    eval " $unit_test_cmd"
    unit_test_rc=$?
    python ${toolsFolder}/unittesthelper.py ${workspaceFolder}
    check_stage $unit_test_rc "Unit tests" "false"
fi

if [ $run_code_violation -eq 1 ]; then
    echo -e "\n********** Stage 4: Code violation checks **********\n"
    cd $workspaceFolder
    pylint_results_dir=$outputFolder/code_violation_result
    mkdir -p ${pylint_results_dir}

    for python_src_path in "${PYTHON_SRC_PATHS[@]}"; do
        # Construct the pylint results file name
        # Remove the top-level path from the full path for brevity
        pylint_results_file_name=$(echo ${python_src_path#${workspaceFolder}/})
        # Replace forward slashes with underscores
        pylint_results_file_name=$(echo $pylint_results_file_name | sed -e 's/\//_/g')        
        # Append the suffix and extension
        pylint_results_file_name+="_pylint_results.out"

        LD_LIBRARY_PATH=$artifactsFolder:$LD_LIBRARY_PATH \
        PYTHONPATH=$PYTHONPATH_VALUE \
        pylint --rcfile=${workspaceFolder}/.pylintrc -r n --msg-template='{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}' ${python_src_path} 2>&1 \
        | tee ${pylint_results_dir}/${pylint_results_file_name}
        code_violation_result=$?

        if [ $code_violation_result -eq 0 ]; then
            if grep -q "error:" ${pylint_results_dir}/${pylint_results_file_name}; then
                echo -e "\n********** ${python_src_path} code violation analysis results START **********\n"
                cat ${pylint_results_dir}/${pylint_results_file_name}
                echo -e "\n********** ${python_src_path} code violation analysis results END **********\n"
            fi
        fi
        check_stage $code_violation_result "Code violation checks: ${python_src_path}" "false"
    done
fi

if [ $run_static_analysis -eq 1 ]; then
    echo -e "\n********** Stage 5: Static analysis **********\n"
    static_analysis_result=0
    clangtidy_results_dir=$outputFolder
    mkdir -p ${clangtidy_results_dir}
    cd $workspaceFolder/build; python2.7 /usr/bin/run-clang-tidy.py >| ${clangtidy_results_dir}/clang-tidy_results.out
    static_analysis_result=$?
    if [ $static_analysis_result -eq 0 ]; then
        if grep -q "error:" "${clangtidy_results_dir}/clang-tidy_results.out"; then
            #Let static analysis pass for warnings, uncomment next line to enfore analysis again
            #static_analysis_result=1
            echo -e "\n********** Static analysis results START **********\n"
            cat ${clangtidy_results_dir}/clang-tidy_results.out
            echo -e "\n********** Static analysis results END **********\n"
        fi
    fi
    check_stage $static_analysis_result "Static analysis" "false"
fi

if [ $run_acceptance_tests -eq 1 ]; then
    echo -e "\n********** Stage 6: Acceptance tests **********\n"
    cd $workspaceFolder/build
    set +e
    make AcceptanceTests
    acceptance_test_rc=$?
    python ${toolsFolder}/acceptancetesthelper.py ${workspaceFolder}
    check_stage $acceptance_test_rc "Acceptance tests" "false"
fi

if [ $run_code_coverage -eq 1 ]; then
    echo -e "\n********** Stage 7: Code coverage **********\n"
    set +e
    pycov_results_dir=$outputFolder/coverage_test_results

    # Loop over the code coverage paths
    for ((index=0;index<${#PYCOV_SRC_PATHS[@]};++index)); do
        pycov_src_path=${PYCOV_SRC_PATHS[index]}
        pycov_test_path=${PYCOV_TEST_PATHS[index]}

	    # Verify that the directories exist
        if [ ! -d ${pycov_src_path} ] || [ ! -d ${pycov_test_path} ]; then
            echo -e "\n[ERROR] Code coverage directories do not exist\n"
            coverage_test_rc=1
        fi

        # Construct the code coverage results file name
        # Remove the top-level path from the full path for brevity
        pycov_results_file_name=$(echo ${pycov_src_path#${workspaceFolder}/})
        # Replace forward slashes with underscores
        pycov_results_file_name=$(echo $pycov_results_file_name | sed -e 's/\//_/g')        
        # Append the suffix and extension
        pycov_results_file_name+="_code_coverage.xml"

        # Run the code coverage
        TORCH_HOME=${AIMET_TORCH_HOME} LD_LIBRARY_PATH=$artifactsFolder:$LD_LIBRARY_PATH PYTHONPATH=$PYTHONPATH_VALUE py.test --cov=${pycov_src_path} ${pycov_test_path} --cov-report xml:${pycov_results_dir}/${pycov_results_file_name}
        coverage_test_rc=$?
        cp -a ${pycov_results_dir}/${pycov_results_file_name} $outputFolder
    done
    check_stage $coverage_test_rc "Code coverage" "false"
fi

echo -e "\n outputFolder = ${outputFolder}"
if grep -q FAIL "${outputFolder}/summary.txt"; then
    EXIT_CODE=3
fi

exit $EXIT_CODE

