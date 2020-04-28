#!/bin/bash
#==============================================================================
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
#==============================================================================

# This script sets up the various environment variables needed to use various modules and scripts

OPTIND=1

# Get directory of the bash script
SOURCEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

usage()
{
cat << EOF
usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-h] [-c CAFFE2_DIRECTORY] [-t TENSORFLOW_DIRECTORY]

Script sets up environment variables needed for using python modules and scripts, where only one of the
Caffe2 or Tensorflow directories have to be specified. Note that if the PYTHONPATH and LD_LIBRARY_PATH are already
set up to point to the training framework python modules and libraries specifying these options is unecessary.

optional arguments:
 -c CAFFE2_DIRECTORY           Specifies Caffe2 directory
 -t TENSORFLOW_DIRECTORY       Specifies TensorFlow directory

EOF
}

unset TENSORFLOWDIR
unset CAFFE2DIR

while getopts ":hc:t:" opt; do
  case $opt in
    h)
      usage
      exit 0
      ;;
    c)
      CAFFE2DIR=$2
      break
      ;;
    t)
      TENSORFLOWDIR=$2
      break
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument" >&2
      usage
      exit 1
      ;;
  esac
done

# Setup the AIMET specific paths
export AIMET_ROOT=$(readlink -f $SOURCEDIR/..)
export PATH=$AIMET_ROOT/bin/x86_64-linux-gnu:$AIMET_ROOT/lib/python/aimet_common/bin:$PATH
export LD_LIBRARY_PATH=$AIMET_ROOT/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

if [ -z $PYTHONPATH ]; then
   export PYTHONPATH=$AIMET_ROOT/lib/python:$AIMET_ROOT/lib/x86_64-linux-gnu
else
   export PYTHONPATH=$AIMET_ROOT/lib/python:$AIMET_ROOT/lib/x86_64-linux-gnu:$PYTHONPATH
fi

if [[ ! -z "$CAFFE2DIR" ]]; then
  if [ ! -d "$CAFFE2DIR" ]; then
      echo "Invalid directory "$CAFFE2DIR" specified. Please rerun the srcipt with a valid directory path."
      usage
      exit 1
  fi
  export CAFFE2_HOME=$CAFFE2DIR
  echo "INFO: Setting CAFFE2_HOME="$CAFFE2DIR

  export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
  export PYTHONPATH=$CAFFE2DIR/build/:$PYTHONPATH
  export PYTHONPATH=/usr/local/:$PYTHONPATH

elif [[ ! -z "$TENSORFLOWDIR" ]]; then
  if [ ! -d "$TENSORFLOWDIR" ]; then
      echo "Invalid directory "$TENSORFLOWDIR" specified. Please rerun the srcipt with a valid directory path."
      usage
      exit 1
  fi
  export TENSORFLOW_HOME=$TENSORFLOWDIR
  echo "INFO: Setting TENSORFLOW_HOME="$TENSORFLOWDIR
fi
