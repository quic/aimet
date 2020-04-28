#!/bin/bash -l
#==============================================================================
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
#==============================================================================

# check that envsetup has been sourced
: ${AIMET_ROOT?"Need AIMET_ROOT set before running this script. Please source ./envsetup"}
crad_docker_config="${AIMET_ROOT}/lib/config/aimet.conf"

if [[ ! -f ${crad_docker_config} ]]; then
    echo "Error: crad-docker config file: $crad_docker_config not found!"
    exit 1
fi

addcrad ()
{
    local DISTRONAME=$(lsb_release -si|tr '[:upper:]' '[:lower:']);
    local DISTROVERS=$(lsb_release -sr);
    local VENV_DIR=/pkg/crad/python-2.7.10/${DISTRONAME}${DISTROVERS};
    [ -f $VENV_DIR/bin/activate ] && source $VENV_DIR/bin/activate
}

addcrad
config_section_entry=`crad-docker -c ${crad_docker_config} -l 2>&1 | awk -F ':' 'END{print $NF}'`
echo -e "Using: \ncrad docker config file: ${crad_docker_config} \nconfig_entry: ${config_section_entry}\n"
crad-docker -c  ${crad_docker_config} ${config_section_entry}