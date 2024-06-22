//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  3. Neither the name of the copyright holder nor the names of its contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//  SPDX-License-Identifier: BSD-3-Clause
//
//  @@-COPYRIGHT-END-@@
//
//==============================================================================
#ifndef PYBINDEX_XMLTYPES_H
#define PYBINDEX_XMLTYPES_H

#include "iostream"
#include "map"
#include "stdexcept"
#include "string"

#define ATTRIB_NAME "Name"
#define OUTPUT "Output"
#define INPUT "Input"
#define PARAMETER "Parameter"
#define MANDATORY "Mandatory"
#define DESCRIPTION "Description"
#define CONTENT "Content"
#define FILTERS "filters"
#define WEIGHTS "weights"
#define WEIGHTS_CAP "Weights"
#define CLOSED_BRACKET "["
#define XML_TRUE "true"
#define XML_FALSE "false"
#define SHAPE "Shape"
#define RANK "Rank"
#define DATATYPE "Datatype"
#define MASTER_OPDEF_LIST "OpDefList"
#define MASTER_OPDEF "OpDef"
#define SUPPLEMENTAL_OPDEF_LIST "SupplementalOpDefList"
#define SUPPLEMENTAL_OPDEF "SupplementalOpDef"
#define SUPPORTED_OPS "SupportedOps"
#define INPUT_SIZE "input_size"
#define OUTPUT_SIZE "output_size"
#define PARAM_SIZE "param_size"
#define INPUT_NAME(num) "in[" + std::to_string(num) + "]"
#define OUTPUT_NAME(num) "out[" + std::to_string(num) + "]"
#define INF_ARG_INDICATOR ".."

#define DEBUG_LOG_INVALID_OP(op_name) "Operation " + op_name + " not found in operation list of the model."
#define DEBUG_LOG_INVALID_PARAM(op_name, param_name) \
    "Operation " + op_name + ": Unexpected parameter name " + attribName + " received."
#define DEBUG_LOG_INVALID_INPUT(op_name, in_size) \
    "Operation " + op_name + ": attrib_num argument expected to be in the range [0," + std::to_string(in_size - 1) + "]"
#define DEBUG_LOG_INVALID_OUTPUT(op_name, out_size)                                                                   \
    "Operation " + op_name + ": attrib_num argument expected to be in the range [0," + std::to_string(out_size - 1) + \
        "]"

#define DEBUG_LOG_XML_FILE_ERROR(path, description)                            \
    {                                                                          \
        std::string error = "Error loading XML: " + path + ": " + description; \
        throw std::runtime_error(error);                                       \
    }

enum class QnnDatatype_t
{
    QNN_DATATYPE_INT_8,
    QNN_DATATYPE_INT_16,
    QNN_DATATYPE_INT_32,
    QNN_DATATYPE_INT_64,

    QNN_DATATYPE_UINT_8,
    QNN_DATATYPE_UINT_16,
    QNN_DATATYPE_UINT_32,
    QNN_DATATYPE_UINT_64,

    QNN_DATATYPE_FLOAT_16,
    QNN_DATATYPE_FLOAT_32,

    QNN_DATATYPE_SFIXED_POINT_8,
    QNN_DATATYPE_SFIXED_POINT_16,
    QNN_DATATYPE_SFIXED_POINT_32,

    QNN_DATATYPE_UFIXED_POINT_8,
    QNN_DATATYPE_UFIXED_POINT_16,
    QNN_DATATYPE_UFIXED_POINT_32,

    QNN_DATATYPE_BOOL_8,

    QNN_DATATYPE_BACKEND_SPECIFIC,
    QNN_DATATYPE_UNDEFINED
};

enum class QnnRank_t
{
    QNN_SCALAR,
    QNN_RANK_1,
    QNN_RANK_2,
    QNN_RANK_3,
    QNN_RANK_4,
    QNN_RANK_5,
    QNN_RANK_N,
    QNN_RANK_INVALID
};
#endif   // PYBINDEX_XMLTYPES_H