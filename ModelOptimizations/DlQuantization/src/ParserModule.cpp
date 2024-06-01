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

#include "DlQuantization/ParserModule.h"

QnnDatatype_t strToDtype(const std::string& dtype)
{
    if (dtype == "QNN_DATATYPE_INT_8")
        return QnnDatatype_t::QNN_DATATYPE_INT_8;
    else if (dtype == "QNN_DATATYPE_INT_16")
        return QnnDatatype_t::QNN_DATATYPE_INT_16;
    else if (dtype == "QNN_DATATYPE_INT_32")
        return QnnDatatype_t::QNN_DATATYPE_INT_32;
    else if (dtype == "QNN_DATATYPE_INT_64")
        return QnnDatatype_t::QNN_DATATYPE_INT_64;

    else if (dtype == "QNN_DATATYPE_UINT_8")
        return QnnDatatype_t::QNN_DATATYPE_UINT_8;
    else if (dtype == "QNN_DATATYPE_UINT_16")
        return QnnDatatype_t::QNN_DATATYPE_UINT_16;
    else if (dtype == "QNN_DATATYPE_UINT_32")
        return QnnDatatype_t::QNN_DATATYPE_UINT_32;
    else if (dtype == "QNN_DATATYPE_UINT_64")
        return QnnDatatype_t::QNN_DATATYPE_UINT_64;

    else if (dtype == "QNN_DATATYPE_FLOAT_16")
        return QnnDatatype_t::QNN_DATATYPE_FLOAT_16;
    else if (dtype == "QNN_DATATYPE_FLOAT_32")
        return QnnDatatype_t::QNN_DATATYPE_FLOAT_32;

    else if (dtype == "QNN_DATATYPE_SFIXED_POINT_8")
        return QnnDatatype_t::QNN_DATATYPE_SFIXED_POINT_8;
    else if (dtype == "QNN_DATATYPE_SFIXED_POINT_16")
        return QnnDatatype_t::QNN_DATATYPE_SFIXED_POINT_16;
    else if (dtype == "QNN_DATATYPE_SFIXED_POINT_32")
        return QnnDatatype_t::QNN_DATATYPE_SFIXED_POINT_32;

    else if (dtype == "QNN_DATATYPE_UFIXED_POINT_8")
        return QnnDatatype_t::QNN_DATATYPE_UFIXED_POINT_8;
    else if (dtype == "QNN_DATATYPE_UFIXED_POINT_16")
        return QnnDatatype_t::QNN_DATATYPE_UFIXED_POINT_16;
    else if (dtype == "QNN_DATATYPE_UFIXED_POINT_32")
        return QnnDatatype_t::QNN_DATATYPE_UFIXED_POINT_32;

    else if (dtype == "QNN_DATATYPE_BOOL_8")
        return QnnDatatype_t::QNN_DATATYPE_BOOL_8;
    else if (dtype == "BACKEND_SPECIFIC")
        return QnnDatatype_t::QNN_DATATYPE_BACKEND_SPECIFIC;
    else
        return QnnDatatype_t::QNN_DATATYPE_UNDEFINED;
}

QnnRank_t strToRank(const std::string& rank)
{
    if (rank == "SCALAR")
        return QnnRank_t::QNN_SCALAR;
    else if (rank == "1D")
        return QnnRank_t::QNN_RANK_1;
    else if (rank == "2D")
        return QnnRank_t::QNN_RANK_2;
    else if (rank == "3D")
        return QnnRank_t::QNN_RANK_3;
    else if (rank == "4D")
        return QnnRank_t::QNN_RANK_4;
    else if (rank == "5D")
        return QnnRank_t::QNN_RANK_5;
    else if (rank == "ND")
        return QnnRank_t::QNN_RANK_N;
    else
        return QnnRank_t::QNN_RANK_INVALID;
}

bool Attribute::isMandatory() const
{
    return m_mandatory;
}

ConstraintType DatatypeConstraint::getConstraint()
{
    ConstraintType constraint;
    constraint.m_dtypeListConstraint = m_datatypes;
    return constraint;
}

ConstraintType RankConstraint::getConstraint()
{
    ConstraintType constraint;
    constraint.m_rankConstraint = m_rank;
    return constraint;
}

void OpConstraints::setIO(std::vector<Attribute> opInputs, std::vector<Attribute> opOutputs)
{
    m_inputs  = std::move(opInputs);
    m_outputs = std::move(opOutputs);
}

void OpConstraints::setParam(std::map<std::string, Attribute> opParams)
{
    m_parameters = std::move(opParams);
}

std::list<QnnDatatype_t> OpDefParser::extractDtypeIp(const std::string& ipName) const
{
    std::list<QnnDatatype_t> dtypeList = {};
    std::string ipNameCopy             = ipName;
    if (isSubstring(INF_ARG_INDICATOR, ipName) != std::string::npos)
    {
        std::string numStr = ipName.substr(3, isSubstring(INF_ARG_INDICATOR, ipName) - 3);
        ipNameCopy         = INPUT_NAME(stoi(numStr));
    }

    for (pugi::xml_node node = m_backendNode.child(INPUT); node; node = node.next_sibling(INPUT))
    {
        std::string name = node.child(ATTRIB_NAME).text().get();
        if (name == ipNameCopy)
        {
            for (pugi::xml_node dtypeNode = node.child(DATATYPE); dtypeNode;
                 dtypeNode                = dtypeNode.next_sibling(DATATYPE))
                dtypeList.emplace_back(strToDtype(dtypeNode.text().get()));
            break;
        }
    }

    return dtypeList;
}

std::list<QnnDatatype_t> OpDefParser::extractDtypeOut(const std::string& outName) const
{
    std::list<QnnDatatype_t> dtypeList = {};
    std::string outNameCopy            = outName;
    if (isSubstring(INF_ARG_INDICATOR, outName) != std::string::npos)
    {
        std::string numStr = outName.substr(4, isSubstring(INF_ARG_INDICATOR, outName) - 4);
        outNameCopy        = OUTPUT_NAME(stoi(numStr));
    }

    for (pugi::xml_node node = m_backendNode.child(OUTPUT); node; node = node.next_sibling(OUTPUT))
    {
        std::string name = node.child(ATTRIB_NAME).text().get();
        if (name == outNameCopy)
        {
            for (pugi::xml_node dtypeNode = node.child(DATATYPE); dtypeNode;
                 dtypeNode                = dtypeNode.next_sibling(DATATYPE))
                dtypeList.emplace_back(strToDtype(dtypeNode.text().get()));
            break;
        }
    }

    return dtypeList;
}

std::list<QnnDatatype_t> OpDefParser::extractDtypeParam(const std::string& paramName) const
{
    std::list<QnnDatatype_t> dtypeList = {};

    for (pugi::xml_node node = m_backendNode.child(PARAMETER); node; node = node.next_sibling(PARAMETER))
    {
        std::string name = node.child(ATTRIB_NAME).text().get();
        if (name == paramName)
        {
            for (pugi::xml_node dtypeNode = node.child(DATATYPE); dtypeNode;
                 dtypeNode                = dtypeNode.next_sibling(DATATYPE))
                dtypeList.emplace_back(strToDtype(dtypeNode.text().get()));
            break;
        }
    }

    return dtypeList;
}

void OpDefParser::parseIO(OpConstraints* constraints) const
{
    std::vector<Attribute> inputs;
    std::vector<Attribute> outputs;
    constraints->m_filterIndex = -1;

    for (pugi::xml_node node = m_masterNode.child(INPUT); node; node = node.next_sibling(INPUT))
    {
        Attribute curIp;
        curIp.m_multiFlag = false;

        std::string ipName = node.child(ATTRIB_NAME).text().get();

        std::list<QnnDatatype_t> datatype = {};
        for (pugi::xml_node dtypeNode = node.child(DATATYPE); dtypeNode; dtypeNode = dtypeNode.next_sibling(DATATYPE))
            datatype.emplace_back(strToDtype(dtypeNode.text().get()));

        bool backendSpecific =
            (find(datatype.begin(), datatype.end(), QnnDatatype_t::QNN_DATATYPE_BACKEND_SPECIFIC) != datatype.end());
        if (backendSpecific)
            curIp.m_datatypeConstraint.m_datatypes = extractDtypeIp(ipName);
        else
            curIp.m_datatypeConstraint.m_datatypes = datatype;

        pugi::xml_node shape          = node.child(SHAPE);
        std::string rank              = shape.child(RANK).text().get();
        curIp.m_rankConstraint.m_rank = strToRank(rank);

        std::string description;
        if (node.child(DESCRIPTION))
        {
            description = node.child(DESCRIPTION).child(CONTENT).text().get();
        }

        curIp.m_mandatory = strcmp(node.child(MANDATORY).text().get(), XML_TRUE) == 0;

        if (isSubstring(INF_ARG_INDICATOR, ipName) != std::string::npos)
        {
            std::string numStr = ipName.substr(3, isSubstring(INF_ARG_INDICATOR, ipName) - 3);
            ipName             = INPUT_NAME(stoi(numStr));
            curIp.m_multiFlag  = true;
        }
        if ((description == FILTERS) || (description == WEIGHTS) || (description == WEIGHTS_CAP))
        {
            std::string numIp          = ipName.substr(3, isSubstring(CLOSED_BRACKET, ipName) - 3);
            constraints->m_filterIndex = stoi(numIp);
        }
        inputs.push_back(curIp);
    }

    for (pugi::xml_node node = m_masterNode.child(OUTPUT); node; node = node.next_sibling(OUTPUT))
    {
        Attribute curOp;
        curOp.m_multiFlag = false;

        std::string outName = node.child(ATTRIB_NAME).text().get();

        std::list<QnnDatatype_t> datatype = {};
        for (pugi::xml_node dtypeNode = node.child(DATATYPE); dtypeNode; dtypeNode = dtypeNode.next_sibling(DATATYPE))
            datatype.emplace_back(strToDtype(dtypeNode.text().get()));

        bool backendSpecific =
            (find(datatype.begin(), datatype.end(), QnnDatatype_t::QNN_DATATYPE_BACKEND_SPECIFIC) != datatype.end());
        if (backendSpecific)
            curOp.m_datatypeConstraint.m_datatypes = extractDtypeOut(outName);
        else
            curOp.m_datatypeConstraint.m_datatypes = datatype;

        pugi::xml_node shape          = node.child(SHAPE);
        std::string rank              = shape.child(RANK).text().get();
        curOp.m_rankConstraint.m_rank = strToRank(rank);

        curOp.m_mandatory = strcmp(node.child(MANDATORY).text().get(), XML_TRUE) == 0;

        if (isSubstring(INF_ARG_INDICATOR, outName) != std::string::npos)
        {
            std::string numStr = outName.substr(4, isSubstring(INF_ARG_INDICATOR, outName) - 4);
            outName            = OUTPUT_NAME(stoi(numStr));
            curOp.m_multiFlag  = true;
        }

        outputs.push_back(curOp);
    }

    constraints->setIO(inputs, outputs);
}

void OpDefParser::parseParams(OpConstraints* constraints) const
{
    std::map<std::string, Attribute> parameters;

    for (pugi::xml_node node = m_masterNode.child(PARAMETER); node; node = node.next_sibling(PARAMETER))
    {
        Attribute curParam;

        std::string paramName = node.child(ATTRIB_NAME).text().get();

        std::list<QnnDatatype_t> datatype = {};
        for (pugi::xml_node dtypeNode = node.child(DATATYPE); dtypeNode; dtypeNode = dtypeNode.next_sibling(DATATYPE))
            datatype.emplace_back(strToDtype(dtypeNode.text().get()));

        bool backendSpecific =
            (find(datatype.begin(), datatype.end(), QnnDatatype_t::QNN_DATATYPE_BACKEND_SPECIFIC) != datatype.end());
        if (backendSpecific)
            curParam.m_datatypeConstraint.m_datatypes = extractDtypeParam(paramName);
        else
            curParam.m_datatypeConstraint.m_datatypes = datatype;

        pugi::xml_node shape             = node.child(SHAPE);
        std::string rank                 = shape.child(RANK).text().get();
        curParam.m_rankConstraint.m_rank = strToRank(rank);

        curParam.m_mandatory = strcmp(node.child(MANDATORY).text().get(), XML_TRUE) == 0;

        parameters[paramName] = curParam;
    }

    constraints->setParam(parameters);
}

void ModelOpDefParser::populate()
{
    OpDefParser parser;
    OpConstraints newConstraints;
    pugi::xml_document masterDoc;
    pugi::xml_document backendDoc;
    std::vector<std::string> missingNodesInMasterOpDef;
    std::vector<std::string> missingNodesInBackendOpDef;

    pugi::xml_parse_result masterResult = masterDoc.load_file(m_masterPath.c_str());
    if (!masterResult)
        DEBUG_LOG_XML_FILE_ERROR(m_masterPath, masterResult.description())

    pugi::xml_parse_result backendResult = backendDoc.load_file(m_backendPath.c_str());
    if (!backendResult)
        DEBUG_LOG_XML_FILE_ERROR(m_backendPath, backendResult.description())

    for (auto i = m_opList.begin(); i != m_opList.end(); ++i)
    {
        std::string opName          = i->data();
        std::string lowercaseOpName = transformLower(opName);
        pugi::xml_node masterRoot;
        pugi::xml_node backendRoot;
        for (pugi::xml_node node = masterDoc.child(MASTER_OPDEF_LIST).child(MASTER_OPDEF); node;
             node                = node.next_sibling(MASTER_OPDEF))
        {
            if (strcmp(transformLower(node.child(ATTRIB_NAME).text().get()).c_str(), lowercaseOpName.c_str()) == 0)
            {
                masterRoot = node;
                break;
            }
        }
        for (pugi::xml_node node = backendDoc.child(SUPPLEMENTAL_OPDEF_LIST).child(SUPPLEMENTAL_OPDEF); node;
             node                = node.next_sibling(SUPPLEMENTAL_OPDEF))
        {
            if (strcmp(transformLower(node.child(ATTRIB_NAME).text().get()).c_str(), lowercaseOpName.c_str()) == 0)
            {
                backendRoot = node;
                break;
            }
        }

        if (!masterRoot)
            missingNodesInMasterOpDef.push_back(opName);
        if (!backendRoot)
            missingNodesInBackendOpDef.push_back(opName);

        parser.m_masterNode  = masterRoot;
        parser.m_backendNode = backendRoot;

        parser.parseIO(&newConstraints);
        parser.parseParams(&newConstraints);
        m_modelOpConstraints[opName] = newConstraints;
    }

    if (!missingNodesInBackendOpDef.empty())
    {
        std::cout << "Op info. not found for these ops: ";
        for (auto x: missingNodesInBackendOpDef)
            std::cout << x << " ";
        std::cout << "\n";
    }
}

ModelOpDefParser::ModelOpDefParser(std::string mPath, std::string bPath, std::list<std::string> opList)
{
    m_masterPath  = std::move(mPath);
    m_backendPath = std::move(bPath);
    m_opList      = std::move(opList);
    populate();
}

std::string compareAndGetOpName(const std::string& opName, std::list<std::string> m_opList)
{
    std::string opNameLower = transformLower(opName);
    std::string opNameRetrieved("");

    for (const auto& s: m_opList)
    {
        if (transformLower(s) == opNameLower)
        {
            opNameRetrieved = s;
            break;
        }
    }
    return opNameRetrieved;
}

int ModelOpDefParser::getFiltersIndex(const std::string& opName)
{
    std::string opNameRetrieved = compareAndGetOpName(opName, m_opList);

    if (opNameRetrieved.empty())
    {
        std::string error = DEBUG_LOG_INVALID_OP(opName);
        throw std::invalid_argument(error);
    }
    OpConstraints opConstraints = m_modelOpConstraints[opNameRetrieved];
    return opConstraints.m_filterIndex;
}

std::map<std::string, int> ModelOpDefParser::getSize(const std::string& opName)
{
    std::string opNameRetrieved = compareAndGetOpName(opName, m_opList);

    if (opNameRetrieved.empty())
    {
        std::string error = DEBUG_LOG_INVALID_OP(opName);
        throw std::invalid_argument(error);
    }
    std::map<std::string, int> sizes;
    OpConstraints opConstraints = m_modelOpConstraints[opNameRetrieved];
    sizes[INPUT_SIZE]           = int(opConstraints.m_inputs.size());
    sizes[OUTPUT_SIZE]          = int(opConstraints.m_outputs.size());
    sizes[PARAM_SIZE]           = int(opConstraints.m_parameters.size());

    return sizes;
}

std::list<QnnDatatype_t> ModelOpDefParser::getInputDataType(const std::string& opName, int attribNum)
{
    std::string opNameRetrieved = compareAndGetOpName(opName, m_opList);

    if (opNameRetrieved.empty())
    {
        std::string error = DEBUG_LOG_INVALID_OP(opName);
        throw std::invalid_argument(error);
    }
    OpConstraints opConstraints = m_modelOpConstraints[opNameRetrieved];

    if (opConstraints.m_inputs.size() - 1 < attribNum)
    {
        std::string error =
            DEBUG_LOG_INVALID_INPUT(opName, opConstraints.m_inputs.size()) + " in getInputDataType() function.";
        throw std::invalid_argument(error);
    }
    std::list<QnnDatatype_t> validDtypes =
        opConstraints.m_inputs[attribNum].m_datatypeConstraint.getConstraint().m_dtypeListConstraint;
    return validDtypes;
}

std::list<QnnDatatype_t> ModelOpDefParser::getOutputDataType(const std::string& opName, int attribNum)
{
    std::string opNameRetrieved = compareAndGetOpName(opName, m_opList);

    if (opNameRetrieved.empty())
    {
        std::string error = DEBUG_LOG_INVALID_OP(opName);
        throw std::invalid_argument(error);
    }
    OpConstraints opConstraints = m_modelOpConstraints[opNameRetrieved];

    if (opConstraints.m_outputs.size() - 1 < attribNum)
    {
        std::string error =
            DEBUG_LOG_INVALID_OUTPUT(opName, opConstraints.m_outputs.size()) + " in getOutputDataType() function.";
        throw std::invalid_argument(error);
    }
    std::list<QnnDatatype_t> validDtypes =
        opConstraints.m_outputs[attribNum].m_datatypeConstraint.getConstraint().m_dtypeListConstraint;
    return validDtypes;
}

std::list<QnnDatatype_t> ModelOpDefParser::getParamDataType(const std::string& opName, const std::string& attribName)
{
    std::string opNameRetrieved = compareAndGetOpName(opName, m_opList);

    if (opNameRetrieved.empty())
    {
        std::string error = DEBUG_LOG_INVALID_OP(opName);
        throw std::invalid_argument(error);
    }

    OpConstraints opConstraints = m_modelOpConstraints[opNameRetrieved];
    std::list<QnnDatatype_t> validDtypes =
        opConstraints.m_parameters[attribName].m_datatypeConstraint.getConstraint().m_dtypeListConstraint;

    if (validDtypes.empty())
    {
        std::string error = DEBUG_LOG_INVALID_PARAM(opName, attribName);
        throw std::invalid_argument(error);
    }
    return validDtypes;
}

QnnRank_t ModelOpDefParser::getInputRank(const std::string& opName, int attribNum)
{
    std::string opNameRetrieved = compareAndGetOpName(opName, m_opList);

    if (opNameRetrieved.empty())
    {
        std::string error = DEBUG_LOG_INVALID_OP(opName);
        throw std::invalid_argument(error);
    }
    OpConstraints opConstraints = m_modelOpConstraints[opNameRetrieved];

    if (opConstraints.m_inputs.size() - 1 < attribNum)
    {
        std::string error =
            DEBUG_LOG_INVALID_INPUT(opName, opConstraints.m_inputs.size()) + " in getInputRank() function.";
        throw std::invalid_argument(error);
    }
    QnnRank_t validRank = opConstraints.m_inputs[attribNum].m_rankConstraint.getConstraint().m_rankConstraint;
    return validRank;
}

QnnRank_t ModelOpDefParser::getOutputRank(const std::string& opName, int attribNum)
{
    std::string opNameRetrieved = compareAndGetOpName(opName, m_opList);

    if (opNameRetrieved.empty())
    {
        std::string error = DEBUG_LOG_INVALID_OP(opName);
        throw std::invalid_argument(error);
    }
    OpConstraints opConstraints = m_modelOpConstraints[opNameRetrieved];

    if (opConstraints.m_outputs.size() - 1 < attribNum)
    {
        std::string error =
            DEBUG_LOG_INVALID_OUTPUT(opName, opConstraints.m_outputs.size()) + " in getOutputRank() function.";
        throw std::invalid_argument(error);
    }
    QnnRank_t validRank = opConstraints.m_outputs[attribNum].m_rankConstraint.getConstraint().m_rankConstraint;
    return validRank;
}

QnnRank_t ModelOpDefParser::getParamRank(const std::string& opName, const std::string& attribName)
{
    std::string opNameRetrieved = compareAndGetOpName(opName, m_opList);

    if (opNameRetrieved.empty())
    {
        std::string error = DEBUG_LOG_INVALID_OP(opName);
        throw std::invalid_argument(error);
    }
    OpConstraints opConstraints = m_modelOpConstraints[opNameRetrieved];
    QnnRank_t validRank = opConstraints.m_parameters[attribName].m_rankConstraint.getConstraint().m_rankConstraint;

    if (validRank == QnnRank_t::QNN_RANK_INVALID)
    {
        std::string error = DEBUG_LOG_INVALID_PARAM(opName, attribName);
        throw std::invalid_argument(error);
    }
    return validRank;
}

bool ModelOpDefParser::getInputMultiFlag(const std::string& opName, int attribNum)
{
    std::string opNameRetrieved = compareAndGetOpName(opName, m_opList);

    if (opNameRetrieved.empty())
    {
        std::string error = DEBUG_LOG_INVALID_OP(opName);
        throw std::invalid_argument(error);
    }
    OpConstraints opConstraints = m_modelOpConstraints[opNameRetrieved];

    if (opConstraints.m_inputs.size() - 1 < attribNum)
    {
        std::string error =
            DEBUG_LOG_INVALID_INPUT(opName, opConstraints.m_inputs.size()) + " in getInputDataType() function.";
        throw std::invalid_argument(error);
    }
    bool flag = opConstraints.m_inputs[attribNum].m_multiFlag;

    return flag;
}

bool ModelOpDefParser::getOutputMultiFlag(const std::string& opName, int attribNum)
{
    std::string opNameRetrieved = compareAndGetOpName(opName, m_opList);

    if (opNameRetrieved.empty())
    {
        std::string error = DEBUG_LOG_INVALID_OP(opName);
        throw std::invalid_argument(error);
    }

    OpConstraints opConstraints = m_modelOpConstraints[opNameRetrieved];

    if (opConstraints.m_outputs.size() - 1 < attribNum)
    {
        std::string error =
            DEBUG_LOG_INVALID_OUTPUT(opName, opConstraints.m_outputs.size()) + " in getOutputDataType() function.";
        throw std::invalid_argument(error);
    }
    bool flag = opConstraints.m_outputs[attribNum].m_multiFlag;

    return flag;
}

size_t isSubstring(std::string shortString, std::string longString)
{
    return longString.find(shortString);
}

std::string transformLower(const std::string& opName)
{
    std::string lowerString = opName;
    std::transform(lowerString.begin(), lowerString.end(), lowerString.begin(), ::tolower);
    return lowerString;
}
