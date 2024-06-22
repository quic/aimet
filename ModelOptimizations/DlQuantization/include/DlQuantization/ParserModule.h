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

#ifndef PYBINDEX_PARSERMODULE_H
#define PYBINDEX_PARSERMODULE_H

#include <bits/stdc++.h>

#include "pugixml.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "DlQuantization/XmlTypes.h"

size_t isSubstring(std::string shortString, std::string longString);

std::string transformLower(const std::string& opName);

QnnDatatype_t strToDtype(const std::string& dtype);

QnnRank_t strToRank(const std::string& rank);

class ConstraintType
{
public:
    std::list<QnnDatatype_t> m_dtypeListConstraint;
    QnnRank_t m_rankConstraint;
};

class Constraint
{
public:
    virtual ConstraintType getConstraint()
    {
        return {};
    }
};

class DatatypeConstraint : public Constraint
{
public:
    std::list<QnnDatatype_t> m_datatypes;

    ConstraintType getConstraint() override;
};

class RankConstraint : public Constraint
{
public:
    QnnRank_t m_rank {};

    ConstraintType getConstraint() override;
};

class Attribute
{
public:
    bool m_mandatory {};
    RankConstraint m_rankConstraint;
    DatatypeConstraint m_datatypeConstraint;
    bool m_multiFlag {};

    [[nodiscard]] bool isMandatory() const;
};

class OpConstraints
{
public:
    std::string m_opName;
    std::vector<Attribute> m_inputs;
    std::vector<Attribute> m_outputs;
    std::map<std::string, Attribute> m_parameters;
    int m_filterIndex;

    void setIO(std::vector<Attribute> opInputs, std::vector<Attribute> opOutputs);

    void setParam(std::map<std::string, Attribute> opParams);
};

class OpDefParser
{
public:
    pugi::xml_node m_masterNode;
    pugi::xml_node m_backendNode;

    [[nodiscard]] std::list<QnnDatatype_t> extractDtypeIp(const std::string& ipName) const;

    [[nodiscard]] std::list<QnnDatatype_t> extractDtypeOut(const std::string& outName) const;

    [[nodiscard]] std::list<QnnDatatype_t> extractDtypeParam(const std::string& paramName) const;

    void parseIO(OpConstraints* constraints) const;

    void parseParams(OpConstraints* constraints) const;
};

class ModelOpDefParser
{
public:
    std::list<std::string> m_opList;
    std::string m_masterPath {};
    std::string m_backendPath {};
    std::map<std::string, OpConstraints> m_modelOpConstraints;
    std::map <std::string, std::list<OpConstraints>> m_modelOpConstraints_v2;
    void populate();

    std::list<std::string> getSupportedOpsInBackend();

    ModelOpDefParser(std::string mPath, std::string bPath);

    int getFiltersIndex(const std::string& opName);

    std::map<std::string, int> getSize(const std::string& opName);

    std::list<QnnDatatype_t> getInputDataType(const std::string& opName, int attribNum);

    std::list<QnnDatatype_t> getOutputDataType(const std::string& opName, int attribNum);

    std::list<QnnDatatype_t> getParamDataType(const std::string& opName, const std::string& attribName);

    QnnRank_t getInputRank(const std::string& opName, int attribNum);

    QnnRank_t getOutputRank(const std::string& opName, int attribNum);

    QnnRank_t getParamRank(const std::string& opName, const std::string& attribName);

    bool getInputMultiFlag(const std::string& opName, int attribNum);

    bool getOutputMultiFlag(const std::string& opName, int attribNum);

    std::list <std::map<std::string, int>> getSizeList(const std::string &opName);

    std::list <std::list <QnnDatatype_t>> getInputDataTypeList(const std::string &opName, int attribNum);

    std::list <std::list <QnnDatatype_t>> getOutputDataTypeList(const std::string &opName, int attribNum);
};

#endif   // PYBINDEX_PARSERMODULE_H