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

#include "RegisterCustomOps.h"

static const char* c_OpDomain    = "aimet.customop.cpu";
static const char* c_OpDomainGPU = "aimet.customop.cuda";


OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api)
{
    OrtCustomOpDomain* domain = nullptr;
    const OrtApi* ortApi      = api->GetApi(ORT_API_VERSION);

    if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain))
    {
        return status;
    }

    AddOrtCustomOpDomainToContainer(domain, ortApi);
    static const QcQuantizeOp c_QcQuantizeOp;
    if (auto status = ortApi->CustomOpDomain_Add(domain, &c_QcQuantizeOp))
    {
        return status;
    }

#ifdef ONNX_CUDA
    OrtCustomOpDomain* cuda_domain = nullptr;
    if (auto status = ortApi->CreateCustomOpDomain(c_OpDomainGPU, &cuda_domain))
    {
        return status;
    }

    AddOrtCustomOpDomainToContainer(cuda_domain, ortApi);
    static const QcQuantizeOpGPU c_QcQuantizeOpGPU;
    if (auto status = ortApi->CustomOpDomain_Add(cuda_domain, &c_QcQuantizeOpGPU))
    {
        return status;
    }
    ortApi->AddCustomOpDomain(options, cuda_domain);
#endif

    return ortApi->AddCustomOpDomain(options, domain);
}
