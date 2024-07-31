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

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <vector>
#include <cmath>
#include <mutex>
#include <system_error>

#include "QcQuantizeOp.h"
#include "AimetOpUtils.h"
#include "onnxruntime_lite_custom_op.h"

static const char* c_OpDomain    = "aimet.customop.cpu";
static const char* c_OpDomainGPU = "aimet.customop.cuda";

// These definitions are missing from the provided header files in onnxruntime but are used in the provided examples
#define ORT_TRY if (true)
#define ORT_CATCH(x) else if (false)
#define ORT_HANDLE_EXCEPTION(func)

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api)
{

    Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);
    OrtStatus* result = nullptr;
    ORT_TRY
    {
        Ort::CustomOpDomain domain {c_OpDomain};
        RegisterOps(domain);

        Ort::UnownedSessionOptions session_options(options);
        session_options.Add(domain);
        AddOrtCustomOpDomainToContainer(std::move(domain));

#ifdef ONNX_CUDA
        // This is for backward compatibility, in the new custom OP API we do not need separate domains for cpu/gpu
        Ort::CustomOpDomain cuda_domain {c_OpDomainGPU};
        RegisterOps(cuda_domain);
        session_options.Add(cuda_domain);
        AddOrtCustomOpDomainToContainer(std::move(cuda_domain));
#endif
    }
    ORT_CATCH(const std::exception& e)
    {
        ORT_HANDLE_EXCEPTION([&]() {
            Ort::Status status{e};
            result = status.release();
        })
    }

    return result;

}
