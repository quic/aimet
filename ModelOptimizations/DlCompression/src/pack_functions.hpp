//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2016-2017, Qualcomm Innovation Center, Inc. All rights reserved.
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

#ifndef PACK_FUNCTIONS_HPP_
#define PACK_FUNCTIONS_HPP_

#include <vector>

namespace DlCompression
{
/**
 * @brief Pack data into compressed format.
 * @param unpackedData The data to be packed.
 * @param packedData Compute the packed byte array such that every bit is used
 * to represent the original data (apart from trailing padding bits at the very
 * end of the vector).
 * @param bw Each item in 'unpackedData' gets packed using this bit-width.
 *
 * Note this function does support bit-widths which are not integer power of 2.
 * The following bit-widths are supported: 1, 2, ..., 7, 8, 16, 32.
 */
void packData(const std::vector<unsigned int>& unpackedData, std::vector<uint8_t>& packedData, unsigned int bw);

/**
 * @brief Unpack data from compressed format.
 * @param packedData The packed byte array such that every bit is used
 * to represent the original data (apart from trailing padding bits at the very
 * end of the vector).
 * @param packedDataCnt The number of items in 'packedData'.
 * @param unpackedData Unpack the data.
 * @param bw Each item in 'packedData' is represented using this bit-width.
 *
 * Note this function does support bit-widths which are not integer power of 2.
 * The following bit-widths are supported: 1, 2, ..., 7, 8, 16, 32.
 */
void unpackData(const std::vector<uint8_t>& packedData, unsigned int packedDataCnt,
                std::vector<unsigned int>& unpackedData, unsigned int bw);

}   // End of namespace DlCompression

#endif   // PACK_FUNCTIONS_HPP_
