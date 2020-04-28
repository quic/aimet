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

#include <cmath>
#include <stdexcept>
#include <stdint.h>
#include <string.h>
#include <vector>

namespace DlCompression
{
using namespace std;

void packData(const vector<unsigned int>& unpacked, vector<uint8_t>& packed, unsigned int bw)
{
    packed.resize(std::ceil(unpacked.size() * bw / 8.0), 0);
    if (bw > 0 && bw < 8)
    {
        for (unsigned int i = 0; i < unpacked.size(); ++i)
        {
            uint8_t* byteId     = (uint8_t*) (packed.data()) + bw * i / 8;
            unsigned int offset = bw * i % 8;
            uint8_t value       = unpacked[i];
            // Or-in 'unpacked[i]' into 'packed[bw*i/8]'.
            // Note if we are on a byte boundary, the MSB's are ignored.
            *byteId |= (value << offset);
            // If we are on a byte boundary:
            // Or-in MSBs of 'unpacked[i]' into 'packed[bw*i/8+1]'.
            if (offset + bw > 8)
            {
                *(byteId + 1) |= (value >> (-offset + 8));
            }
        }
    }

    else if (bw == 8)
    {
        std::copy(unpacked.begin(), unpacked.end(), packed.begin());
    }

    else if (bw == 16)
    {
        // Cast all 32-bit values to 16-bit values
        vector<uint16_t> val16(unpacked.begin(), unpacked.end());
        // Copy the result into 'packed'
        memcpy(packed.data(), val16.data(), packed.size());
    }

    else if (bw == 32)
    {
        memcpy(packed.data(), unpacked.data(), packed.size());
    }

    else
    {
        throw std::runtime_error("Invalid bit-width for packing. Valid bit-widths: "
                                 "1, 2, ..., 7, 8, 16, 32.");
    }
}

void unpackData(const vector<uint8_t>& packed, unsigned int packedCnt, vector<unsigned int>& unpacked, unsigned int bw)
{
    // Check the size of 'packed'
    if (std::ceil(packedCnt * bw / 8.0) != packed.size())
    {
        throw runtime_error("Size of packed vector doesn't match the bit-width and "
                            "number of packed data points.");
    }
    unpacked.resize(packedCnt, 0);
    if (bw > 0 && bw < 8)
    {
        for (unsigned int i = 0; i < packedCnt; ++i)
        {
            const uint8_t* byteId = packed.data() + bw * i / 8;
            unsigned int offset   = bw * i % 8;
            // Grab LSB's of i-th packed value from byteId[0]
            uint8_t tmp = *byteId >> offset & ((1 << bw) - 1);
            // If we are on a byte boundary, grab the remaining MSB's from byteId[1]
            if (offset + bw > 8)
            {
                uint8_t mask = (1 << (bw + offset - 8)) - 1;
                tmp |= (*(byteId + 1) & mask) << (8 - offset);
            }
            unpacked[i] = tmp;
        }
    }

    else if (bw == 8)
    {
        unpacked.assign(packed.begin(), packed.end());
    }

    else if (bw == 16)
    {
        // Copy packed data into 16-bit vector, byte by byte
        vector<uint16_t> val16(packedCnt);
        memcpy(val16.data(), packed.data(), packedCnt * 2);
        // Cast the result to 32-bit values
        unpacked.assign(val16.begin(), val16.end());
    }

    else if (bw == 32)
    {
        memcpy(unpacked.data(), packed.data(), packed.size());
    }

    else
    {
        throw std::runtime_error("Invalid bit-width for packing. Valid bit-widths: "
                                 "1, 2, ..., 7, 8, 16, 32.");
    }
}

}   // End of namespace DlCompression
