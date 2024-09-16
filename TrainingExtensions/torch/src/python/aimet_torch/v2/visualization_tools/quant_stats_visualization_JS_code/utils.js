// =============================================================================
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2023-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
// =============================================================================


function booleanAnd(arr1, arr2) {
    return arr1.map((value, index) => value && arr2[index]);
}

function booleanOr(arr1, arr2) {
    return arr1.map((value, index) => value || arr2[index]);
}

function findMin(a, b) {
    if (a<=b) {
        return a;
    }
    return b;
}

function findMax(a, b) {
    if (a>=b) {
        return a;
    }
    return b;
}

function arrayMin(arr) {
    var min;
    arr.forEach((val, index) => {
        if (index==0) {
            min = val;
        } else {
            min = findMin(val, min);
        }
    })
    return min;
}

function arrayMax(arr) {
    var max;
    arr.forEach((val, index) => {
        if (index==0) {
            max = val;
        } else {
            max = findMax(val, max);
        }
    })
    return max;
}

function process_table_view(view, name_filter, min_thresh_filter, max_thresh_filter) {
    let table_booleans;

    if (view == "All") {
        table_booleans = name_filter.booleans;
    } else if (view == "Min") {
        table_booleans = booleanAnd(name_filter.booleans, min_thresh_filter.booleans);
    } else if (view == "Max") {
        table_booleans = booleanAnd(name_filter.booleans, max_thresh_filter.booleans);
    } else if (view == "Min | Max") {
        table_booleans = booleanAnd(name_filter.booleans, booleanOr(min_thresh_filter.booleans, max_thresh_filter.booleans));
    } else if (view == "Min & Max") {
        table_booleans = booleanAnd(name_filter.booleans, booleanAnd(min_thresh_filter.booleans, max_thresh_filter.booleans));
    }

    return table_booleans
}
