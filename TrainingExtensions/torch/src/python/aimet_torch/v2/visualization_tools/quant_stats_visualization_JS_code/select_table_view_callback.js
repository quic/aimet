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


const source_data = data_source.data;
const idx = source_data['idx'];
const minlist = source_data['minlist'];
const maxlist = source_data['maxlist'];
const namelist = source_data['namelist'];

let table_booleans;
const table_idx = [];
const table_namelist = [];
const table_minlist = [];
const table_maxlist = [];

var view = select.value;
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

for (let i = 0; i < idx.length; i++) {
    if (table_booleans[i] == true) {
        table_idx.push(idx[i]);
        table_namelist.push(namelist[i]);
        table_minlist.push(minlist[i]);
        table_maxlist.push(maxlist[i]);
    }
}

table_data_source.data["idx"] = table_idx;
table_data_source.data["namelist"] = table_namelist;
table_data_source.data["minlist"] = table_minlist;
table_data_source.data["maxlist"] = table_maxlist;

table_data_source.change.emit();

const selected_indices = [];
var layer_idx;
for (let i = 0; i < table_idx.length; i++) {
    layer_idx = table_idx[i];
    if (data_source.data["selected"][layer_idx] == true) {
        selected_indices.push(i);
    }
}

table_data_source.selected.indices = selected_indices;

table_data_source.change.emit();

// Force redraw of table by making an inert change to table properties
table.name = "placeholder_1";
table.name = "placeholder_0";