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


// Resetting the limits source with default values
limits_source.data['ymax'] = default_values_source.data['default_ymax'];
limits_source.data['ymin'] = default_values_source.data['default_ymin'];
limits_source.data['xmax'] = default_values_source.data['default_xmax'];
limits_source.data['xmin'] = default_values_source.data['default_xmin'];
limits_source.data['maxclip'] = default_values_source.data['default_maxclip'];
limits_source.data['minclip'] = default_values_source.data['default_minclip'];
const limits_data = limits_source.data;

// Resetting the plot ranges
plot.y_range.start = limits_data['ymin'][0]*1.05;
plot.y_range.end = limits_data['ymax'][0]*1.05;
plot.x_range.start = limits_data['xmin'][0];
plot.x_range.end = limits_data['xmax'][0];

// Resetting the input widget values
ymax_input.value = limits_data['ymax'][0].toString();
ymin_input.value = limits_data['ymin'][0].toString();
maxclip_input.value = limits_data['maxclip'][0].toString();
minclip_input.value = limits_data['minclip'][0].toString();

const source_data = data_source.data;
const idx = source_data['idx'];
const minlist = source_data['minlist'];
const maxlist = source_data['maxlist'];
const namelist = source_data['namelist'];

source_data['marker_yminlist'] = source_data['minlist'].map(t => findMax(t, limits_data['ymin'][0]));
source_data['marker_ymaxlist'] = source_data['maxlist'].map(t => findMin(t, limits_data['ymax'][0]));

min_thresh_filter.booleans = minlist.map(t => t <= limits_data['minclip'][0]);
max_thresh_filter.booleans = maxlist.map(t => t >= limits_data['maxclip'][0]);

name_filter.booleans = Array(idx.length).fill(true);
name_input.value = "";

let table_booleans;
const table_idx = [];
const table_namelist = [];
const table_minlist = [];
const table_maxlist = [];

select.value = "Min | Max";
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

table_data_source.selected.indices = [];
data_source.data["selected"] = Array(data_source.data["idx"].length).fill(false);

selected_data_source.data["floor"] = [];
selected_data_source.data["ceil"] = [];

for (let j = 0; j < selection_columns.length; j++) {
    selected_data_source.data[selection_columns[j]] = [];
}

if (mode == "advanced") {
    boxplot.x_range.factors = [];
}

// Emitting the changes made to ColumnDataSources
limits_source.change.emit();
data_source.change.emit();
table_data_source.change.emit();
selected_data_source.change.emit();