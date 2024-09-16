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


// Reading values from input widgets and setting plot y axis range
const limits_data = limits_source.data;
limits_data['ymax'] = [parseFloat(ymax_input.value)];
limits_data['ymin'] = [parseFloat(ymin_input.value)];
plot.y_range.start = limits_data['ymin'][0]*1.05;
plot.y_range.end = limits_data['ymax'][0]*1.05;
limits_data['maxclip'] = [parseFloat(maxclip_input.value)];
limits_data['minclip'] = [parseFloat(minclip_input.value)];

const source_data = data_source.data;
const idx = source_data['idx'];
const minlist = source_data['minlist'];
const maxlist = source_data['maxlist'];
const namelist = source_data['namelist'];
source_data['marker_yminlist'] = source_data['minlist'].map(t => findMax(t, limits_data['ymin'][0]));
source_data['marker_ymaxlist'] = source_data['maxlist'].map(t => findMin(t, limits_data['ymax'][0]));

// Updating the filters for finding layers that cross the min or max thresholds
min_thresh_filter.booleans = minlist.map(t => t <= limits_data['minclip'][0]);
max_thresh_filter.booleans = maxlist.map(t => t >= limits_data['maxclip'][0]);

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

selected_data_source.data["floor"].push(limits_source.data['ymin'][0]*1.05);
selected_data_source.data["ceil"].push(limits_source.data['ymax'][0]*1.05);

// Emitting the changes made to ColumnDataSources
limits_source.change.emit();
data_source.change.emit();
table_data_source.change.emit();
selected_data_source.change.emit();