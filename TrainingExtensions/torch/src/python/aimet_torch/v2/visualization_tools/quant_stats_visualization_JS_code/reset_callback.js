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

// Resetting the plot ranges
plot.y_range.start = limits_source.data['ymin'][0]*1.05;
plot.y_range.end = limits_source.data['ymax'][0]*1.05;
plot.x_range.start = limits_source.data['xmin'][0];
plot.x_range.end = limits_source.data['xmax'][0];

// Resetting the input widget values
ymax_input.value = limits_source.data['ymax'][0].toString();
ymin_input.value = limits_source.data['ymin'][0].toString();
maxclip_input.value = limits_source.data['maxclip'][0].toString();
minclip_input.value = limits_source.data['minclip'][0].toString();

data_source.data['marker_yminlist'] = data_source.data['minlist'].map(t => findMax(t, limits_source.data['ymin'][0]));
data_source.data['marker_ymaxlist'] = data_source.data['maxlist'].map(t => findMin(t, limits_source.data['ymax'][0]));

min_thresh_filter.booleans = data_source.data['minlist'].map(t => t <= limits_source.data['minclip'][0]);
max_thresh_filter.booleans = data_source.data['maxlist'].map(t => t >= limits_source.data['maxclip'][0]);

name_filter.booleans = Array(data_source.data['idx'].length).fill(true);
name_input.value = "";

select.value = "Min | Max";
var view = select.value;
let table_booleans = process_table_view(view, name_filter, min_thresh_filter, max_thresh_filter);

for (let i = 0; i < data_source.data['idx'].length; i++) {
    if (table_booleans[i] == true) {
        for (let j = 0; j < table_columns.length; j++) {
            table_data_source.data[table_columns[j]].push(data_source.data[table_columns[j]][i]);
        }
    }
}

table_data_source.selected.indices = [];
data_source.data["selected"] = Array(data_source.data["idx"].length).fill(false);

selected_data_source.data["floor"] = [];
selected_data_source.data["ceil"] = [];

for (let j = 0; j < selection_columns.length; j++) {
    selected_data_source.data[selection_columns[j]] = [];
}

if (mode == "advanced") {
    boxplot.x_range.factors = [];
    boxplot.width = boxplot_unit_width * 5;
}

// Emitting the changes made to ColumnDataSources
limits_source.change.emit();
data_source.change.emit();
table_data_source.change.emit();
selected_data_source.change.emit();