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
limits_source.data['ymax'] = [parseFloat(ymax_input.value)];
limits_source.data['ymin'] = [parseFloat(ymin_input.value)];
plot.y_range.start = limits_source.data['ymin'][0]*1.05;
plot.y_range.end = limits_source.data['ymax'][0]*1.05;
limits_source.data['maxclip'] = [parseFloat(maxclip_input.value)];
limits_source.data['minclip'] = [parseFloat(minclip_input.value)];

data_source.data['marker_yminlist'] = data_source.data['minlist'].map(t => findMax(t, limits_source.data['ymin'][0]));
data_source.data['marker_ymaxlist'] = data_source.data['maxlist'].map(t => findMin(t, limits_source.data['ymax'][0]));

// Updating the filters for finding layers that cross the min or max thresholds
min_thresh_filter.booleans = data_source.data['minlist'].map(t => t <= limits_source.data['minclip'][0]);
max_thresh_filter.booleans = data_source.data['maxlist'].map(t => t >= limits_source.data['maxclip'][0]);

var view = select.value;
let table_booleans = process_table_view(view, name_filter, min_thresh_filter, max_thresh_filter);

for (let j = 0; j < table_columns.length; j++) {
    table_data_source.data[table_columns[j]] = [];
}

for (let i = 0; i < data_source.data['idx'].length; i++) {
    if (table_booleans[i] == true) {
        for (let j = 0; j < table_columns.length; j++) {
            table_data_source.data[table_columns[j]].push(data_source.data[table_columns[j]][i]);
        }
    }
}

selected_data_source.data["floor"].push(limits_source.data['ymin'][0]*1.05);
selected_data_source.data["ceil"].push(limits_source.data['ymax'][0]*1.05);

// Emitting the changes made to ColumnDataSources
limits_source.change.emit();
data_source.change.emit();
table_data_source.change.emit();
selected_data_source.change.emit();