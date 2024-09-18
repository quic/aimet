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


table_data_source.data["idx"].forEach(i => {
    data_source.data["selected"][i] = false;
});
table_data_source.selected.indices.forEach(i => {
    let layer_idx = table_data_source.data["idx"][i];
    data_source.data["selected"][layer_idx] = true;
});
data_source.change.emit();

selected_data_source.data["floor"] = [];
selected_data_source.data["ceil"] = [];

for (let j = 0; j < selection_columns.length; j++) {
    selected_data_source.data[selection_columns[j]] = [];
}

selected_data_source.change.emit();

data_source.data["selected"].forEach((bool,index) => {
    if (bool==true) {
        selected_data_source.data["floor"].push(limits_source.data['ymin'][0]*1.05);
        selected_data_source.data["ceil"].push(limits_source.data['ymax'][0]*1.05);
        for (let j = 0; j < selection_columns.length; j++) {
            selected_data_source.data[selection_columns[j]].push(data_source.data[selection_columns[j]][index]);
        }
    }
})

if (mode=="advanced") {
    boxplot.x_range.factors = selected_data_source.data["stridx"];
    let box_min = arrayMin(selected_data_source.data["minlist"]) * 1.2;
    let box_max = arrayMax(selected_data_source.data["maxlist"]) * 1.2;
    let whisker_min = arrayMin(selected_data_source.data["boxplot_lower_list"]) * 1.2;
    let whisker_max = arrayMax(selected_data_source.data["boxplot_upper_list"]) * 1.2;
    let box_abs = arrayMax([box_max, -box_min, whisker_max, -whisker_min]);
    boxplot.y_range.start = -box_abs;
    boxplot.y_range.end = box_abs;
    if (selected_data_source.data['idx'].length > 5) {
        boxplot.width = boxplot_unit_width * selected_data_source.data['idx'].length;
    }
    else {
        boxplot.width = boxplot_unit_width * 5;
    }
}

selected_data_source.change.emit();
