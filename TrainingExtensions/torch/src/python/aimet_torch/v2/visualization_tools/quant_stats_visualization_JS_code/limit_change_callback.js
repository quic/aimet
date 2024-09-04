// Function to adaptively format numerical values in scientific notation
// if they are large in magnitude
function formatValue(value) {
    if (Math.abs(value) < 1e-3 || Math.abs(value) > 1e3) {
        return value.toExponential(2);
    } else {
        return value.toFixed(2);
    }
}

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