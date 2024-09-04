table_data_source.data["idx"].forEach(i => {
    data_source.data["selected"][i] = false;
});
table_data_source.selected.indices.forEach(i => {
    let layer_idx = table_data_source.data["idx"][i];
    data_source.data["selected"][layer_idx] = true;
});
data_source.change.emit();

selected_data_source.data["namelist"] = [];
selected_data_source.data["idx"] = [];
selected_data_source.data["floor"] = [];
selected_data_source.data["ceil"] = [];
selected_data_source.data["minlist"] = [];
selected_data_source.data["maxlist"] = [];
selected_data_source.change.emit();

data_source.data["selected"].forEach((bool,index) => {
    if (bool==true) {
        selected_data_source.data["namelist"].push(data_source.data["namelist"][index]);
        selected_data_source.data["idx"].push(index);
        selected_data_source.data["floor"].push(limits_source.data['ymin'][0]*1.05);
        selected_data_source.data["ceil"].push(limits_source.data['ymax'][0]*1.05);
        selected_data_source.data["minlist"].push(data_source.data["minlist"][index]);
        selected_data_source.data["maxlist"].push(data_source.data["maxlist"][index]);
    }
})
selected_data_source.change.emit();