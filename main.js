var draw = null; // this is your svg element
var concept_size = 11;
var concept_spacing = 10;
var layer_spacing = 20;

var selected_model = 'clip_r50_4Lay';
var selected_class = 'tennis_ball';
var cur_edges = {};

// Load your JSON data
var connectome_spec = null; // Will be loaded from the JSON file
var weight_text = [];

// Define model name mapping
var model_name_mapping = {
    'clip_r50_4Lay': 'CLIP',
    'mvit_4Lay': 'MViT',
    'resnet50_4Lay': 'ResNet50',
    'vgg16_4Lay': 'VGG16'
};

async function loadJSON(json_path) {
    const response = await fetch(json_path);
    connectome_spec = await response.json();
}

function hide_all_edges(){
    for (let l in cur_edges){
	    for (let c in cur_edges[l]){
		    let edge_list = cur_edges[l][c];
		    for (let idx=0;idx<edge_list.length;idx++){
			    edge_list[idx].svg_ele.stroke({opacity:0});
		    }
	    }
    }
    for (let idx in weight_text){
	    weight_text[idx].remove();
    }
}

function drawNode(layer, concept, imageSrc, offsetY, weird_layer_idx) {
    // we need weird_layer_idx because you named the layers using arbitrary names
    // we only use this var to find its edges
    let x_pos = layer * (concept_size + layer_spacing);
    let y_pos = offsetY + concept * (concept_size + concept_spacing);
    let node_ele = draw.image(imageSrc, concept_size, concept_size).move(x_pos,y_pos);
    node_ele.node.addEventListener('click',function(){
	    document.getElementById('magnifier').src = imageSrc;
	    hide_all_edges();
            let edge_list = cur_edges[weird_layer_idx][concept];
	    for (let idx=0;idx<edge_list.length;idx++){
		    let weight = edge_list[idx].weight;
		    edge_list[idx].svg_ele.stroke({color:'red',opacity:weight,width:weight*1.5});
		    console.log(edge_list[idx].start_x,edge_list[idx].start_y)

		    let txt_ele = draw.text(`${weight.toFixed(2)}`).font({ size: 4 }).move(edge_list[idx].start_x,edge_list[idx].start_y);
		    weight_text.push(txt_ele);
	    }
    });
    return {
        x_left: x_pos,
        y_middle: y_pos + concept_size / 2,
        x_right: x_pos + concept_size,
        y_middle: y_pos + concept_size / 2
    };
}

function weight_mapping(weight){
	return weight;
	// return Math.pow(weight,2);
}

function drawEdge(startPos, endPos, weight) {
    console.log(`Drawing edge from ${startPos.x_right},${startPos.y_middle} to ${endPos.x_left},${endPos.y_middle} with weight ${weight}`);
    let edge_ele = draw.line(startPos.x_right, startPos.y_middle, endPos.x_left, endPos.y_middle).stroke({ width: weight, color: '#000', opacity: weight_mapping(weight) });
    return {svg_ele: edge_ele, weight: weight, start_x: startPos.x_right, start_y: startPos.y_middle }
}

async function create_connectome_svg(json_path) {
    await loadJSON(json_path);
    draw = SVG('diagram').size(1000, 800); // Increased width to make room for the arrow and labels
    label_name = json_path.split("/")[2];
    let n_layers = connectome_spec.size_spec.length;
    let positions = {};

    // Calculate the maximum height of the canvas
    let maxHeight = Math.max(...connectome_spec.size_spec.map(size => typeof size === 'string' ? parseInt(size) : size * (concept_size + concept_spacing)));
    let maxWidth = n_layers * concept_size + (n_layers - 1) * layer_spacing;

    // Crop the figure to fig html element
    draw.viewbox(0, 0, maxWidth, maxHeight - concept_spacing);

    // Draw nodes and store their positions
    for (let l = 0; l < n_layers; l++) {
        let layerName = connectome_spec.layers[l];
        cur_edges[parseInt(layerName.replace("layer", ""))] = {}; // Create structure to store edge stuff
        let n_concepts = connectome_spec.size_spec[l];
        if (typeof n_concepts === 'string') n_concepts = parseInt(n_concepts);
        positions[layerName] = [];

        // Calculate the offset for centering the nodes in this layer
        let layerHeight = n_concepts * (concept_size + concept_spacing);
        let offsetY = (maxHeight - layerHeight) / 2;

        for (let c = 0; c < n_concepts; c++) {
            cur_edges[parseInt(layerName.replace("layer", ""))][c] = []; // Create structure to store edge stuff
            let imageSrc = connectome_spec.images[layerName][c] + ".png"; // Assuming image paths are saved without the .png extension
            let pos = drawNode(l, c, imageSrc, offsetY, parseInt(layerName.replace("layer", "")));
            positions[layerName].push(pos);
        }
    }

    // Draw edges
    for (let layerName in connectome_spec.edge_weights) {
        let edges = connectome_spec.edge_weights[layerName];
        for (let edge in edges) {
            console.log((edge));
            let [deeperLayer, earlyLayer] = edge.split("-");
            let [deeperConceptIndex, earlyConceptIndex] = [parseInt(deeperLayer.split(" ")[1].replace(label_name + "_concept", "")) - 1, parseInt(earlyLayer.split(" ")[1].replace(label_name + "_concept", "")) - 1];
            let deeperLayerIndex = parseInt(deeperLayer.split(" ")[0].replace("layer", ""))

            if (deeperLayer.split(" ")[0] === "class") {
                deeperConceptIndex = 0;
            }
            let deeperLayerName = deeperLayer.split(" ")[0];
            let earlyLayerName = earlyLayer.split(" ")[0];

            let weight = edges[edge]; // Now weight is a single float

            let startPos = positions[earlyLayerName][earlyConceptIndex];
            let endPos = positions[deeperLayerName][deeperConceptIndex];

            if (startPos && endPos) {
                let ele = drawEdge(startPos, endPos, weight);
                let lower_edges_list = cur_edges[deeperLayerIndex][deeperConceptIndex];
                lower_edges_list.push(ele);
            }
        }
    }


    // Add labels indicating shallow to deep layers
    let shallowLabelX = -15 ; // Positioning the "Shallow Layers" label to the right of the diagram
    let shallowLabelY = maxHeight / 2.1; // Center vertically
    let layersLabelX = -15 ; // Positioning the "Shallow Layers" label to the right of the diagram
    let layerLabelY = maxHeight / 2; // Center vertically

    let deepLabelX = maxWidth + 1; // Positioning the "Deeper Layers" label to the right of the last node
    let deepLabelY = positions[connectome_spec.layers[n_layers - 1]][0].y_middle + 2; // Aligning with the last node

    draw.text("Shallow").move(shallowLabelX, shallowLabelY).font({ size: 4, anchor: 'start', fill: '#000000' });
    draw.text("Layers").move(layersLabelX, layerLabelY).font({ size: 4, anchor: 'start', fill: '#000000' });
    draw.text("Output").move(deepLabelX, deepLabelY).font({ size: 4, anchor: 'start', fill: '#000000' });

    console.log(cur_edges);
    console.log(n_layers);
}


function switch_connectome(){
	var json_path = `demo_outputs2_processed/${selected_model}/${selected_class}/vcc_info.json`;
	var container = document.getElementById('diagram_container');
	var old_fig = document.getElementById('diagram');
	old_fig.remove();
	cur_edges = {};
	weight_text = [];
	var new_fig = document.createElement('div');
	new_fig.id = 'diagram';
	container.appendChild(new_fig);
	create_connectome_svg(json_path);
}


function select_model(name){
	selected_model = name;
    document.getElementById("selected_model").textContent = model_name_mapping[name];
	switch_connectome();
}

function select_class(name){
    selected_class = name;
    document.getElementById("selected_class").textContent = name;
    switch_connectome();
}

function updateDiagram() {
    if (selected_model !== "None" && selected_class !== "None") {
        let vccInfoPath = `demo_outputs2_processed/${selected_model}/${selected_class}/vcc_info.json`;
        switch_connectome(vccInfoPath);
    }
}

// Runs main function on document load
document.addEventListener("DOMContentLoaded", function(event) {
    // path structure: demo_outputs2_processed / model / class / vcc_info.json
	switch_connectome();
});
