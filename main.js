var draw = null; // this is your svg element
var concept_size = 11;
var concept_spacing = 10;
var layer_spacing = 20;

// Load your JSON data
var connectome_spec = null; // Will be loaded from the JSON file

async function loadJSON() {
    const response = await fetch('demo_outputs_processed/clip_r50_4Lay/apron/vcc_info.json');
    connectome_spec = await response.json();
}

function drawNode(layer, concept, imageSrc, offsetY) {
    let x_pos = layer * (concept_size + layer_spacing);
    let y_pos = offsetY + concept * (concept_size + concept_spacing);
    draw.image(imageSrc, concept_size, concept_size).move(x_pos, y_pos);
    return {
        x_left: x_pos,
        y_middle: y_pos + concept_size / 2,
        x_right: x_pos + concept_size,
        y_middle: y_pos + concept_size / 2
    };
}

function drawEdge(startPos, endPos, weight) {
    console.log(`Drawing edge from ${startPos.x_right},${startPos.y_middle} to ${endPos.x_left},${endPos.y_middle} with weight ${weight}`);
    draw.line(startPos.x_right, startPos.y_middle, endPos.x_left, endPos.y_middle).stroke({ width: weight, color: '#000', opacity: weight });
}

async function main() {
    await loadJSON();
    draw = SVG('diagram').size(800, 800);

    let n_layers = connectome_spec.size_spec.length;
    let positions = {};

    // Calculate the maximum height of the canvas
    let maxHeight = Math.max(...connectome_spec.size_spec.map(size => size * (concept_size + concept_spacing)));

    // Draw nodes and store their positions
    for (let l = 0; l < n_layers; l++) {
        let layerName = connectome_spec.layers[l];
        let n_concepts = connectome_spec.size_spec[l];
        if (typeof n_concepts === 'string') n_concepts = parseInt(n_concepts);
        positions[layerName] = [];

        // Calculate the offset for centering the nodes in this layer
        let layerHeight = n_concepts * (concept_size + concept_spacing);
        let offsetY = (maxHeight - layerHeight) / 2;

        for (let c = 0; c < n_concepts; c++) {
            let imageSrc = connectome_spec.images[layerName][c] + ".png"; // Assuming image paths are saved without the .png extension
            let pos = drawNode(l, c, imageSrc, offsetY);
            positions[layerName].push(pos);
        }
    }

    // Draw edges
    for (let layerName in connectome_spec.edge_weights) {
        let edges = connectome_spec.edge_weights[layerName];
        for (let edge in edges) {
            console.log((edge));
            let [deeperLayer, earlyLayer] = edge.split("-");
            let [deeperConceptIndex, earlyConceptIndex] = [parseInt(deeperLayer.split(" ")[1].replace("apron_concept", "")) - 1, parseInt(earlyLayer.split(" ")[1].replace("apron_concept", "")) - 1];
            if (deeperLayer.split(" ")[0] === "class"){
                deeperConceptIndex = 1;
            }
            let deeperLayerName = deeperLayer.split(" ")[0];
            let earlyLayerName = earlyLayer.split(" ")[0];

            let weightList = edges[edge];
            weightList.forEach(weight => {
                let startPos = positions[earlyLayerName][earlyConceptIndex];
                let endPos = positions[deeperLayerName][deeperConceptIndex];

                if (startPos && endPos) {
                    // check if class is present
                    // if (deeperLayerName === "class") {
                    //     drawEdge(startPos, endPos, weight);
                    // }
                    drawEdge(startPos, endPos, weight);
                }
            });
        }
    }

    console.log(n_layers);
}

// Runs main function on document load
document.addEventListener("DOMContentLoaded", function(event) {
    main();
});