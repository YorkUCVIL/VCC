

var draw = null; // this is your svg element
var concept_size = 100;
var concept_spacing = 10;
var layer_spacing = 20;
var connectome_spec = {
	size_spec: [2,4,4,1], // check len for n layers
};

function main(){
	draw = SVG('diagram').size(600,600);

	// this is how you draw rects
	//draw.rect().attr({width:concept_size, height:concept_size, fill:'red', x:0, y:0});

	// draw tree
	let n_layers = connectome_spec.size_spec.length;
	for (let l=0;l<n_layers;l++){
		let n_concepts = connectome_spec.size_spec[l];
		for (let c=0;c<n_concepts;c++){
			x_pos = l*(concept_size+layer_spacing); // we push box right as a multiple of the layer num and layer spacing
			y_pos = c*(concept_size+concept_spacing);
			draw.rect().attr({width:concept_size, height:concept_size, fill:'red', x:x_pos, y:y_pos});
		}
	}
	console.log(n_layers);
}


// runs main function on document load
document.addEventListener("DOMContentLoaded", function(event) {
	main();
});
