$(document).ready(function() {
  set_default_training_data();
  set_default_config();
  init();

  $("#btn-restart").click(updateAndRestartButtonClick);
  $("#btn-next").click(nextButtonClick);
  $("#btn-pca").click(updatePCAButtonClick);
  $("#btn-next20").click(function(){batchTrain(20)});
  $("#btn-next100").click(function(){batchTrain(100)});
  $("#btn-next500").click(function(){batchTrain(500)});
  $("#btn-learning-rate").click(function(){load_config()});
});

function init() {
  $("#error").empty().hide();
  load_training_data();  // this needs to be loaded first to determine vocab
  load_config();
  setup_neural_net();
  setup_neural_net_svg();
  update_neural_net_svg();
  setup_heatmap_svg();
  update_heatmap_svg();
  update_pca();
  setup_scatterplot_svg();
  update_scatterplot_svg();

  // initial feed-forward
  do_feed_forward();
  update_neural_excite_value();
}

function set_default_config() {
  var default_config_obj = {
    hidden_size: 5,
    random_state: 1,
    learning_rate: 0.2,
  };
  $('#config-text').html(JSON.stringify(default_config_obj, null, ''));
}

function load_config() {
  config_obj = {};  // global
  var config_json = $("#config-text").val();
  try{
    config_obj = JSON.parse(config_json);
  } catch (e) {
    show_error("Error parsing the configuration json");
    show_error('The json: ' + config_json);
    return;
  }
  if (config_obj.hidden_size > vocab.length) {
    show_error("Error: hidden layer size cannot exceed vocabulary size.");
  }
}

function set_default_training_data() {
  var presets = 
    [{name:"Fruit and juice", data:"eat|apple,eat|orange,eat|rice,drink|juice,drink|milk,drink|water,orange|juice,apple|juice,rice|milk,milk|drink,water|drink,juice|drink"},
     {name:"Fruit and juice (CBOW)", data: "drink^juice|apple,eat^apple|orange,drink^juice|rice,drink^milk|juice,drink^rice|milk,drink^milk|water,orange^apple|juice,apple^drink|juice,rice^drink|milk,milk^water|drink,water^juice|drink,juice^water|drink"},
     {name:"Fruit and juice (Skip-gram)", data: "apple|drink^juice,orange|eat^apple,rice|drink^juice,juice|drink^milk,milk|drink^rice,water|drink^milk,juice|orange^apple,juice|apple^drink,milk|rice^drink,drink|milk^water,drink|water^juice,drink|juice^water"},
     {name:"Self loop (5-point)", data:"A|A,B|B,C|C,D|D,E|E"},
     {name:"Directed loop (5-point)", data:"A|B,B|C,C|D,D|E,E|A"},
     {name:"Undirected loop (5-point)", data:"A|B,B|C,C|D,D|E,E|A,B|A,C|B,D|C,E|D,A|E"},
     {name:"King and queen", data: "king|kindom,queen|kindom,king|palace,queen|palace,king|royal,queen|royal,king|George,queen|Mary,man|rice,woman|rice,man|farmer,woman|farmer,man|house,woman|house,man|George,woman|Mary"},
     {name:"King and queen (symbol)", data: "king|a,queen|a,king|b,queen|b,king|c,queen|c,king|x,queen|y,man|d,woman|d,man|e,woman|e,man|f,woman|f,man|x,woman|y"},
    ];
  
  $('#input-text').html(presets[0].data);

  var select = d3.select("#sel-presets");

  var options = select.selectAll("option")
    .data(presets)
    .enter()
    .append("option")
    .attr("value", function(d) {return d.name})
    .html(function(d) {return d.name});

  select.on("change", function() {
    var selectedIndex = select.property("selectedIndex");
    var selectedPreset = options.filter(function(d,i) {return i == selectedIndex});
    $('#input-text').html(selectedPreset.datum().data);
    updateAndRestartButtonClick();
  });
}

function load_training_data() {
  input_pairs = [];  // global
  vocab = [];  // global
  current_input = null;  // global, when inactivate, should be null
  current_input_idx = -1;
  var input_text = $("#input-text").val();
  var pairs = input_text.trim().split(",")
  pairs.forEach(function(s) {
    tokens = s.trim().split("|");
    assert(tokens.length == 2, "Bad input format: " + s);
    tokens[0] = tokens[0].trim().split("^");  // input tokens
    tokens[1] = tokens[1].trim().split("^");  // output tokens
    input_pairs.push(tokens);
    tokens[0].forEach(function(t) {vocab.push(t)});
    tokens[1].forEach(function(t) {vocab.push(t)});
  });
  vocab = $.unique(vocab.sort());
}

// "context word" === "input word"
function isCurrentContextWord(w) {
  if (current_input == null) return;
  var context_words = current_input[0];
  if (context_words.length == 1) {
    return w == context_words[0];
  } else if (context_words.length > 1) {
    var matched = false;
    context_words.forEach(function(cw) {
      if (cw == w) {
        matched = true;
        return;
      }
    });
    return matched;
  }
  return false;
}

function isCurrentTargetWord(w) {
  if (current_input == null) return;
  var target_words = current_input[1];
  if (target_words.length == 1) {
    return w == target_words[0];
  } else if (target_words.length > 1) {
    var matched = false;
    target_words.forEach(function(tw) {
      if (tw == w) {
        matched = true;
        return;
      }
    });
    return matched;
  }
  return false;
}

// Regardless the value of current_input, forward to the next input
function activateNextInput() {
  current_input_idx = (current_input_idx + 1) % input_pairs.length;
  current_input = input_pairs[current_input_idx];
  inputNeurons.forEach(function(n, i) {
    n['value'] = isCurrentContextWord(n['word']) ? 1 : 0;
    n['always_excited'] = isCurrentContextWord(n['word']);
  });
  do_feed_forward();  // model
  update_neural_excite_value();  // visual
}

function deactivateCurrentInput() {
  current_input = null;
  inputNeurons.forEach(function(n, i) {
    // n['value'] = 0;
    n['always_excited'] = false;
  });
  do_feed_forward();  // model
  update_neural_excite_value();  // visual
}

function show_error(e) {
  console.log(e);
  var new_error = '<p>' + e + '</p>';
  $('#error').append(new_error);
  $('#error').show();
}

function setup_neural_net() {
  inputNeurons = [];  // global (same below)
  outputNeurons = [];
  hiddenNeurons = [];
  vocab.forEach(function(word, i) {
    inputNeurons.push({word: word, value: 0, idx: i});
    outputNeurons.push({word: word, value: 0, idx: i});
  });
  for (var j = 0; j < config_obj.hidden_size; j++) {
    hiddenNeurons.push({value: 0, idx: j});
  }

  vocabSize = vocab.length;
  hiddenSize = config_obj.hidden_size;
  inputEdges = [];
  outputEdges = [];
  inputVectors = [];  // keeps references to the same set of underlying objects as inputEdges
  outputVectors = [];
  seed_random(config_obj.random_state);  // vector_math.js
  for (var i = 0; i < vocabSize; i++) {
    var inVecTmp = [];
    var outVecTmp = [];
    for (var j = 0; j < hiddenSize; j++) {
      var inWeightTmp = {source: i, target: j, weight: get_random_init_weight(hiddenSize)};
      var outWeightTmp = {source: j, target: i, weight: get_random_init_weight(hiddenSize)};
      inputEdges.push(inWeightTmp);
      outputEdges.push(outWeightTmp);
      inVecTmp.push(inWeightTmp);
      outVecTmp.push(outWeightTmp);
    }
    inputVectors.push(inVecTmp);
    outputVectors.push(outVecTmp);
  }
}

function setup_neural_net_svg() {
  nn_svg_width = 1000;  // view box, not physical
  nn_svg_height = 600;  // W/H ratio should match padding-bottom in wevi.css
  d3.select('div#neuron-vis > *').remove();
  nn_svg = d3.select('div#neuron-vis')
   .append("div")
   .classed("svg-container", true) //container class to make it responsive
   .classed("neural-net", true)
   .append("svg")
   //responsive SVG needs these 2 attributes and no width and height attr
   .attr("preserveAspectRatio", "xMinYMin meet")
   .attr("viewBox", "0 0 " + nn_svg_width + " " + nn_svg_height)
   //class to make it responsive
   .classed("svg-content-responsive", true)
   .classed("neural-net", true);  // for picking up svg from outside

   /* Adding a colored svg background to help debug. */
  // svg.append("rect")
  //   .attr("width", "100%")
  //   .attr("height", "100%")
  //   .attr("fill", "#E8E8EE");

  // Prepare for drawing arrows indicating inputs/outputs
  nn_svg.append('svg:defs')
    .append("svg:marker")
    .attr("id", "marker_arrow")
    .attr('markerHeight', 3.5)
    .attr('markerWidth', 5)
    .attr('markerUnits', 'strokeWidth')
    .attr('orient', 'auto')
    .attr('refX', 0)
    .attr('refY', 0)
    .attr('viewBox', '-5 -5 10 10')
    .append('svg:path')
      .attr('d', 'M 0,0 m -5,-5 L 5,0 L -5,5 Z')
      .attr('fill', io_arrow_color());
}

function update_neural_net_svg() {  
  var colors = ["#427DA8", "#6998BB", "#91B3CD", "#BAD0E0", 
                "#E1ECF3", "#FADEE0", "#F2B5BA", "#EA8B92", 
                "#E2636C", "#DB3B47"];
  numToColor = d3.scale.linear()
    .domain(d3.range(0, 1, 1 / (colors.length - 1)))
    .range(colors);  // global

  var inputNeuronCX = nn_svg_width * 0.2;
  var outputNeuronCX = nn_svg_width - inputNeuronCX;
  var ioNeuronCYMin = nn_svg_height * 0.125;
  var ioNeuronCYInt = (nn_svg_height - 2 * ioNeuronCYMin) / (vocabSize - 1 + 1e-6);
  var hiddenNeuronCX = nn_svg_width / 2;
  var hiddenNeuronCYMin = nn_svg_height * 0.15;
  var hiddenNeuronCYInt = (nn_svg_height - 2 * hiddenNeuronCYMin) / (hiddenSize - 1 + 1e-6);
  var neuronRadius = nn_svg_width * 0.015;
  var neuronLabelOffset = neuronRadius * 1.4;

  var inputNeuronElems = nn_svg
    .selectAll("g.input-neuron")
    .data(inputNeurons)
    .enter()
    .append("g")
    .classed("input-neuron", true)
    .classed("neuron", true);

  inputNeuronElems
    .append("circle")
    .attr("cx", inputNeuronCX)
    .attr("cy", function (d, i) {return ioNeuronCYMin + ioNeuronCYInt * i});

  inputNeuronElems
    .append("text")
    .classed("neuron-label", true)
    .attr("x", inputNeuronCX - neuronLabelOffset)
    .attr("y", function (d, i) {return ioNeuronCYMin + ioNeuronCYInt * i})
    .attr("text-anchor", "end");

  var outputNeuronElems = nn_svg
    .selectAll("g.output-neuron")
    .data(outputNeurons)
    .enter()
    .append("g")
    .classed("output-neuron", true)
    .classed("neuron", true);

  outputNeuronElems
    .append("circle")
    .attr("cx", outputNeuronCX)
    .attr("cy", function (d, i) {return ioNeuronCYMin + ioNeuronCYInt * i});

  outputNeuronElems
    .append("text")
    .classed("neuron-label", true)
    .attr("x", outputNeuronCX + neuronLabelOffset)
    .attr("y", function (d, i) {return ioNeuronCYMin + ioNeuronCYInt * i})
    .attr("text-anchor", "start");

  nn_svg.selectAll("g.hidden-neuron")
    .data(hiddenNeurons)
    .enter()
    .append("g")
    .classed("hidden-neuron", true)
    .classed("neuron", true)
    .append("circle")
    .attr("cx", hiddenNeuronCX)
    .attr("cy", function (d, i) {return hiddenNeuronCYMin + hiddenNeuronCYInt * i;});

  nn_svg.selectAll("g.neuron > circle")
    .attr("r", neuronRadius)
    .attr("stroke-width", "2")
    .attr("stroke", "grey")
    .attr("fill", function(d) {return numToColor(0.5);});

  nn_svg.selectAll(".neuron-label")
    .attr("alignment-baseline", "middle")
    .style("font-size", 24)
    .text(function(d) {return d.word});

  nn_svg.selectAll("g.input-edge")
    .data(inputEdges)
    .enter()
    .append("g")
    .classed("input-edge", true)
    .classed("edge", true)
    .append("line")
    .attr("x1", inputNeuronCX + neuronRadius)
    .attr("x2", hiddenNeuronCX - neuronRadius)
    .attr("y1", function (d) {return ioNeuronCYMin + ioNeuronCYInt * d['source']})
    .attr("y2", function (d) {return hiddenNeuronCYMin + hiddenNeuronCYInt * d['target']})
    .attr("stroke", function (d) {return getInputEdgeStrokeColor(d)})
    .attr("stroke-width", function (d) {return getInputEdgeStrokeWidth(d)});


  nn_svg.selectAll("g.output-edge")
    .data(outputEdges)
    .enter()
    .append("g")
    .classed("output-edge", true)
    .classed("edge", true)
    .append("line")
    .attr("x1", hiddenNeuronCX + neuronRadius)
    .attr("x2", outputNeuronCX - neuronRadius)
    .attr("y1", function (d) {return hiddenNeuronCYMin + hiddenNeuronCYInt * d['source']})
    .attr("y2", function (d) {return ioNeuronCYMin + ioNeuronCYInt * d['target']})
    .attr("stroke", function (d) {return getOutputEdgeStrokeColor(d)})
    .attr("stroke-width", function (d) {return getOutputEdgeStrokeWidth(d)});

  // This function needs to be here, because it needs to "see" ioNeuronCYMin and such...
  draw_input_output_arrows = function() {
    inputNeurons.forEach(function(n, neuronIdx) {
      if (isCurrentContextWord(n.word)) {
        nn_svg.append("line")
          .classed("nn-io-arrow", true)  // used for erasing
          .attr("x1", "0")
          .attr("y1", ioNeuronCYMin + ioNeuronCYInt * neuronIdx)
          .attr("x2", nn_svg_width * 0.075)
          .attr("y2", ioNeuronCYMin + ioNeuronCYInt * neuronIdx)
          .attr("marker-end", "url(#marker_arrow)")
          .style("stroke", io_arrow_color())
          .style("stroke-width", "10");
      }
    });
    outputNeurons.forEach(function(n, neuronIdx) {
      if (isCurrentTargetWord(n.word)) {
        nn_svg.append("line")
          .classed("nn-io-arrow", true)  // used for erasing
          .attr("x1", nn_svg_width)
          .attr("y1", ioNeuronCYMin + ioNeuronCYInt * neuronIdx)
          .attr("x2", nn_svg_width * (1-0.075))
          .attr("y2", ioNeuronCYMin + ioNeuronCYInt * neuronIdx)
          .attr("marker-end", "url(#marker_arrow)")
          .style("stroke", io_arrow_color())
          .style("stroke-width", "10");
      }
    });
  };

  // Set up hover behavior
  d3.selectAll(".input-neuron > circle")
    .on("mouseover", mouseHoverInputNeuron)
    .on("mouseout", mouseOutInputNeuron)
    .on("click", mouseClickInputNeuron);
}

function getInputEdgeStrokeWidth(edge) {
  return isNeuronExcited(inputNeurons[edge.source]) ? 5 : 1;
}

function getInputEdgeStrokeColor(edge) {
  if (isNeuronExcited(inputNeurons[edge.source])) return exciteValueToColor(edge.weight);
  else return "grey";
}

function getOutputEdgeStrokeWidth(edge) {
  return isNeuronExcited(outputNeurons[edge.target]) ? 5 : 1;
}

function getOutputEdgeStrokeColor(edge) {
  if (isNeuronExcited(outputNeurons[edge.target])) return exciteValueToColor(edge.weight * outputNeurons[edge.target].value);
  else return "grey";
}

function isNeuronExcited(neuron) {
  if (! ('value' in neuron)) {
    return false;
  } else {
    return neuron.value > 1.2 / vocabSize;  
  }
}

/*
  Only re-color some elements, without changing the neural-network structure.
*/
function update_neural_excite_value() {
  nn_svg.selectAll("g.neuron > circle")
    .attr("fill", function(d) {return exciteValueToColor(d['value'])});
  nn_svg.selectAll("g.input-edge > line")
    .attr("stroke-width", function(d) {return getInputEdgeStrokeWidth(d)})
    .attr("stroke", function(d) {return getInputEdgeStrokeColor(d)});
  nn_svg.selectAll("g.output-edge > line")
    .attr("stroke-width", function(d) {return getOutputEdgeStrokeWidth(d)})
    .attr("stroke", function(d) {return getOutputEdgeStrokeColor(d)});
}

// Color of arrows indicating context/target words
function io_arrow_color() {
  return "#d62728";
}

function erase_input_output_arrows() {
  nn_svg.selectAll(".nn-io-arrow").remove();
}

// Helper function
// Actual method implemented in vector_math.js
function do_feed_forward() {
  feedforward(inputVectors, outputVectors, inputNeurons, hiddenNeurons, outputNeurons);
}

// Helper function
// Actual method implemented in vector_math.js
function do_backpropagate() {
  var expectedOutput = [];
  outputNeurons.forEach(function(n) {
    if (isCurrentTargetWord(n.word)) {
      expectedOutput.push(1);
    } else {
      expectedOutput.push(0);
    }
  });
  backpropagate(inputVectors, outputVectors, inputNeurons, hiddenNeurons, outputNeurons, expectedOutput);
}

// Helper function
// Actual method implemented in vector_math.js
function do_apply_gradients() {
  apply_gradient(inputVectors, outputVectors, getCurrentLearningRate());
}

function getCurrentLearningRate() {
  return config_obj['learning_rate'];
}

// Input: neural excitement level, can be positive, negative
// Output: a value between 0 and 1, for display
function exciteValueToNum(x) {
  x = x * 5;  // exaggerate it a bit
  return 1 / (1+Math.exp(-x));  // sigmoid
}

function exciteValueToColor(x) {
  return numToColor(exciteValueToNum(x));
}

function mouseHoverInputNeuron(d) {
  // Excite this neuron, inhibit others; ignore always-excited ones
  inputNeurons.forEach(function(n,i) {
    if (('always_excited' in n) && n['always_excited']) {
      return;
    }
    if (i == d.idx) n['value'] = 1;
    else n['value'] = 0;
  });
  do_feed_forward();
  update_neural_excite_value(); 
}

function mouseOutInputNeuron(d) {
  // Inhibit all neurons, except always-excited ones
  inputNeurons.forEach(function(n,i) {
    if (('always_excited' in n) && n['always_excited']) return;
    n['value'] = 0;
  });
  do_feed_forward();
  update_neural_excite_value(); 
}

function mouseClickInputNeuron(d) {
  if (isCurrentContextWord(d['word'])) return;
  var n = inputNeurons[d['idx']];
  if (('always_excited' in n) && n['always_excited']) n['always_excited'] = false;
  else n['always_excited'] = true;
}

function setup_heatmap_svg() {
  hmap_svg_width = 1000;  // view box, not physical
  hmap_svg_height = 700;  // W/H ratio should match padding-bottom in wevi.css
  d3.select('div#heatmap-vis > *').remove();
  hmap_svg = d3.select('div#heatmap-vis')
   .append("div")
   .classed("svg-container", true) //container class to make it responsive
   .classed("heatmap", true)
   .append("svg")
   //responsive SVG needs these 2 attributes and no width and height attr
   .attr("preserveAspectRatio", "xMinYMin meet")
   .attr("viewBox", "0 0 " + hmap_svg_width + " " + hmap_svg_height)
   //class to make it responsive
   .classed("svg-content-responsive", true)
   .classed("heatmap-vis", true);  // for picking up svg from outside
}

function update_heatmap_svg() {
  var inputCellBaseX = 0.15 * hmap_svg_width;
  var ioCellBaseY = 0.1 * hmap_svg_height;
  var matrixPadding = 0.05 * hmap_svg_width;
  var matrixRightMargin = 0.05 * hmap_svg_width;
  var matrixBottomMargin = 0.05 * hmap_svg_height;
  var matrixWidth = (hmap_svg_width - inputCellBaseX - matrixPadding - matrixRightMargin) / 2;
  var outputCellBaseX = inputCellBaseX + matrixWidth + matrixPadding;
  var matrixHeight = hmap_svg_height - ioCellBaseY - matrixBottomMargin;
  var cellWidth = matrixWidth / hiddenSize;
  var cellHeight = matrixHeight / vocabSize;
  var cellFillWidth = 0.95 * cellWidth;
  var cellFillHeight = 0.95 * cellHeight;
  var rowLabelOffset = 0.03 * hmap_svg_width;
  var inputHeaderCX = inputCellBaseX + matrixWidth / 2;
  var outputHeaderCX = hmap_svg_width - matrixRightMargin - matrixWidth/2;
  var ioHeaderBaseY = ioCellBaseY - 0.03 * hmap_svg_height;

  var inputWeightElems = hmap_svg
    .selectAll("g.hmap-input-cell")
    .data(inputEdges)
    .enter()
    .append("g")
    .classed("hmap-input-cell", true)
    .classed("hmap-cell", true)
    .append("rect")
    .attr("x", function (d) {return inputCellBaseX + cellWidth * d['target']})
    .attr("y", function (d) {return ioCellBaseY + cellHeight * d['source']})
    .attr("width", cellFillWidth)
    .attr("height", cellFillHeight);

  var outputWeightElems = hmap_svg
    .selectAll("g.hmap-output-cell")
    .data(outputEdges)
    .enter()
    .append("g")
    .classed("hmap-output-cell", true)
    .classed("hmap-cell", true)
    .append("rect")
    .attr("x", function (d) {return outputCellBaseX + cellWidth * d['source']})
    .attr("y", function (d) {return ioCellBaseY + cellHeight * d['target']})
    .attr("width", cellFillWidth)
    .attr("height", cellFillHeight);

  hmap_svg
    .selectAll("g.hmap-cell > rect")
    .style("fill", function(d) {return exciteValueToColor(d['weight'])});

  hmap_svg
    .selectAll("text.hmap-vocab-label")
    .data(inputNeurons)
    .enter()
    .append("text")
    .classed("hmap-vocab-label", true)
    .text(function(d) {return d.word})
    .attr("x", inputCellBaseX - rowLabelOffset)
    .attr("y", function (d, i) {return ioCellBaseY + cellHeight * i + 0.5 * cellHeight})
    .attr("text-anchor", "end")
    .attr("alignment-baseline", "middle")
    .style("font-size", 30);

  var heatmap_labels = [
    {text: "Input Vector", x: inputHeaderCX, y: ioHeaderBaseY},
    {text: "Output Vector", x: outputHeaderCX, y: ioHeaderBaseY},
  ];
  hmap_svg
    .selectAll("text.hmap-matrix-label")
    .data(heatmap_labels)
    .enter()
    .append("text")
    .classed("hmap-matrix-label", true)
    .attr("x", function(d){return d['x']})
    .attr("y", function(d){return d['y']})
    .text(function(d){return d['text']})
    .style("font-size", 40)
    .style("fill", "grey")
    .attr("text-anchor", "middle")
    .attr("alignment-baseline", "ideographic");
}

/*
  Updates PCA model using current input weights.
  PCA implemented in pca.js
*/
function update_pca() {
  var inputWeightMatrix = []
  inputVectors.forEach(function(v) {
    var tmpRow = [];
    v.forEach(function(e) {tmpRow.push(e['weight'])});
    inputWeightMatrix.push(tmpRow);
  });
  var pca = new PCA();
  var matrixNormalized = pca.scale(inputWeightMatrix, true, true);
  principal_components = pca.pca(matrixNormalized);  // global
}

function setup_scatterplot_svg() {
  scatter_svg_width = 1000;  // view box, not physical
  scatter_svg_height = 700;  // W/H ratio should match padding-bottom in wevi.css
  d3.select('div#scatterplot-vis > *').remove();
  scatter_svg = d3.select('div#scatterplot-vis')  // global
   .append("div")
   .classed("svg-container", true) //container class to make it responsive
   .classed("scatterplot", true)
   .append("svg")
   //responsive SVG needs these 2 attributes and no width and height attr
   .attr("preserveAspectRatio", "xMinYMin meet")
   .attr("viewBox", "0 0 " + scatter_svg_width + " " + scatter_svg_height)
   //class to make it responsive
   .classed("svg-content-responsive", true)
   .classed("scatterplot-vis", true);  // for picking up svg from outside
}

function update_scatterplot_svg() {
  var vectorProjections = [];
  var pc0 = principal_components[0]
  var pc1 = principal_components[1];
  inputVectors.forEach(function(v, i) {
    var tmpVec = [];
    v.forEach(function(e) {tmpVec.push(e['weight'])});
    var proj0 = dot_product(pc0, tmpVec);
    var proj1 = dot_product(pc1, tmpVec);
    vectorProjections.push({
      proj0: proj0,
      proj1: proj1,
      word: inputNeurons[i].word,
      type: 'input',
    });
  });
  outputVectors.forEach(function(v, i) {
    var tmpVec = [];
    v.forEach(function(e) {tmpVec.push(e['weight'])});
    var proj0 = dot_product(pc0, tmpVec);
    var proj1 = dot_product(pc1, tmpVec);
    vectorProjections.push({
      proj0: proj0,
      proj1: proj1,
      word: outputNeurons[i].word,
      type: 'output',
    });
  });
  hiddenNeurons.forEach(function(v, i) {
    var tmpVec = [];
    for (var j = 0; j < hiddenSize; j++) {
      tmpVec[j] = j == i ? 1 : 0;  // a column of an index matrix
    }
    var proj0 = dot_product(pc0, tmpVec);
    var proj1 = dot_product(pc1, tmpVec);
    vectorProjections.push({
      proj0: proj0,
      proj1: proj1,
      word: "h" + i,
      type: 'hidden',
    });
  });

  // Clear up SVG
  scatter_svg.selectAll("*").remove();

  // Add grid line
  var vecRenderBaseX = scatter_svg_width / 2;
  var vecRenderBaseY = scatter_svg_height / 2;
  scatter_svg.append("line")
    .classed("grid-line", true)
    .attr("x1", 0)
    .attr("x2", scatter_svg_width)
    .attr("y1", vecRenderBaseY)
    .attr("y2", vecRenderBaseY);
  scatter_svg.append("line")
    .classed("grid-line", true)
    .attr("x1", vecRenderBaseX)
    .attr("x2", vecRenderBaseX)
    .attr("y1", 0)
    .attr("y2", scatter_svg_height);
  scatter_svg.selectAll(".grid-line")
    .style("stroke", "grey")
    .style("stroke-dasharray", ("30,3"))
    .style("stroke-width", 2)
    .style("stroke-opacity", 0.75);

  var scatter_groups = scatter_svg
    .selectAll("g.scatterplot-vector")
    .data(vectorProjections)
    .enter()
    .append("g")
    .classed("scatterplot-vector", true);

  var ioVectors = scatter_groups
    .filter(function(d) {return d['type'] == "input" || d['type'] == "output"});

  ioVectors
    .append("circle")
    //.attr("x", function (d) {return d['proj0']*1000+500})
    //.attr("y", function (d) {return d['proj1']*1000+500})
    .attr("r", 10)
    .attr("stroke-width", "2")
    .attr("stroke", "grey")
    .attr("fill", getVectorColorBasedOnType);

  ioVectors
    .append("text")
    .attr("dx", "6")
    .attr("dy", "-0.25em")
    .attr("alignment-baseline", "ideographic")
    .style("font-size", 28)
    .style("fill", getVectorColorBasedOnType)
    .text(function(d) {return d.word});

  // Calculate a proper scale
  vecRenderScale = 9999999999;  // global
  vectorProjections.forEach(function(v) {
    if (v['type'] == 'input' || v['type'] == 'output') {
      vecRenderScale = Math.min(vecRenderScale, 0.4 * scatter_svg_width / Math.abs(v['proj0']));
      vecRenderScale = Math.min(vecRenderScale, 0.45 * scatter_svg_height / Math.abs(v['proj1']));
    }
  });

  ioVectors
    .attr("transform", function(d) {
      var x = d['proj0'] * vecRenderScale + vecRenderBaseX;
      var y = d['proj1'] * vecRenderScale + vecRenderBaseY;
      return "translate(" + x + ',' + y +")";
    });
}

function getVectorColorBasedOnType(d) {
  return d['type'] == "input" ? "#1f77b4" : "#ff7f0e";
}

function updatePCAButtonClick() {
  update_pca();
  update_scatterplot_svg();
}

function nextButtonClick() {
  if (current_input) {
    deactivateCurrentInput();
    erase_input_output_arrows();
    do_apply_gradients();  // subtract gradients from weights
    update_heatmap_svg();
    update_scatterplot_svg();
  } else {
    activateNextInput();
    draw_input_output_arrows();
    do_backpropagate();  // compute gradients (without updating weights)
  }
}

function updateAndRestartButtonClick() {
  init();
}

// Train in batch
function batchTrain(numIter) {
  // Step 1:
  activateNextInput();
  draw_input_output_arrows();
  setTimeout(function() {
    do_backpropagate();
    // Step 2:
    deactivateCurrentInput();
    erase_input_output_arrows();
    do_apply_gradients();
    update_heatmap_svg();
    update_scatterplot_svg();
    if (numIter == 1) return;
    else setTimeout(function() {
      batchTrain(numIter - 1)
    }, numIter % 10 == 0 ? 50 : 0);  // when to stop for scatter plots
  }, numIter % 10 == 0 ? 50 : 0);  // when to show input/output arrows
}

/*
  For making the slides presentation only.
*/
function addColorPalette() {
  hmap_svg_width = 1000;  // view box, not physical
  hmap_svg_height = 700;  // W/H ratio should match padding-bottom in wevi.css
  d3.select('div#heatmap-vis > *').remove();
  hmap_svg = d3.select('div#heatmap-vis')
   .append("div")
   .classed("svg-container", true) //container class to make it responsive
   .classed("heatmap", true)
   .append("svg")
   //responsive SVG needs these 2 attributes and no width and height attr
   .attr("preserveAspectRatio", "xMinYMin meet")
   .attr("viewBox", "0 0 " + hmap_svg_width + " " + hmap_svg_height)
   //class to make it responsive
   .classed("svg-content-responsive", true)
   .classed("heatmap-vis", true);  // for picking up svg from outside

  var tmpArray = [];
  for (var i = -1; i < 1; i += 0.03) {
    tmpArray.push(i);
  }

  d3.select("svg.heatmap-vis").selectAll("rect")
    .data(tmpArray)
    .enter()
    .append("rect")
    .attr("x", function (d, i) {return i * 15 + 20})
    .attr("y", 20)
    .attr("width", 17)
    .attr("height", 100)
    .style("fill", function(d) {return exciteValueToColor(d)});
}
