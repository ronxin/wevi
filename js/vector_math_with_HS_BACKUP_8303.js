<<<<<<< HEAD
/*
  Vector mathematics for WEVI.
*/
function get_random_init_weight(hidden_size) {
  random_float = get_random() / 65536;
  return (random_float - 0.5) / hidden_size;
}

next_random = 1;
function get_random() {
  next_random = (next_random * 25214903917 + 11) & 0xffff;
  return next_random;
}

function seed_random(seed) {
  next_random = seed;
}

/*
  Modify the "value" attribute of each neuron in hiddenNeurons and outputNeurons.
  The "value" attributes of inputNeurons are expected to be either 0 or 1.
*/
function feedforward(inputVectors, outputVectors, inputNeurons, hiddenNeurons, outputNeurons) {
  var hiddenSize = hiddenNeurons.length;
  var vocabSize = inputNeurons.length;
  
  /* Sanity check */
  assert(vocabSize == inputVectors.length);
  assert(vocabSize == outputVectors.length);
  assert(vocabSize == outputNeurons.length);
  inputVectors.forEach(function(v) {assert(hiddenSize == v.length)});
  outputVectors.forEach(function(v) {assert(hiddenSize == v.length)});

  var hiddenValueTemp = [];
  for (var j = 0; j < hiddenSize; j++) hiddenValueTemp.push(0);

  var numInputExcited = 0;
  inputNeurons.forEach(function(n,i) {
    if (n['value'] < 1e-5) return;  // should be either 0 or 1
    numInputExcited += 1;
    for (var j = 0; j < hiddenSize; j++) hiddenValueTemp[j] += inputVectors[i][j]['weight'];
  });

  hiddenNeurons.forEach(function(n,j) {
    if (numInputExcited > 0) {
      n['value'] = hiddenValueTemp[j] / numInputExcited;  // taking average (for CBOW situation)  
    } else {
      n['value'] = 0;
    }
  });

  var outValueTemp = [];
  var sumExpNetInput = 0.0;  // denominator of softmax
  for (var i = 0; i < vocabSize; i++) {
    tmpSum = 0.0;  // net input of neuron i in output layer
    for (var j = 0; j < hiddenSize; j++) {
      tmpSum += outputVectors[i][j]['weight'] * hiddenNeurons[j]['value'];
    }
    outputNeurons[i]['net_input'] = tmpSum;
    expNetInput = exponential(tmpSum);
    if (expNetInput == Infinity) expNetInput = Number.MAX_VALUE;
    sumExpNetInput += expNetInput;
    outValueTemp.push(expNetInput);
  }
  
  if (sumExpNetInput == Infinity) sumExpNetInput = Number.MAX_VALUE;
  
  for (var i = 0; i < vocabSize; i++) {  // softmax
    outputNeurons[i]['value'] = outValueTemp[i] / sumExpNetInput;
  }
}


/* ADAPTED FEEDFORWARD WITH HIERARCHICAL SOFTMAX */

function feedforward_with_HS(inputVectors, outputVectors, inputNeurons, hiddenNeurons, outputNeurons, outputNodes) {
  var hiddenSize = hiddenNeurons.length;
  var vocabSize = inputNeurons.length;
  
  /* Sanity check */
  assert(vocabSize == inputVectors.length);
  assert(vocabSize == outputVectors.length);
  assert(vocabSize == outputNeurons.length);
  inputVectors.forEach(function(v) {assert(hiddenSize == v.length)});
  outputVectors.forEach(function(v) {assert(hiddenSize == v.length)});

  var hiddenValueTemp = [];
  for (var j = 0; j < hiddenSize; j++) hiddenValueTemp.push(0);

  var numInputExcited = 0;
  inputNeurons.forEach(function(n,i) {
    if (n['value'] < 1e-5) return;  // should be either 0 or 1
    numInputExcited += 1;
    for (var j = 0; j < hiddenSize; j++) hiddenValueTemp[j] += inputVectors[i][j]['weight'];
  });

  hiddenNeurons.forEach(function(n,j) {
    if (numInputExcited > 0) {
      n['value'] = hiddenValueTemp[j] / numInputExcited;  // taking average (for CBOW situation)  
    } else {
      n['value'] = 0;
    }
  });

  for (var i = 0; i < vocabSize; i++) {
    outputNeurons[i]['value'] = hs_prob(outputNodes, inputNeurons[i].word);
  }

  /* ARCHIVED FROM VECTOR_MATH.js
  var outValueTemp = [];
  var sumExpNetInput = 0.0;  // denominator of softmax
  for (var i = 0; i < vocabSize; i++) {
    tmpSum = 0.0;  // net input of neuron i in output layer
    for (var j = 0; j < hiddenSize; j++) {
      tmpSum += outputVectors[i][j]['weight'] * hiddenNeurons[j]['value'];
    }
    outputNeurons[i]['net_input'] = tmpSum;
    expNetInput = exponential(tmpSum);
    if (expNetInput == Infinity) expNetInput = Number.MAX_VALUE;
    sumExpNetInput += expNetInput;
    outValueTemp.push(expNetInput);
  }
  
  if (sumExpNetInput == Infinity) sumExpNetInput = Number.MAX_VALUE;
  
  for (var i = 0; i < vocabSize; i++) {  // softmax
    outputNeurons[i]['value'] = outValueTemp[i] / sumExpNetInput;
  }
  */
}

/*
  Modifies inputVectors and outputVectors, populating the "gradient" field of each.
  These gradients need to be later "applied" using the apply_gradient() function.
  Also modifies hiddenNeurons and outputNeurons, populating their "net_input_gradient" field.
  @param expectedOutput - an array of 0s and 1s.
*/
function backpropagate(inputVectors, outputVectors, inputNeurons, hiddenNeurons, outputNeurons, expectedOutput) {
  var hiddenSize = hiddenNeurons.length;
  var vocabSize = inputNeurons.length;
  
  /* Sanity check */
  assert(vocabSize == inputVectors.length);
  assert(vocabSize == outputVectors.length);
  assert(vocabSize == outputNeurons.length);
  assert(vocabSize == expectedOutput.length);
  inputVectors.forEach(function(v) {assert(hiddenSize == v.length)});
  outputVectors.forEach(function(v) {assert(hiddenSize == v.length)});

  var errors = [];
  outputNeurons.forEach(function(n, i) {
    error_i = n['value'] - expectedOutput[i]
    errors.push(error_i);
    n['net_input_gradient'] = error_i;
  });

  hiddenNeurons.forEach(function(n, j) {
    n['net_input_gradient'] = 0.0;
  });

  outputVectors.forEach(function(v, i) {  // i: vocab index (opposite to my paper's notations)
    v.forEach(function(e, j) {  // j: hidden layer index
      e['gradient'] = errors[i] * hiddenNeurons[j]['value'];
      hiddenNeurons[j]['net_input_gradient'] += errors[i] * e['weight'];
    });
  });

  var numInputExcited = 0;
  var isInputExcitedArray = [];
  inputNeurons.forEach(function(n,i) {
    if (n['value'] < 1e-5) {  // should be either 0 or 1
      isInputExcitedArray.push(false);
    } else {
      isInputExcitedArray.push(true);
      numInputExcited += 1;
    }
  });
  assert(numInputExcited > 0, "With no input assigned, how can you backpropagate??!");
  
  for (var i = 0; i < vocabSize; i++) {
    for (var j = 0; j < hiddenSize; j++) {
      if (isInputExcitedArray[i])  {
        inputVectors[i][j]['gradient'] = hiddenNeurons[j]['net_input_gradient'] / numInputExcited;
      } else {
        // this is necessary -- it will reset the gradients of non-invovled input vectors.
        inputVectors[i][j]['gradient'] = 0;
      }
    }
  }
}


/* ADAPTED BACKPROP WITH HIERARCHICAL SOFTMAX */

function backpropagate_with_HS(inputVectors, outputVectors, inputNeurons, hiddenNeurons, outputNeurons, expectedOutput, outputNodes) {
  var hiddenSize = hiddenNeurons.length;
  var vocabSize = inputNeurons.length;
  
  /* Sanity check */
  assert(vocabSize == inputVectors.length);
  assert(vocabSize == outputVectors.length);
  assert(vocabSize == outputNeurons.length);
  assert(vocabSize == expectedOutput.length);
  inputVectors.forEach(function(v) {assert(hiddenSize == v.length)});
  outputVectors.forEach(function(v) {assert(hiddenSize == v.length)});

  var errors = [];
  outputNeurons.forEach(function(n, i) {
    error_i = n['value'] - expectedOutput[i]
    errors.push(error_i);
    n['net_input_gradient'] = error_i;
  });

  hiddenNeurons.forEach(function(n, j) {
    n['net_input_gradient'] = 0.0;
  });

  //only need to update vectors on path
  var start_word = outputNeurons[expectedOutput.indexOf(1)].word;

  var path_index = get_index(outputNodes, start_word);
  while(outputNodes[path_index].parent) {
    var tj = 0;
    if (outputNodes[path_index] == outputNodes[path_index].parent.left) {tj = 1;}

    path_index = outputNodes[path_index].parent.index;

    var dotprod = 0.0;
    for (var j = 0; j < hiddenSize; j++) {
      dotprod += outputNodes[path_index].vect[j] * hiddenNeurons[j]['value'];
    }

    var grad = sigmoid(dotprod) - tj;
    
    for (var j = 0; j < hiddenSize; j++) {
      outputNodes[path_index].gradient[j] = grad * hiddenNeurons[j]['value'];
      hiddenNeurons[j]['net_input_gradient'] += grad * outputNodes[path_index].vect[j];
    }
  }

  /* ARCHIVED FROM VECTOR_MATH.js
  outputVectors.forEach(function(v, i) {  // i: vocab index (opposite to my paper's notations)
    v.forEach(function(e, j) {  // j: hidden layer index
      e['gradient'] = errors[i] * hiddenNeurons[j]['value'];
      hiddenNeurons[j]['net_input_gradient'] += errors[i] * e['weight'];
    });
  });
  */

  var numInputExcited = 0;
  var isInputExcitedArray = [];
  inputNeurons.forEach(function(n,i) {
    if (n['value'] < 1e-5) {  // should be either 0 or 1
      isInputExcitedArray.push(false);
    } else {
      isInputExcitedArray.push(true);
      numInputExcited += 1;
    }
  });
  assert(numInputExcited > 0, "With no input assigned, how can you backpropagate??!");
  
  for (var i = 0; i < vocabSize; i++) {
    for (var j = 0; j < hiddenSize; j++) {
      if (isInputExcitedArray[i])  {
        inputVectors[i][j]['gradient'] = hiddenNeurons[j]['net_input_gradient'] / numInputExcited;
      } else {
        // this is necessary -- it will reset the gradients of non-invovled input vectors.
        inputVectors[i][j]['gradient'] = 0;
      }
    }
  }
}


function apply_gradient(inputVectors, outputVectors, learning_rate) {
  inputVectors.forEach(function(v) {
    v.forEach(function(e) {
      e['weight'] -= learning_rate * e['gradient'];
    });
  });

  outputVectors.forEach(function(v) {
    v.forEach(function(e) {
      e['weight'] -= learning_rate * e['gradient'];
    });
  });
}

/* ADAPTED APPLY GRADIENT WITH HIERARCHICAL SOFTMAX */

function apply_gradient_with_HS(inputVectors, outputVectors, outputNodes, learning_rate) {
  inputVectors.forEach(function(v) {
    v.forEach(function(e) {
      e['weight'] -= learning_rate * e['gradient'];
    });
  });

  outputVectors.forEach(function(v) {
    v.forEach(function(e) {
      e['weight'] -= learning_rate * e['gradient'];
    });
  });

  for(var i = 0; i < outputNodes.length; i++) {
    for (var j = 0; j < hiddenSize; j++) {
      outputNodes[i].vect[j] -= learning_rate * outputNodes[i].gradient[j];
      outputNodes[i].recent_gradient[j] = outputNodes[i].gradient[j]; //store gradients for future inspection
      outputNodes[i].gradient[j] = 0; //reset gradients after single use
    }
  }
}

function exponential(x) {
  return Math.exp(x);
}

function dot_product(a, b) {
  assert(a.length == b.length);
  tmpSum = 0;
  for (var i = 0; i < a.length; i++) {
    tmpSum += a[i] * b[i];
  }
  return tmpSum;
}

function sigmoid(x) {
  return (1 / (1 + Math.exp(-x)));
}

function get_index(nodes, string) {
  return nodes.findIndex(x => x.value == string);
}

function hs_prob(nodes, string) {
  var i = get_index(nodes, string);
  var cumProd = 1.0;
  var tempProd = 1.0;	

  while(nodes[i].parent) {
    var sign = -1;
    if (nodes[i] == nodes[i].parent.left) {sign = 1;}

    i = nodes[i].parent.index;
    var dotprod = 0.0;
    for (var j = 0; j < hiddenSize; j++) {
      dotprod += nodes[i].vect[j] * hiddenNeurons[j]['value'];
    }

    tempProd = sigmoid(dotprod * sign);
    cumProd *= tempProd;
  }

  return cumProd;
=======
/*
  Vector mathematics for WEVI.
*/
function get_random_init_weight(hidden_size) {
  random_float = get_random() / 65536;
  return (random_float - 0.5) / hidden_size;
}

next_random = 1;
function get_random() {
  next_random = (next_random * 25214903917 + 11) & 0xffff;
  return next_random;
}

function seed_random(seed) {
  next_random = seed;
}

/*
  Modify the "value" attribute of each neuron in hiddenNeurons and outputNeurons.
  The "value" attributes of inputNeurons are expected to be either 0 or 1.
*/
function feedforward(inputVectors, outputVectors, inputNeurons, hiddenNeurons, outputNeurons) {
  var hiddenSize = hiddenNeurons.length;
  var vocabSize = inputNeurons.length;
  
  /* Sanity check */
  assert(vocabSize == inputVectors.length);
  assert(vocabSize == outputVectors.length);
  assert(vocabSize == outputNeurons.length);
  inputVectors.forEach(function(v) {assert(hiddenSize == v.length)});
  outputVectors.forEach(function(v) {assert(hiddenSize == v.length)});

  var hiddenValueTemp = [];
  for (var j = 0; j < hiddenSize; j++) hiddenValueTemp.push(0);

  var numInputExcited = 0;
  inputNeurons.forEach(function(n,i) {
    if (n['value'] < 1e-5) return;  // should be either 0 or 1
    numInputExcited += 1;
    for (var j = 0; j < hiddenSize; j++) hiddenValueTemp[j] += inputVectors[i][j]['weight'];
  });

  hiddenNeurons.forEach(function(n,j) {
    if (numInputExcited > 0) {
      n['value'] = hiddenValueTemp[j] / numInputExcited;  // taking average (for CBOW situation)  
    } else {
      n['value'] = 0;
    }
  });

  var outValueTemp = [];
  var sumExpNetInput = 0.0;  // denominator of softmax
  for (var i = 0; i < vocabSize; i++) {
    tmpSum = 0.0;  // net input of neuron i in output layer
    for (var j = 0; j < hiddenSize; j++) {
      tmpSum += outputVectors[i][j]['weight'] * hiddenNeurons[j]['value'];
    }
    outputNeurons[i]['net_input'] = tmpSum;
    expNetInput = exponential(tmpSum);
    if (expNetInput == Infinity) expNetInput = Number.MAX_VALUE;
    sumExpNetInput += expNetInput;
    outValueTemp.push(expNetInput);
  }
  
  if (sumExpNetInput == Infinity) sumExpNetInput = Number.MAX_VALUE;
  
  for (var i = 0; i < vocabSize; i++) {  // softmax
    outputNeurons[i]['value'] = outValueTemp[i] / sumExpNetInput;
  }
}


/* ADAPTED FEEDFORWARD WITH HIERARCHICAL SOFTMAX */

function feedforward_with_HS(inputVectors, outputVectors, inputNeurons, hiddenNeurons, outputNeurons, outputNodes) {
  var hiddenSize = hiddenNeurons.length;
  var vocabSize = inputNeurons.length;
  
  /* Sanity check */
  assert(vocabSize == inputVectors.length);
  assert(vocabSize == outputVectors.length);
  assert(vocabSize == outputNeurons.length);
  inputVectors.forEach(function(v) {assert(hiddenSize == v.length)});
  outputVectors.forEach(function(v) {assert(hiddenSize == v.length)});

  var hiddenValueTemp = [];
  for (var j = 0; j < hiddenSize; j++) hiddenValueTemp.push(0);

  var numInputExcited = 0;
  inputNeurons.forEach(function(n,i) {
    if (n['value'] < 1e-5) return;  // should be either 0 or 1
    numInputExcited += 1;
    for (var j = 0; j < hiddenSize; j++) hiddenValueTemp[j] += inputVectors[i][j]['weight'];
  });

  hiddenNeurons.forEach(function(n,j) {
    if (numInputExcited > 0) {
      n['value'] = hiddenValueTemp[j] / numInputExcited;  // taking average (for CBOW situation)  
    } else {
      n['value'] = 0;
    }
  });

  for (var i = 0; i < vocabSize; i++) {
    outputNeurons[i]['value'] = hs_prob(outputNodes, inputNeurons[i].word);
  }

  /* ARCHIVED FROM VECTOR_MATH.js
  var outValueTemp = [];
  var sumExpNetInput = 0.0;  // denominator of softmax
  for (var i = 0; i < vocabSize; i++) {
    tmpSum = 0.0;  // net input of neuron i in output layer
    for (var j = 0; j < hiddenSize; j++) {
      tmpSum += outputVectors[i][j]['weight'] * hiddenNeurons[j]['value'];
    }
    outputNeurons[i]['net_input'] = tmpSum;
    expNetInput = exponential(tmpSum);
    if (expNetInput == Infinity) expNetInput = Number.MAX_VALUE;
    sumExpNetInput += expNetInput;
    outValueTemp.push(expNetInput);
  }
  
  if (sumExpNetInput == Infinity) sumExpNetInput = Number.MAX_VALUE;
  
  for (var i = 0; i < vocabSize; i++) {  // softmax
    outputNeurons[i]['value'] = outValueTemp[i] / sumExpNetInput;
  }
  */
}

/*
  Modifies inputVectors and outputVectors, populating the "gradient" field of each.
  These gradients need to be later "applied" using the apply_gradient() function.
  Also modifies hiddenNeurons and outputNeurons, populating their "net_input_gradient" field.
  @param expectedOutput - an array of 0s and 1s.
*/
function backpropagate(inputVectors, outputVectors, inputNeurons, hiddenNeurons, outputNeurons, expectedOutput) {
  var hiddenSize = hiddenNeurons.length;
  var vocabSize = inputNeurons.length;
  
  /* Sanity check */
  assert(vocabSize == inputVectors.length);
  assert(vocabSize == outputVectors.length);
  assert(vocabSize == outputNeurons.length);
  assert(vocabSize == expectedOutput.length);
  inputVectors.forEach(function(v) {assert(hiddenSize == v.length)});
  outputVectors.forEach(function(v) {assert(hiddenSize == v.length)});

  var errors = [];
  outputNeurons.forEach(function(n, i) {
    error_i = n['value'] - expectedOutput[i]
    errors.push(error_i);
    n['net_input_gradient'] = error_i;
  });

  hiddenNeurons.forEach(function(n, j) {
    n['net_input_gradient'] = 0.0;
  });

  outputVectors.forEach(function(v, i) {  // i: vocab index (opposite to my paper's notations)
    v.forEach(function(e, j) {  // j: hidden layer index
      e['gradient'] = errors[i] * hiddenNeurons[j]['value'];
      hiddenNeurons[j]['net_input_gradient'] += errors[i] * e['weight'];
    });
  });

  var numInputExcited = 0;
  var isInputExcitedArray = [];
  inputNeurons.forEach(function(n,i) {
    if (n['value'] < 1e-5) {  // should be either 0 or 1
      isInputExcitedArray.push(false);
    } else {
      isInputExcitedArray.push(true);
      numInputExcited += 1;
    }
  });
  assert(numInputExcited > 0, "With no input assigned, how can you backpropagate??!");
  
  for (var i = 0; i < vocabSize; i++) {
    for (var j = 0; j < hiddenSize; j++) {
      if (isInputExcitedArray[i])  {
        inputVectors[i][j]['gradient'] = hiddenNeurons[j]['net_input_gradient'] / numInputExcited;
      } else {
        // this is necessary -- it will reset the gradients of non-invovled input vectors.
        inputVectors[i][j]['gradient'] = 0;
      }
    }
  }
}


/* ADAPTED BACKPROP WITH HIERARCHICAL SOFTMAX */

function backpropagate_with_HS(inputVectors, outputVectors, inputNeurons, hiddenNeurons, outputNeurons, expectedOutput, outputNodes) {
  var hiddenSize = hiddenNeurons.length;
  var vocabSize = inputNeurons.length;
  
  /* Sanity check */
  assert(vocabSize == inputVectors.length);
  assert(vocabSize == outputVectors.length);
  assert(vocabSize == outputNeurons.length);
  assert(vocabSize == expectedOutput.length);
  inputVectors.forEach(function(v) {assert(hiddenSize == v.length)});
  outputVectors.forEach(function(v) {assert(hiddenSize == v.length)});

  var errors = [];
  outputNeurons.forEach(function(n, i) {
    error_i = n['value'] - expectedOutput[i]
    errors.push(error_i);
    n['net_input_gradient'] = error_i;
  });

  hiddenNeurons.forEach(function(n, j) {
    n['net_input_gradient'] = 0.0;
  });

  //only need to update vectors on path
  var start_word = outputNeurons[expectedOutput.indexOf(1)].word;

  var path_index = get_index(outputNodes, start_word);
  while(outputNodes[path_index].parent) {
    var tj = 0;
    if (outputNodes[path_index] == outputNodes[path_index].parent.left) {tj = 1;}

    path_index = outputNodes[path_index].parent.index;

    var dotprod = 0.0;
    for (var j = 0; j < hiddenSize; j++) {
      dotprod += outputNodes[path_index].vect[j] * hiddenNeurons[j]['value'];
    }

    var grad = sigmoid(dotprod) - tj;
    
    for (var j = 0; j < hiddenSize; j++) {
      outputNodes[path_index].gradient[j] = grad * hiddenNeurons[j]['value'];
      hiddenNeurons[j]['net_input_gradient'] += grad * outputNodes[path_index].vect[j];
    }
  }

  /* ARCHIVED FROM VECTOR_MATH.js
  outputVectors.forEach(function(v, i) {  // i: vocab index (opposite to my paper's notations)
    v.forEach(function(e, j) {  // j: hidden layer index
      e['gradient'] = errors[i] * hiddenNeurons[j]['value'];
      hiddenNeurons[j]['net_input_gradient'] += errors[i] * e['weight'];
    });
  });
  */

  var numInputExcited = 0;
  var isInputExcitedArray = [];
  inputNeurons.forEach(function(n,i) {
    if (n['value'] < 1e-5) {  // should be either 0 or 1
      isInputExcitedArray.push(false);
    } else {
      isInputExcitedArray.push(true);
      numInputExcited += 1;
    }
  });
  assert(numInputExcited > 0, "With no input assigned, how can you backpropagate??!");
  
  for (var i = 0; i < vocabSize; i++) {
    for (var j = 0; j < hiddenSize; j++) {
      if (isInputExcitedArray[i])  {
        inputVectors[i][j]['gradient'] = hiddenNeurons[j]['net_input_gradient'] / numInputExcited;
      } else {
        // this is necessary -- it will reset the gradients of non-invovled input vectors.
        inputVectors[i][j]['gradient'] = 0;
      }
    }
  }
}


function apply_gradient(inputVectors, outputVectors, learning_rate) {
  inputVectors.forEach(function(v) {
    v.forEach(function(e) {
      e['weight'] -= learning_rate * e['gradient'];
    });
  });

  outputVectors.forEach(function(v) {
    v.forEach(function(e) {
      e['weight'] -= learning_rate * e['gradient'];
    });
  });
}

/* ADAPTED APPLY GRADIENT WITH HIERARCHICAL SOFTMAX */

function apply_gradient_with_HS(inputVectors, outputVectors, outputNodes, learning_rate) {
  inputVectors.forEach(function(v) {
    v.forEach(function(e) {
      e['weight'] -= learning_rate * e['gradient'];
    });
  });

  outputVectors.forEach(function(v) {
    v.forEach(function(e) {
      e['weight'] -= learning_rate * e['gradient'];
    });
  });

  for(var i = 0; i < outputNodes.length; i++) {
    for (var j = 0; j < hiddenSize; j++) {
      outputNodes[i].vect[j] -= learning_rate * outputNodes[i].gradient[j];
      outputNodes[i].recent_gradient[j] = outputNodes[i].gradient[j]; //store gradients for future inspection
      outputNodes[i].gradient[j] = 0; //reset gradients after single use
    }
  }
}

function exponential(x) {
  return Math.exp(x);
}

function dot_product(a, b) {
  assert(a.length == b.length);
  tmpSum = 0;
  for (var i = 0; i < a.length; i++) {
    tmpSum += a[i] * b[i];
  }
  return tmpSum;
}

function sigmoid(x) {
  return (1 / (1 + Math.exp(-x)));
}

function get_index(nodes, string) {
  return nodes.findIndex(x => x.value == string);
}

function hs_prob(nodes, string) {
  var i = get_index(nodes, string);
  var cumProd = 1.0;
  var tempProd = 1.0;	

  while(nodes[i].parent) {
    var sign = -1;
    if (nodes[i] == nodes[i].parent.left) {sign = 1;}

    i = nodes[i].parent.index;
    var dotprod = 0.0;
    for (var j = 0; j < hiddenSize; j++) {
      dotprod += nodes[i].vect[j] * hiddenNeurons[j]['value'];
    }

    tempProd = sigmoid(dotprod * sign);
    cumProd *= tempProd;
  }

  return cumProd;
>>>>>>> ce2785f3a4664246f47cd70759f54bcc40541bce
}