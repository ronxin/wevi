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
  
  /*
   Sanity check from left to right:
   * inputNeurons
   * inputVectors
   * hiddenNeurons
   * outputVectors
   * outputNeurons
  */
  assert(vocabSize == inputNeurons.length);
  assert(vocabSize == inputVectors.length);
  inputVectors.forEach(function(v) {assert(hiddenSize == v.length)});
  assert(hiddenSize == hiddenNeurons.length);
  assert(vocabSize == outputVectors.length);
  outputVectors.forEach(function(v) {assert(hiddenSize == v.length)});
  assert(vocabSize == outputNeurons.length);

  var hiddenValueTemp = [];
  for (var i = 0; i < hiddenSize; i++) hiddenValueTemp.push(0);

  var numInputExcited = 0;
  inputNeurons.forEach(function(n, k) {
    if (n['value'] < 1e-5) return;  // should be either 0 or 1
    numInputExcited += 1;
    for (var i = 0; i < hiddenSize; i++) hiddenValueTemp[i] += inputVectors[k][i]['weight'];
  });

  hiddenNeurons.forEach(function(n, i) {
    if (numInputExcited > 0) {
      n['value'] = hiddenValueTemp[i] / numInputExcited;  // taking average (for CBOW situation)  
    } else {
      n['value'] = 0;
    }
  });

  var outValueTemp = [];
  var sumExpNetInput = 0.0;  // denominator of softmax
  for (var j = 0; j < vocabSize; j++) {
    tmpSum = 0.0;  // net input of neuron j in output layer
    for (var i = 0; i < hiddenSize; i++) {
      tmpSum += hiddenNeurons[i]['value'] * outputVectors[j][i]['weight'];
    }
    outputNeurons[j]['net_input'] = tmpSum;
    expNetInput = exponential(tmpSum);
    if (expNetInput == Infinity) {
      // take max number available in case of exponential blows up
      expNetInput = Number.MAX_VALUE;
    }
    sumExpNetInput += expNetInput;
    outValueTemp.push(expNetInput);
  }
  
  if (sumExpNetInput == Infinity) {
    // take max number available in case of exponential blows up
    sumExpNetInput = Number.MAX_VALUE;
  }
  
  for (var j = 0; j < vocabSize; j++) {  // softmax
    outputNeurons[j]['value'] = outValueTemp[j] / sumExpNetInput;
  }
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
  /*
   Sanity check from left to right:
   * inputNeurons
   * inputVectors
   * hiddenNeurons
   * outputVectors
   * outputNeurons
   * expectedOutput
  */
  assert(vocabSize == inputNeurons.length);
  assert(vocabSize == inputVectors.length);
  inputVectors.forEach(function(v) {assert(hiddenSize == v.length)});
  assert(hiddenSize == hiddenNeurons.length);
  assert(vocabSize == outputVectors.length);
  outputVectors.forEach(function(v) {assert(hiddenSize == v.length)});
  assert(vocabSize == outputNeurons.length);
  assert(vocabSize == expectedOutput.length);

  var errors = [];
  outputNeurons.forEach(function(n, j) {
    // error is the difference between output y_j and expected value t
    // as in the paper Eq(77)
    error_j = n['value'] - expectedOutput[j]
    errors.push(error_j);
    // net_input_gradient is the EI'_j as in the paper Eq(78)
    n['net_input_gradient'] = error_j * n['value'] * (1.0 - n['value']);
  });

  hiddenNeurons.forEach(function(n) {
    n['net_input_gradient'] = 0.0;
  });

  outputVectors.forEach(function(v, j) {  // j: vocab index
    v.forEach(function(e, i) {  // i: hidden layer index
      // this is the gradient defined as partial E partial w'_ij Eq(79)
      e['gradient'] = outputNeurons[j]['net_input_gradient'] * hiddenNeurons[i]['value'];
      // partial E partial h_i Eq (82). This is also Eq(83) since
      // h_i = u_i as defined in the feed forward function
      hiddenNeurons[i]['net_input_gradient'] += outputNeurons[j]['net_input_gradient'] * e['weight'];
    });
  });

  var numInputExcited = 0;
  var isInputExcitedArray = [];
  inputNeurons.forEach(function(n) {
    if (n['value'] < 1e-5) {  // should be either 0 or 1
      isInputExcitedArray.push(false);
    } else {
      isInputExcitedArray.push(true);
      numInputExcited += 1;
    }
  });
  assert(numInputExcited > 0, "With no input assigned, how can you backpropagate??!");
  
  for (var k = 0; k < vocabSize; k++) {
    for (var i = 0; i < hiddenSize; i++) {
      // Eq(84). Note x_k is either 1 or 0
      if (isInputExcitedArray[k])  {
        inputVectors[k][i]['gradient'] = hiddenNeurons[i]['net_input_gradient'] / numInputExcited;
      } else {
        // this is necessary -- it will reset the gradients of non-invovled input vectors.
        inputVectors[k][i]['gradient'] = 0;
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
