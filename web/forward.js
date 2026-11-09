(function (root) {
  "use strict";

  function forward(input, model) {
    if (input.length !== model.input_features) {
      throw new Error("Expected " + model.input_features + " input values");
    }
    let values = input;
    for (const layer of model.layers) {
      const next = new Array(layer.out_features);
      for (let output = 0; output < layer.out_features; output += 1) {
        let sum = layer.bias[output];
        for (let feature = 0; feature < layer.in_features; feature += 1) {
          sum += values[feature] * layer.weights[feature * layer.out_features + output];
        }
        next[output] = layer.activation === "relu" ? Math.max(0, sum) : sum;
      }
      values = next;
    }
    const maximum = Math.max(...values);
    const exponentials = values.map((value) => Math.exp(value - maximum));
    const total = exponentials.reduce((sum, value) => sum + value, 0);
    return exponentials.map((value) => value / total);
  }

  function splitColumnProjection(projection, minimumGap, minimumWidth) {
    let first = -1;
    let last = -1;
    const runs = [];
    for (let x = 0; x < projection.length; x += 1) {
      if (projection[x] <= 0) continue;
      if (first < 0) {
        first = x;
      } else if (x - last - 1 > minimumGap) {
        runs.push([first, last + 1]);
        first = x;
      }
      last = x;
    }
    if (first >= 0) runs.push([first, last + 1]);
    if (runs.length < 2 || runs.some(([start, end]) => end - start < minimumWidth)) {
      return runs.length ? [[runs[0][0], runs[runs.length - 1][1]]] : [];
    }
    return runs;
  }

  const api = { forward, splitColumnProjection };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.TinyDL = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
