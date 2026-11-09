const fs = require("fs");
const { forward } = require("./forward.js");
const model = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const vectors = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const tolerance = 1e-4;
let maximumError = 0;
for (const vector of vectors) {
  const actual = forward(vector.input, model);
  for (let index = 0; index < actual.length; index += 1) {
    maximumError = Math.max(maximumError, Math.abs(actual[index] - vector.expected_probabilities[index]));
  }
}
if (maximumError >= tolerance) {
  throw new Error("Maximum probability error " + maximumError + " exceeds " + tolerance);
}
process.stdout.write("JavaScript/Python parity passed; max absolute error=" + maximumError + "\n");
