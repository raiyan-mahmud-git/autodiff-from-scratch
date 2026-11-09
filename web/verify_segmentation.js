const assert = require("assert");
const { splitColumnProjection } = require("./forward.js");
assert.deepStrictEqual(
  splitColumnProjection([0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 4, 4, 4, 0], 3, 2),
  [[2, 5], [10, 13]],
);
assert.deepStrictEqual(
  splitColumnProjection([0, 2, 2, 0, 0, 2, 2, 0], 2, 2),
  [[1, 7]],
);
assert.deepStrictEqual(
  splitColumnProjection([0, 2, 0, 0, 0, 0, 5, 5, 5, 0], 2, 2),
  [[1, 9]],
);
process.stdout.write("Digit segmentation checks passed\n");
