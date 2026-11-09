(function (root) {
  "use strict";

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

  const api = { splitColumnProjection };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.TinyDLPreprocess = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
