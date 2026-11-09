(function () {
  "use strict";

  const modelElement = document.getElementById("model-data");
  const result = document.getElementById("result");
  let model;
  try {
    model = JSON.parse(modelElement.textContent);
    if (!window.TinyDL || !model || !Array.isArray(model.layers)) {
      throw new Error("Model or demo script is missing");
    }
  } catch (error) {
    result.textContent = "The digit model could not be loaded. Reload the exported demo files.";
    console.error(error);
    return;
  }

  const canvas = document.getElementById("drawing");
  const context = canvas.getContext("2d", { willReadFrequently: true });
  let drawing = false;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = "white";
  context.lineWidth = 26;

  function clearCanvas() {
    context.fillStyle = "black";
    context.fillRect(0, 0, canvas.width, canvas.height);
    result.textContent = "Drawing cleared. Draw one or more separated digits, or load a sample.";
  }

  function point(event) {
    const bounds = canvas.getBoundingClientRect();
    return {
      x: (event.clientX - bounds.left) * canvas.width / bounds.width,
      y: (event.clientY - bounds.top) * canvas.height / bounds.height
    };
  }

  canvas.addEventListener("pointerdown", (event) => {
    drawing = true;
    canvas.setPointerCapture(event.pointerId);
    const p = point(event);
    context.beginPath();
    context.moveTo(p.x, p.y);
    context.lineTo(p.x + 0.1, p.y + 0.1);
    context.stroke();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drawing) return;
    const p = point(event);
    context.lineTo(p.x, p.y);
    context.stroke();
  });
  canvas.addEventListener("pointerup", () => { drawing = false; });
  canvas.addEventListener("pointercancel", () => { drawing = false; });

  function normalizeRegion(minX, minY, maxX, maxY) {
    const boxWidth = maxX - minX + 1;
    const boxHeight = maxY - minY + 1;
    const scale = 20 / Math.max(boxWidth, boxHeight);
    const width = boxWidth * scale;
    const height = boxHeight * scale;
    const grid = document.createElement("canvas");
    grid.width = 28; grid.height = 28;
    const gridContext = grid.getContext("2d", { willReadFrequently: true });
    gridContext.fillStyle = "black";
    gridContext.fillRect(0, 0, 28, 28);
    gridContext.imageSmoothingEnabled = true;
    gridContext.drawImage(canvas, minX, minY, boxWidth, boxHeight,
      (28 - width) / 2, (28 - height) / 2, width, height);
    const bitmap = gridContext.getImageData(0, 0, 28, 28).data;
    let mass = 0, weightedX = 0, weightedY = 0;
    for (let y = 0; y < 28; y += 1) {
      for (let x = 0; x < 28; x += 1) {
        const value = bitmap[(y * 28 + x) * 4] / 255;
        mass += value; weightedX += x * value; weightedY += y * value;
      }
    }
    const shiftX = 13.5 - weightedX / mass;
    const shiftY = 13.5 - weightedY / mass;
    const centered = document.createElement("canvas");
    centered.width = 28; centered.height = 28;
    const centeredContext = centered.getContext("2d", { willReadFrequently: true });
    centeredContext.fillStyle = "black";
    centeredContext.fillRect(0, 0, 28, 28);
    centeredContext.drawImage(grid, shiftX, shiftY);
    const centeredPixels = centeredContext.getImageData(0, 0, 28, 28).data;
    const input = new Array(784);
    for (let index = 0; index < input.length; index += 1) {
      input[index] = centeredPixels[index * 4] / 255;
    }
    return input;
  }

  function preprocessAll() {
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    const projection = new Array(canvas.width).fill(0);
    let minY = canvas.height, maxY = -1;
    for (let y = 0; y < canvas.height; y += 1) {
      for (let x = 0; x < canvas.width; x += 1) {
        if (pixels[(y * canvas.width + x) * 4] > 24) {
          projection[x] += 1;
          minY = Math.min(minY, y);
          maxY = Math.max(maxY, y);
        }
      }
    }
    const activeColumns = projection.flatMap((count, x) => count > 0 ? [x] : []);
    if (!activeColumns.length) return [];
    const totalWidth = activeColumns[activeColumns.length - 1] - activeColumns[0] + 1;
    const minimumGap = Math.max(3, Math.round(totalWidth * 0.025));
    const minimumWidth = Math.max(4, Math.round(totalWidth * 0.08));
    const spans = window.TinyDL.splitColumnProjection(
      projection, minimumGap, minimumWidth
    );
    return spans.map(([startX, endX]) => {
      let top = canvas.height, bottom = -1;
      for (let y = 0; y < canvas.height; y += 1) {
        for (let x = startX; x < endX; x += 1) {
          if (pixels[(y * canvas.width + x) * 4] > 24) {
            top = Math.min(top, y);
            bottom = Math.max(bottom, y);
          }
        }
      }
      return normalizeRegion(startX, top, endX - 1, bottom);
    });
  }

  function classify(input) {
    const probabilities = window.TinyDL.forward(input, model);
    return probabilities.map((probability, digit) => ({ digit, probability }))
      .sort((left, right) => right.probability - left.probability);
  }

  function predict() {
    const inputs = preprocessAll();
    if (!inputs.length) {
      result.textContent = "Draw a digit first, or load a sample digit.";
      return;
    }
    const predictions = inputs.map(classify);
    const number = predictions.map((ranking) => ranking[0].digit).join("");
    if (predictions.length > 1) {
      const heading = document.createElement("p");
      heading.textContent = "Predicted number: " + number;
      const list = document.createElement("ol");
      list.className = "predictions";
      predictions.forEach((ranking, index) => {
        const row = document.createElement("li");
        const top = ranking.slice(0, 3).map((item) =>
          item.digit + " (" + (item.probability * 100).toFixed(1) + "%)"
        ).join(", ");
        row.textContent = "Position " + (index + 1) + " — " + top;
        list.appendChild(row);
      });
      result.replaceChildren(heading, list);
      result.setAttribute("aria-label", "Predicted number " + number + ". " +
        predictions.map((ranking, index) =>
          "position " + (index + 1) + " top choices " + ranking.slice(0, 3)
            .map((item) => item.digit + ", " + (item.probability * 100).toFixed(1) + " percent")
            .join("; ")
        ).join(". "));
      return;
    }
    const list = document.createElement("ol");
    list.className = "predictions";
    for (const item of predictions[0].slice(0, 3)) {
      const row = document.createElement("li");
      row.textContent = "Digit " + item.digit + ": " + (item.probability * 100).toFixed(1) + "%";
      list.appendChild(row);
    }
    result.replaceChildren(list);
    result.setAttribute("aria-label", "Top predictions: " + predictions[0].slice(0, 3).map((item) =>
      "digit " + item.digit + ", " + (item.probability * 100).toFixed(1) + " percent"
    ).join("; "));
  }

  function sample(digit) {
    clearCanvas();
    context.strokeStyle = "white";
    context.lineWidth = 27;
    context.beginPath();
    if (digit === 0) {
      context.ellipse(140, 140, 61, 91, 0, 0, Math.PI * 2);
    } else if (digit === 1) {
      context.moveTo(105, 100); context.lineTo(143, 65); context.lineTo(143, 220);
      context.moveTo(100, 220); context.lineTo(185, 220);
    } else {
      context.moveTo(75, 95);
      context.bezierCurveTo(90, 48, 190, 45, 202, 98);
      context.bezierCurveTo(210, 130, 165, 155, 78, 218);
      context.lineTo(207, 218);
    }
    context.stroke();
    predict();
  }

  document.getElementById("clear").addEventListener("click", clearCanvas);
  document.getElementById("predict").addEventListener("click", predict);
  document.querySelectorAll("[data-sample]").forEach((button) => {
    button.addEventListener("click", () => sample(Number(button.dataset.sample)));
  });
  const themeToggle = document.getElementById("theme-toggle");
  const systemPrefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  themeToggle.textContent = systemPrefersDark ? "Use light theme" : "Use dark theme";
  themeToggle.setAttribute("aria-label", "Switch to " + (systemPrefersDark ? "light" : "dark") + " theme");
  themeToggle.addEventListener("click", () => {
    const current = document.documentElement.dataset.theme || (systemPrefersDark ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    themeToggle.textContent = next === "dark" ? "Use light theme" : "Use dark theme";
    themeToggle.setAttribute("aria-label", "Switch to " + (next === "dark" ? "light" : "dark") + " theme");
  });

  clearCanvas();
})();
