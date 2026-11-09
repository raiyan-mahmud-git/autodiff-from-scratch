"""Export trained TinyDL weights and JavaScript parity vectors for the web demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from tinydl.data import load_mnist


def _forward(input_values: npt.NDArray[np.float64], layers: list[dict[str, Any]]) -> npt.NDArray[np.float64]:
    """Run the exported dense network in float64 using float32-quantized weights."""
    values = input_values.astype(np.float64, copy=False)
    for layer in layers:
        inputs = int(layer["in_features"])
        outputs = int(layer["out_features"])
        weight = np.asarray(layer["weights"], dtype=np.float32).astype(np.float64).reshape(inputs, outputs)
        bias = np.asarray(layer["bias"], dtype=np.float32).astype(np.float64)
        values = values @ weight + bias
        if layer["activation"] == "relu":
            values = np.maximum(values, 0.0)
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def export_web_model(
    weights_path: str | Path,
    output_dir: str | Path,
    *,
    data_dir: str | Path = "data/mnist",
    template_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write float32 model weights, parity vectors, and a self-contained HTML page."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_weights = np.load(weights_path)
    arrays = [source_weights[key] for key in sorted(source_weights.files)]
    expected_shapes = [(784, 128), (128,), (128, 64), (64,), (64, 10), (10,)]
    if [array.shape for array in arrays] != expected_shapes:
        raise ValueError("weights file does not match the expected 784-128-64-10 model")
    layers: list[dict[str, Any]] = []
    for index, activation in ((0, "relu"), (2, "relu"), (4, "linear")):
        weight = np.asarray(arrays[index], dtype=np.float32)
        bias = np.asarray(arrays[index + 1], dtype=np.float32)
        layers.append(
            {
                "in_features": int(weight.shape[0]),
                "out_features": int(weight.shape[1]),
                "weights": weight.ravel().astype(float).tolist(),
                "bias": bias.astype(float).tolist(),
                "activation": activation,
            }
        )
    model = {
        "format": "tinydl-dense-v1",
        "weight_dtype": "float32",
        "input_features": 784,
        "layers": layers,
    }
    calibration_path = Path(weights_path).with_suffix(".json")
    if calibration_path.exists():
        model["calibration"] = json.loads(calibration_path.read_text(encoding="utf-8"))
    model_path = output / "model.json"
    model_path.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")

    split = load_mnist(data_dir, validation_size=5000, seed=2026)
    rng = np.random.default_rng(2026)
    indices = rng.choice(len(split.x_train), size=3, replace=False)
    vectors = []
    for index in indices:
        values = split.x_train[index]
        probabilities = _forward(values, layers)
        vectors.append(
            {
                "input": values.astype(float).tolist(),
                "label": int(split.y_train[index]),
                "expected_probabilities": probabilities.astype(float).tolist(),
            }
        )
    vectors_path = output / "test_vectors.json"
    vectors_path.write_text(json.dumps(vectors, separators=(",", ":")), encoding="utf-8")

    template = Path(template_path) if template_path else output / "index.template.html"
    html = template.read_text(encoding="utf-8")
    marker = '<script id="model-data" type="application/json"></script>'
    if marker not in html:
        raise ValueError(f"model data placeholder not found in {template}")
    embedded = '<script id="model-data" type="application/json">' + json.dumps(model, separators=(",", ":")) + "</script>"
    (output / "index.html").write_text(html.replace(marker, embedded), encoding="utf-8")
    return model_path, vectors_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path, default=Path("artifacts/robustness/calibrated_seed0.npz")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    parser.add_argument("--output-dir", type=Path, default=Path("web"))
    parser.add_argument("--template", type=Path, default=Path("web/index.template.html"))
    args = parser.parse_args()
    model_path, vectors_path = export_web_model(
        args.weights, args.output_dir, data_dir=args.data_dir, template_path=args.template
    )
    print(f"Wrote {model_path}, {vectors_path}, and {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
