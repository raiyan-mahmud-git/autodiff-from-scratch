"""Calibrate digit-nine sensitivity using validation data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tinydl.data import load_mnist


def _logits(images: np.ndarray, arrays: list[np.ndarray]) -> np.ndarray:
    """Run the 784-128-64-10 affine model before softmax."""
    first = np.maximum(images @ arrays[0] + arrays[1], 0.0)
    second = np.maximum(first @ arrays[2] + arrays[3], 0.0)
    return second @ arrays[4] + arrays[5]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("artifacts/robustness/affine_seed0.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/robustness/calibrated_seed0.npz"))
    parser.add_argument("--max-accuracy-drop", type=float, default=0.001)
    args = parser.parse_args()
    if args.max_accuracy_drop < 0:
        parser.error("max-accuracy-drop must be nonnegative")

    split = load_mnist(seed=2026)
    archive = np.load(args.weights)
    arrays = [archive[key].copy() for key in sorted(archive.files)]
    logits = _logits(split.x_val, arrays)
    candidates = []
    for offset in np.arange(0.0, 1.001, 0.05):
        adjusted = logits.copy()
        adjusted[:, 9] += offset
        predicted = adjusted.argmax(axis=1)
        accuracy = float(np.mean(predicted == split.y_val))
        mask9 = split.y_val == 9
        recall9 = float(np.mean(predicted[mask9] == 9))
        precision9 = float(np.mean(split.y_val[predicted == 9] == 9))
        candidates.append((float(offset), accuracy, recall9, precision9))
    best_accuracy = max(row[1] for row in candidates)
    eligible = [row for row in candidates if row[1] >= best_accuracy - args.max_accuracy_drop]
    selected = max(eligible, key=lambda row: (row[2], -row[0]))

    offset, accuracy, recall9, precision9 = selected
    arrays[5][9] += offset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, *arrays)
    report = {
        "validation_split_seed": 2026,
        "max_accuracy_drop": args.max_accuracy_drop,
        "best_validation_accuracy": best_accuracy,
        "selected_digit9_logit_offset": offset,
        "calibrated_validation_accuracy": accuracy,
        "calibrated_digit9_recall": recall9,
        "calibrated_digit9_precision": precision9,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    baseline = next(row for row in candidates if row[0] == 0.0)
    print("calibration | validation accuracy | digit 9 recall | digit 9 precision")
    print(f"baseline    | {baseline[1]:.4f}               | {baseline[2]:.4f}         | {baseline[3]:.4f}")
    print(f"calibrated  | {accuracy:.4f}               | {recall9:.4f}         | {precision9:.4f}")
    print(f"selected digit 9 logit offset: {offset:.2f}")
    print(f"Saved {args.output} and {report_path}")


if __name__ == "__main__":
    main()
