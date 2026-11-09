"""Compare standard and affine-augmented MNIST training on validation data only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import make_model, metrics

from tinydl.data import batches, load_mnist
from tinydl.functional import cross_entropy
from tinydl.optim import Adam
from tinydl.tensor import Tensor


def affine_augment(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply small per-image shifts, rotations, scales, and shears with bilinear sampling."""
    batch = len(images)
    source = images.reshape(batch, 28, 28)
    yy, xx = np.mgrid[0:28, 0:28].astype(np.float64)
    xx = xx[None, :, :] - 13.5
    yy = yy[None, :, :] - 13.5
    angles = rng.uniform(-0.12, 0.12, size=(batch, 1, 1))
    scales = rng.uniform(0.90, 1.10, size=(batch, 1, 1))
    shears = rng.uniform(-0.06, 0.06, size=(batch, 1, 1))
    shift_x = rng.uniform(-1.5, 1.5, size=(batch, 1, 1))
    shift_y = rng.uniform(-1.5, 1.5, size=(batch, 1, 1))
    cosine = np.cos(angles)
    sine = np.sin(angles)
    moved_x = xx - shift_x
    moved_y = yy - shift_y
    source_x = (cosine * moved_x + sine * moved_y) / scales - shears * moved_y + 13.5
    source_y = (-sine * moved_x + cosine * moved_y) / scales + 13.5
    inside = (source_x >= 0) & (source_x <= 27) & (source_y >= 0) & (source_y <= 27)
    x0 = np.floor(source_x).astype(np.int64).clip(0, 27)
    y0 = np.floor(source_y).astype(np.int64).clip(0, 27)
    x1 = np.minimum(x0 + 1, 27)
    y1 = np.minimum(y0 + 1, 27)
    dx = source_x - x0
    dy = source_y - y0
    rows = np.arange(batch)[:, None, None]
    top_left = source[rows, y0, x0]
    top_right = source[rows, y0, x1]
    bottom_left = source[rows, y1, x0]
    bottom_right = source[rows, y1, x1]
    transformed = (
        top_left * (1 - dx) * (1 - dy)
        + top_right * dx * (1 - dy)
        + bottom_left * (1 - dx) * dy
        + bottom_right * dx * dy
    )
    return (transformed * inside).reshape(batch, 784)


def _validation_stats(model, images: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Return validation loss, overall accuracy, and macro-average recall."""
    val_loss, val_accuracy = metrics(model, images, labels)
    predictions = []
    for start in range(0, len(images), 512):
        x = Tensor(images[start : start + 512], requires_grad=False)
        predictions.append(model(x).data.argmax(axis=1))
    predicted = np.concatenate(predictions)
    recalls = [np.mean(predicted[labels == digit] == digit) for digit in range(10)]
    return val_loss, val_accuracy, float(np.mean(recalls))


def _train(split, seed: int, augmented: bool, epochs: int) -> dict[str, object]:
    model = make_model(seed)
    optimizer = Adam(model.parameters(), learning_rate=0.001)
    shuffle_rng = np.random.default_rng(seed + 10_000)
    augment_rng = np.random.default_rng(seed + 30_000)
    best_loss = float("inf")
    best_parameters = []
    best_epoch = 0
    history = []
    best_stats = (0.0, 0.0, 0.0)
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for x_batch, y_batch in batches(
            split.x_train, split.y_train, 128, rng=shuffle_rng, shuffle=True
        ):
            if augmented:
                x_batch = affine_augment(x_batch, augment_rng)
            optimizer.zero_grad()
            loss = cross_entropy(
                model(Tensor(x_batch, requires_grad=False)), Tensor(y_batch, requires_grad=False)
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.data) * len(y_batch)
            count += len(y_batch)
        stats = _validation_stats(model, split.x_val, split.y_val)
        history.append((epoch, loss_sum / count, stats[0], stats[1]))
        print(
            f"seed={seed} affine={augmented} epoch={epoch} "
            f"train_loss={loss_sum / count:.5f} val_loss={stats[0]:.5f} "
            f"val_accuracy={stats[1]:.4f}", flush=True
        )
        if stats[0] < best_loss:
            best_loss = stats[0]
            best_stats = stats
            best_epoch = epoch
            best_parameters = [parameter.data.copy() for parameter in model.parameters()]
    model = make_model(seed)
    for parameter, values in zip(model.parameters(), best_parameters, strict=True):
        parameter.data[...] = values
    predictions = []
    for start in range(0, len(split.x_val), 512):
        predictions.append(
            model(Tensor(split.x_val[start : start + 512], requires_grad=False)).data.argmax(axis=1)
        )
    predicted = np.concatenate(predictions)
    mask9 = split.y_val == 9
    return {
        "seed": seed,
        "augmented": augmented,
        "epoch": best_epoch,
        "val_loss": best_stats[0],
        "val_accuracy": best_stats[1],
        "macro_recall": best_stats[2],
        "recall9": float(np.mean(predicted[mask9] == 9)),
        "history": history,
        "parameters": best_parameters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/robustness"))
    args = parser.parse_args()
    split = load_mnist(seed=2026)
    results = [
        _train(split, seed, augmented, args.epochs)
        for augmented in (False, True)
        for seed in args.seeds
    ]
    print("\ntraining | validation accuracy mean±sd | macro recall mean±sd | digit 9 recall mean±sd")
    for augmented, label in ((False, "standard"), (True, "affine")):
        runs = [row for row in results if row["augmented"] == augmented]
        print(
            f"{label:>8} | {np.mean([row['val_accuracy'] for row in runs]):.4f}±{np.std([row['val_accuracy'] for row in runs], ddof=1):.4f} | "
            f"{np.mean([row['macro_recall'] for row in runs]):.4f}±{np.std([row['macro_recall'] for row in runs], ddof=1):.4f} | "
            f"{np.mean([row['recall9'] for row in runs]):.4f}±{np.std([row['recall9'] for row in runs], ddof=1):.4f}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "validation_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["seed", "affine", "best_epoch", "val_loss", "val_accuracy", "macro_recall", "digit9_recall"])
        for row in results:
            writer.writerow([row[key] for key in ("seed", "augmented", "epoch", "val_loss", "val_accuracy", "macro_recall", "recall9")])
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for augmented, label in ((False, "standard"), (True, "affine")):
        runs = [row for row in results if row["augmented"] == augmented]
        max_epochs = max(len(row["history"]) for row in runs)
        curve = [
            np.mean([row["history"][epoch][3] for row in runs if len(row["history"]) > epoch])
            for epoch in range(max_epochs)
        ]
        axis.plot(range(1, len(curve) + 1), curve, marker="o", label=label)
    axis.set(xlabel="Epoch", ylabel="Validation accuracy", title="Affine augmentation check")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "validation_accuracy.png", dpi=150)
    plt.close(figure)
    seed_zero = next(row for row in results if row["augmented"] and row["seed"] == 0)
    np.savez(args.output_dir / "affine_seed0.npz", *seed_zero["parameters"])
    print(f"Saved validation_metrics.csv, validation_accuracy.png, and affine_seed0.npz in {args.output_dir}")


if __name__ == "__main__":
    main()
