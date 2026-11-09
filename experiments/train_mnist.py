"""Train TinyDL's MNIST classifier and report validation-selected results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from tinydl.data import MNISTSplit, batches, load_mnist
from tinydl.functional import cross_entropy
from tinydl.nn import Linear, ReLU, Sequential
from tinydl.optim import Adam
from tinydl.tensor import Tensor


def _model(seed: int) -> Sequential:
    """Create the fixed 784-128-64-10 ReLU classifier for one seed."""
    rng = np.random.default_rng(seed)
    return Sequential(
        Linear(784, 128, rng=rng), ReLU(),
        Linear(128, 64, rng=rng), ReLU(),
        Linear(64, 10, rng=rng),
    )


def _evaluate(model: Sequential, images: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return mean cross-entropy and accuracy without updating parameters."""
    total_loss = 0.0
    total_correct = 0
    for start in range(0, len(images), 512):
        x = Tensor(images[start : start + 512], requires_grad=False)
        y = labels[start : start + 512]
        logits = model(x)
        loss = cross_entropy(logits, Tensor(y, requires_grad=False))
        total_loss += float(loss.data) * len(y)
        total_correct += int(np.count_nonzero(logits.data.argmax(axis=1) == y))
    return total_loss / len(images), total_correct / len(images)


def _train_candidate(
    split: MNISTSplit,
    *,
    seed: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
) -> dict[str, object]:
    """Train one seed/rate candidate and retain its best validation checkpoint."""
    model = _model(seed)
    optimizer = Adam(model.parameters(), learning_rate=learning_rate)
    rng = np.random.default_rng(seed + 10_000)
    best_loss = float("inf")
    best_accuracy = 0.0
    best_epoch = 0
    best_parameters: list[np.ndarray] = []
    history: list[tuple[int, float, float, float, float]] = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        examples = 0
        for x_batch, y_batch in batches(
            split.x_train, split.y_train, batch_size, rng=rng, shuffle=True
        ):
            optimizer.zero_grad()
            logits = model(Tensor(x_batch, requires_grad=False))
            loss = cross_entropy(logits, Tensor(y_batch, requires_grad=False))
            loss.backward()
            optimizer.step()
            count = len(y_batch)
            total_loss += float(loss.data) * count
            examples += count

        train_loss = total_loss / examples
        val_loss, val_accuracy = _evaluate(model, split.x_val, split.y_val)
        history.append((epoch, train_loss, val_loss, val_accuracy, learning_rate))
        print(
            f"seed={seed} lr={learning_rate:g} epoch={epoch} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_accuracy={val_accuracy:.4f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_accuracy = val_accuracy
            best_epoch = epoch
            best_parameters = [parameter.data.copy() for parameter in model.parameters()]

    return {
        "seed": seed,
        "learning_rate": learning_rate,
        "epoch": best_epoch,
        "val_loss": best_loss,
        "val_accuracy": best_accuracy,
        "parameters": best_parameters,
        "history": history,
    }


def _restore(model: Sequential, arrays: list[np.ndarray]) -> None:
    """Copy parameter values into a model in parameter order."""
    for parameter, values in zip(model.parameters(), arrays, strict=True):
        parameter.data[...] = values


def _save_curve(histories: list[list[tuple[int, float, float, float, float]]], path: Path) -> None:
    """Plot training and validation losses from the selected learning-rate runs."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    for index, history in enumerate(histories):
        epochs = [row[0] for row in history]
        axis.plot(epochs, [row[1] for row in history], alpha=0.55, label=f"seed {index} train")
        axis.plot(epochs, [row[2] for row in history], linestyle="--", label=f"seed {index} val")
    axis.set(xlabel="Epoch", ylabel="Cross-entropy", title="MNIST training and validation loss")
    axis.legend(ncol=2, fontsize="small")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.001, 0.003])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--split-seed", type=int, default=2026)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or any(rate <= 0 for rate in args.learning_rates):
        parser.error("epochs, batch size, and learning rates must be positive")

    split = load_mnist(args.data_dir, validation_size=5000, seed=args.split_seed)
    print(
        f"MNIST split: train={len(split.y_train)} val={len(split.y_val)} "
        f"test={len(split.y_test)}; selection uses validation only"
    )
    selected: list[dict[str, object]] = []
    selected_histories: list[list[tuple[int, float, float, float, float]]] = []
    for seed in args.seeds:
        candidates = [
            _train_candidate(
                split, seed=seed, learning_rate=rate,
                epochs=args.epochs, batch_size=args.batch_size,
            )
            for rate in args.learning_rates
        ]
        winner = min(candidates, key=lambda candidate: float(candidate["val_loss"]))
        selected.append(winner)
        selected_histories.append(winner["history"])  # type: ignore[arg-type]

    # Test-set access happens after every checkpoint has been selected on validation loss.
    test_accuracies: list[float] = []
    for winner in selected:
        model = _model(int(winner["seed"]))
        _restore(model, winner["parameters"])  # type: ignore[arg-type]
        _, accuracy = _evaluate(model, split.x_test, split.y_test)
        test_accuracies.append(accuracy)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = args.output_dir / "mnist_loss_curves.png"
    _save_curve(selected_histories, curve_path)
    result_path = args.output_dir / "mnist_results.csv"
    with result_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["seed", "learning_rate", "best_epoch", "val_loss", "val_accuracy", "test_accuracy"])
        for winner, accuracy in zip(selected, test_accuracies, strict=True):
            writer.writerow([
                winner["seed"], winner["learning_rate"], winner["epoch"],
                f"{winner['val_loss']:.8f}", f"{winner['val_accuracy']:.6f}", f"{accuracy:.6f}",
            ])

    mean = float(np.mean(test_accuracies))
    standard_deviation = float(np.std(test_accuracies, ddof=1))
    print("\nseed | learning_rate | best_epoch | validation_accuracy | test_accuracy")
    for winner, accuracy in zip(selected, test_accuracies, strict=True):
        print(
            f"{winner['seed']:>4} | {winner['learning_rate']:>13.4g} | {winner['epoch']:>10} | "
            f"{winner['val_accuracy']:>19.4f} | {accuracy:>13.4f}"
        )
    print(f"Test accuracy: mean={mean:.4f}, sample_std={standard_deviation:.4f}, n={len(test_accuracies)}")
    print(f"Wrote {result_path} and {curve_path}")

    # Save seed zero's validation-selected model as Phase 8's starting point.
    seed_zero = next((row for row in selected if row["seed"] == 0), selected[0])
    export_model = _model(int(seed_zero["seed"]))
    _restore(export_model, seed_zero["parameters"])  # type: ignore[arg-type]
    np.savez(args.output_dir / "mnist_model.npz", *[p.data for p in export_model.parameters()])


if __name__ == "__main__":
    main()
