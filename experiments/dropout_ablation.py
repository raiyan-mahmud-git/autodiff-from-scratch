"""Compare dropout off and on with five fixed seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import train_run

from tinydl.data import load_mnist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--probability", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    split = load_mnist(seed=2026)
    results = {
        label: [
            train_run(
                split, seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                optimizer="adam", dropout=probability,
            )
            for seed in args.seeds
        ]
        for label, probability in (("off", 0.0), ("on", args.probability))
    }
    print("dropout | train_accuracy mean±sd | val_accuracy mean±sd | train-val gap mean±sd")
    for label, runs in results.items():
        train = [run.history[-1]["train_accuracy"] for run in runs if run.history]
        valid = [run.history[-1]["val_accuracy"] for run in runs if run.history]
        gaps = np.asarray(train) - np.asarray(valid)
        print(
            f"{label:>7} | {np.mean(train):.4f}±{np.std(train, ddof=1):.4f} | "
            f"{np.mean(valid):.4f}±{np.std(valid, ddof=1):.4f} | "
            f"{np.mean(gaps):.4f}±{np.std(gaps, ddof=1):.4f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for label, runs in results.items():
        epochs = len(runs[0].history)
        train_curve = [np.mean([run.history[e]["train_accuracy"] for run in runs]) for e in range(epochs)]
        val_curve = [np.mean([run.history[e]["val_accuracy"] for run in runs]) for e in range(epochs)]
        axes[0].plot(range(1, epochs + 1), train_curve, marker="o", label=f"{label} train")
        axes[0].plot(range(1, epochs + 1), val_curve, marker="s", linestyle="--", label=f"{label} validation")
        axes[1].plot(range(1, epochs + 1), np.asarray(train_curve) - np.asarray(val_curve), marker="o", label=label)
    axes[0].set(xlabel="Epoch", ylabel="Accuracy", title="Train and validation accuracy")
    axes[0].legend()
    axes[1].set(xlabel="Epoch", ylabel="Train minus validation accuracy", title="Generalization gap")
    axes[1].legend()
    figure.tight_layout()
    path = args.output_dir / "dropout_ablation.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
