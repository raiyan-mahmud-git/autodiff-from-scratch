"""Compare He and naive initialization with five fixed seeds."""

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
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    split = load_mnist(seed=2026)
    results = {
        init: [
            train_run(
                split, seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                optimizer="adam", init=init,
            )
            for seed in args.seeds
        ]
        for init in ("he", "naive")
    }
    print("init | stable runs | train_loss mean±sd | val_accuracy mean±sd")
    for init, runs in results.items():
        losses = [run.history[-1]["train_loss"] for run in runs if run.history]
        accuracy = [run.history[-1]["val_accuracy"] for run in runs if run.history]
        print(
            f"{init:>5} | {sum(run.stable for run in runs)}/{len(runs)} | "
            f"{np.mean(losses):.5f}±{np.std(losses, ddof=1):.5f} | "
            f"{np.mean(accuracy):.4f}±{np.std(accuracy, ddof=1):.4f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for init, runs in results.items():
        epochs = max(len(run.history) for run in runs)
        loss_curve = [
            np.mean([run.history[e]["train_loss"] for run in runs if len(run.history) > e])
            for e in range(epochs)
        ]
        val_curve = [
            np.mean([run.history[e]["val_accuracy"] for run in runs if len(run.history) > e])
            for e in range(epochs)
        ]
        axes[0].plot(range(1, len(loss_curve) + 1), loss_curve, marker="o", label=init)
        axes[1].plot(range(1, len(val_curve) + 1), val_curve, marker="o", label=init)
    axes[0].set(xlabel="Epoch", ylabel="Training loss", title="Initialization and loss")
    axes[1].set(xlabel="Epoch", ylabel="Validation accuracy", title="Initialization and accuracy")
    for axis in axes:
        axis.legend()
    figure.tight_layout()
    path = args.output_dir / "init_comparison.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
