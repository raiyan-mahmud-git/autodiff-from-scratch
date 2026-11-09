"""Compare SGD, momentum SGD, and Adam using the same five seeds."""

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
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    split = load_mnist(seed=2026)
    results = {
        kind: [
            train_run(
                split, seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                optimizer=kind, target_accuracy=args.target_accuracy,
            )
            for seed in args.seeds
        ]
        for kind in ("sgd", "momentum", "adam")
    }
    print("\noptimizer | target_seconds mean±sd | final_val_accuracy mean±sd")
    for kind, runs in results.items():
        reached = [run.seconds_to_target for run in runs if run.seconds_to_target is not None]
        target_summary = (
            f"{np.mean(reached):.2f}±{np.std(reached, ddof=1):.2f} ({len(reached)}/{len(runs)} reached)"
            if len(reached) > 1
            else (f"{reached[0]:.2f} (1/{len(runs)} reached)" if reached else f"not reached (0/{len(runs)})")
        )
        final = [run.history[-1]["val_accuracy"] if run.history else 0.0 for run in runs]
        print(f"{kind:>9} | {target_summary:>28} | {np.mean(final):.4f}±{np.std(final, ddof=1):.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for kind, runs in results.items():
        max_epochs = max((len(run.history) for run in runs), default=0)
        curve = [
            np.mean([run.history[e]["val_accuracy"] for run in runs if len(run.history) > e])
            for e in range(max_epochs)
        ]
        axes[0].plot(np.arange(1, len(curve) + 1), curve, marker="o", label=kind)
    axes[0].set(xlabel="Epoch", ylabel="Validation accuracy", title="Optimizer comparison")
    axes[0].legend()
    for index, (kind, runs) in enumerate(results.items()):
        values = [run.history[-1]["val_accuracy"] for run in runs if run.history]
        axes[1].scatter([index] * len(values), values, label=kind)
    axes[1].set_xticks(range(3), list(results))
    axes[1].set(ylabel="Final validation accuracy", title="Per-seed outcomes")
    figure.tight_layout()
    path = args.output_dir / "optimizer_comparison.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
