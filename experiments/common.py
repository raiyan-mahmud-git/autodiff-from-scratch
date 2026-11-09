"""Shared deterministic training helpers for the MNIST experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from tinydl.data import MNISTSplit, batches
from tinydl.functional import cross_entropy
from tinydl.nn import Dropout, Linear, ReLU, Sequential
from tinydl.optim import SGD, Adam
from tinydl.tensor import Tensor


@dataclass
class RunResult:
    """Per-epoch metrics and final run measurements for one seed/configuration."""

    history: list[dict[str, float]]
    elapsed_seconds: float
    seconds_to_target: float | None
    stable: bool


def make_model(seed: int, *, init: str = "he", dropout: float = 0.0) -> Sequential:
    """Build the shared 784-128-64-10 model with selected init and dropout."""
    rng = np.random.default_rng(seed)
    return Sequential(
        Linear(784, 128, init=init, rng=rng),
        ReLU(),
        Dropout(dropout, seed=seed + 20_000),
        Linear(128, 64, init=init, rng=rng),
        ReLU(),
        Dropout(dropout, seed=seed + 40_000),
        Linear(64, 10, init=init, rng=rng),
    )


def metrics(model: Sequential, images: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Measure mean loss and accuracy in evaluation mode, in manageable batches."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    for start in range(0, len(images), 512):
        image_batch = Tensor(images[start : start + 512], requires_grad=False)
        label_batch = labels[start : start + 512]
        logits = model(image_batch)
        loss = cross_entropy(logits, Tensor(label_batch, requires_grad=False))
        total_loss += float(loss.data) * len(label_batch)
        correct += int(np.count_nonzero(logits.data.argmax(axis=1) == label_batch))
    model.train(was_training)
    return total_loss / len(images), correct / len(images)


def make_optimizer(kind: str, parameters: list[Tensor]):
    """Create an experiment optimizer with documented fixed learning rates."""
    if kind == "sgd":
        return SGD(parameters, learning_rate=0.01)
    if kind == "momentum":
        return SGD(parameters, learning_rate=0.01, momentum=0.9)
    if kind == "adam":
        return Adam(parameters, learning_rate=0.001)
    raise ValueError(f"unknown optimizer: {kind}")


def train_run(
    split: MNISTSplit,
    *,
    seed: int,
    epochs: int = 5,
    batch_size: int = 128,
    optimizer: str = "adam",
    init: str = "he",
    dropout: float = 0.0,
    target_accuracy: float | None = None,
) -> RunResult:
    """Train one controlled run, recording train and validation metrics each epoch."""
    model = make_model(seed, init=init, dropout=dropout)
    update = make_optimizer(optimizer, model.parameters())
    rng = np.random.default_rng(seed + 10_000)
    history: list[dict[str, float]] = []
    seconds_to_target = None
    start_time = time.perf_counter()
    stable = True

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for image_batch, label_batch in batches(
            split.x_train, split.y_train, batch_size, rng=rng, shuffle=True
        ):
            update.zero_grad()
            logits = model(Tensor(image_batch, requires_grad=False))
            loss = cross_entropy(logits, Tensor(label_batch, requires_grad=False))
            if not np.isfinite(loss.data).all():
                stable = False
                break
            loss.backward()
            update.step()
            total_loss += float(loss.data) * len(label_batch)
            seen += len(label_batch)
        if not stable:
            break

        train_loss = total_loss / seen
        train_eval_loss, train_accuracy = metrics(model, split.x_train, split.y_train)
        val_loss, val_accuracy = metrics(model, split.x_val, split.y_val)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_eval_loss": train_eval_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        if target_accuracy is not None and seconds_to_target is None and val_accuracy >= target_accuracy:
            seconds_to_target = time.perf_counter() - start_time
        print(
            f"seed={seed} optimizer={optimizer} init={init} dropout={dropout:g} epoch={epoch} "
            f"train_loss={train_loss:.5f} train_accuracy={train_accuracy:.4f} "
            f"val_loss={val_loss:.5f} val_accuracy={val_accuracy:.4f}",
            flush=True,
        )

    elapsed = time.perf_counter() - start_time
    return RunResult(history, elapsed, seconds_to_target, stable)
