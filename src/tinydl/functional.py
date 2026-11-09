"""Numerically stable classification functions built from Tensor operations."""

from __future__ import annotations

import numpy as np

from tinydl.tensor import Tensor


def log_softmax(logits: Tensor, axis: int = -1) -> Tensor:
    """Return log probabilities, subtracting the maximum for stability."""
    maximum = logits.data.max(axis=axis, keepdims=True)
    shifted = logits - maximum
    return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Return mean cross-entropy for class indices or one-hot targets."""
    log_probs = log_softmax(logits, axis=-1)
    if targets.data.shape == logits.data.shape:
        per_item = -(targets * log_probs).sum(axis=-1)
    else:
        if targets.data.ndim != logits.data.ndim - 1:
            raise ValueError("class-index targets must have one fewer dimension than logits")
        indices = targets.data.astype(int)
        if not (targets.data == indices).all():
            raise ValueError("class-index targets must be integers")
        if (indices < 0).any() or (indices >= logits.shape[-1]).any():
            raise ValueError("class-index target is outside the logits class range")
        mask = np.eye(logits.shape[-1], dtype=logits.data.dtype)[indices]
        per_item = -(Tensor(mask, requires_grad=False) * log_probs).sum(axis=-1)
    return per_item.mean()
