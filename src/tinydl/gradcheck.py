"""Finite-difference tools for checking autodiff gradients."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from tinydl.tensor import Tensor

Array = npt.NDArray[np.float64]


def _scalar_value(function: Callable[..., Tensor], values: Sequence[Array]) -> float:
    """Evaluate a scalar-valued tensor function without tracking gradients."""
    inputs = [Tensor(value, requires_grad=False) for value in values]
    output = function(*inputs)
    if output.data.size != 1:
        raise ValueError("gradient checking requires a scalar-valued function")
    return float(output.data.reshape(()))


def numerical_gradient(
    function: Callable[..., Tensor],
    values: Sequence[Any],
    *,
    eps: float = 1e-5,
) -> list[Array]:
    """Estimate input gradients using central differences in float64."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    arrays = [np.asarray(value, dtype=np.float64).copy() for value in values]
    gradients = [np.zeros_like(value) for value in arrays]
    for input_index, value in enumerate(arrays):
        for index in np.ndindex(value.shape):
            original = value[index]
            value[index] = original + eps
            plus = _scalar_value(function, arrays)
            value[index] = original - eps
            minus = _scalar_value(function, arrays)
            value[index] = original
            gradients[input_index][index] = (plus - minus) / (2.0 * eps)
    return gradients


def check_gradients(
    function: Callable[..., Tensor],
    values: Sequence[Any],
    *,
    eps: float = 1e-5,
) -> float:
    """Check analytic gradients and return their maximum relative error.

    Relative error for each coordinate is ``abs(a - n) / max(1, abs(a), abs(n))``.
    This denominator avoids unstable ratios near zero while retaining a useful
    scale for larger gradients.
    """
    inputs = [Tensor(np.asarray(value, dtype=np.float64).copy()) for value in values]
    output = function(*inputs)
    if output.data.size != 1:
        raise ValueError("gradient checking requires a scalar-valued function")
    output.backward()
    analytic = [tensor.grad for tensor in inputs]
    if any(gradient is None for gradient in analytic):
        raise ValueError("all checked inputs must participate in the output")
    numeric = numerical_gradient(function, values, eps=eps)
    errors = [
        np.abs(actual - expected) / np.maximum(1.0, np.maximum(np.abs(actual), np.abs(expected)))
        for actual, expected in zip(analytic, numeric, strict=True)
    ]
    return max((float(np.max(error)) for error in errors), default=0.0)
