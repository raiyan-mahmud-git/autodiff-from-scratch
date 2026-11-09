"""A small NumPy-backed tensor with reverse-mode automatic differentiation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def _array(value: Any) -> Array:
    """Convert a value to a float64 NumPy array."""
    return np.asarray(value, dtype=np.float64)


def _unbroadcast(gradient: Array, shape: tuple[int, ...]) -> Array:
    """Sum broadcast dimensions so gradient has exactly ``shape``."""
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(shape)


class Tensor:
    """An ndarray value that records operations for reverse-mode gradients."""

    def __init__(
        self,
        data: Any,
        *,
        requires_grad: bool = True,
        _parents: Iterable[Tensor] = (),
        _backward: Callable[[], None] | None = None,
        _op: str = "",
    ) -> None:
        self.data = _array(data)
        self.requires_grad = requires_grad
        self.grad: Array | None = None
        self._parents = tuple(_parents)
        self._backward = _backward or (lambda: None)
        self._op = _op

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the underlying array."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self.data.ndim

    @property
    def T(self) -> Tensor:
        """Return a tensor with axes reversed."""
        return self.transpose()

    def _accumulate(self, gradient: Array) -> None:
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += gradient

    def backward(self, gradient: Any | None = None) -> None:
        """Backpropagate from this tensor; non-scalars need a seed gradient."""
        if gradient is None:
            if self.data.size != 1:
                raise ValueError("gradient must be provided for a non-scalar tensor")
            seed = np.ones_like(self.data)
        else:
            seed = _array(gradient)
            if seed.shape != self.shape:
                raise ValueError(f"gradient shape {seed.shape} does not match {self.shape}")

        order: list[Tensor] = []
        visited: set[int] = set()

        def visit(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._parents:
                visit(parent)
            order.append(node)

        visit(self)
        self._accumulate(seed)
        for node in reversed(order):
            node._backward()

    def __add__(self, other: Any) -> Tensor:
        right = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + right.data, _parents=(self, right), _op="add")

        def backward() -> None:
            if out.grad is not None:
                self._accumulate(_unbroadcast(out.grad, self.shape))
                right._accumulate(_unbroadcast(out.grad, right.shape))

        out._backward = backward
        out.requires_grad = self.requires_grad or right.requires_grad
        return out

    __radd__ = __add__

    def __neg__(self) -> Tensor:
        out = Tensor(-self.data, _parents=(self,), _op="neg")

        def backward() -> None:
            if out.grad is not None:
                self._accumulate(-out.grad)

        out._backward = backward
        out.requires_grad = self.requires_grad
        return out

    def __sub__(self, other: Any) -> Tensor:
        return self + (-other if isinstance(other, Tensor) else -_array(other))

    def __rsub__(self, other: Any) -> Tensor:
        return (-self) + other

    def __mul__(self, other: Any) -> Tensor:
        right = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * right.data, _parents=(self, right), _op="mul")

        def backward() -> None:
            if out.grad is not None:
                self._accumulate(_unbroadcast(out.grad * right.data, self.shape))
                right._accumulate(_unbroadcast(out.grad * self.data, right.shape))

        out._backward = backward
        out.requires_grad = self.requires_grad or right.requires_grad
        return out

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Tensor:
        right = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data / right.data, _parents=(self, right), _op="div")

        def backward() -> None:
            if out.grad is not None:
                self._accumulate(_unbroadcast(out.grad / right.data, self.shape))
                right._accumulate(_unbroadcast(-out.grad * self.data / right.data**2, right.shape))

        out._backward = backward
        out.requires_grad = self.requires_grad or right.requires_grad
        return out

    def __rtruediv__(self, other: Any) -> Tensor:
        return Tensor(other, requires_grad=False) / self

    def __pow__(self, power: float) -> Tensor:
        out = Tensor(self.data**power, _parents=(self,), _op="pow")

        def backward() -> None:
            if out.grad is not None:
                self._accumulate(out.grad * power * self.data ** (power - 1))

        out._backward = backward
        out.requires_grad = self.requires_grad
        return out

    def __matmul__(self, other: Any) -> Tensor:
        right = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data @ right.data, _parents=(self, right), _op="matmul")

        def backward() -> None:
            if out.grad is None:
                return
            left_data = self.data
            right_data = right.data
            grad = out.grad
            left_vector = left_data.ndim == 1
            right_vector = right_data.ndim == 1
            if left_vector:
                left_data = np.expand_dims(left_data, axis=-2)
            if right_vector:
                right_data = np.expand_dims(right_data, axis=-1)
            if left_vector and right_vector:
                grad = np.reshape(grad, (1, 1))
            elif left_vector:
                grad = np.expand_dims(grad, axis=-2)
            elif right_vector:
                grad = np.expand_dims(grad, axis=-1)
            grad_left = grad @ np.swapaxes(right_data, -1, -2)
            grad_right = np.swapaxes(left_data, -1, -2) @ grad
            if left_vector:
                grad_left = np.squeeze(grad_left, axis=-2)
            if right_vector:
                grad_right = np.squeeze(grad_right, axis=-1)
            self._accumulate(_unbroadcast(grad_left, self.shape))
            right._accumulate(_unbroadcast(grad_right, right.shape))

        out._backward = backward
        out.requires_grad = self.requires_grad or right.requires_grad
        return out

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        """Sum values along an axis, retaining a backward reduction rule."""
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), _parents=(self,), _op="sum")

        def backward() -> None:
            if out.grad is None:
                return
            grad = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                for dim in sorted(a % self.ndim for a in axes):
                    grad = np.expand_dims(grad, axis=dim)
            self._accumulate(np.broadcast_to(grad, self.shape))

        out._backward = backward
        out.requires_grad = self.requires_grad
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        """Average values along an axis."""
        if axis is None:
            count = self.data.size
        elif isinstance(axis, int):
            count = self.shape[axis]
        else:
            count = int(np.prod([self.shape[a] for a in axis]))
        return self.sum(axis=axis, keepdims=keepdims) / count

    def reshape(self, *shape: int | tuple[int, ...]) -> Tensor:
        """Return a reshaped tensor."""
        new_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        out = Tensor(self.data.reshape(new_shape), _parents=(self,), _op="reshape")

        def backward() -> None:
            if out.grad is not None:
                self._accumulate(out.grad.reshape(self.shape))

        out._backward = backward
        out.requires_grad = self.requires_grad
        return out

    def transpose(self, *axes: int) -> Tensor:
        """Transpose axes, reversing them by default."""
        permutation = axes or tuple(reversed(range(self.ndim)))
        out = Tensor(self.data.transpose(permutation), _parents=(self,), _op="transpose")
        inverse = tuple(np.argsort(permutation))

        def backward() -> None:
            if out.grad is not None:
                self._accumulate(out.grad.transpose(inverse))

        out._backward = backward
        out.requires_grad = self.requires_grad
        return out

    def exp(self) -> Tensor:
        """Apply the exponential elementwise."""
        value = np.exp(self.data)
        out = Tensor(value, _parents=(self,), _op="exp")

        def backward() -> None:
            if out.grad is not None:
                self._accumulate(out.grad * value)

        out._backward = backward
        out.requires_grad = self.requires_grad
        return out

    def log(self) -> Tensor:
        """Apply the natural logarithm elementwise."""
        out = Tensor(np.log(self.data), _parents=(self,), _op="log")

        def backward() -> None:
            if out.grad is not None:
                self._accumulate(out.grad / self.data)

        out._backward = backward
        out.requires_grad = self.requires_grad
        return out

    def relu(self) -> Tensor:
        """Apply the rectified linear unit elementwise."""
        out = Tensor(np.maximum(0.0, self.data), _parents=(self,), _op="relu")

        def backward() -> None:
            if out.grad is not None:
                self._accumulate(out.grad * (self.data > 0))

        out._backward = backward
        out.requires_grad = self.requires_grad
        return out

    def tanh(self) -> Tensor:
        """Apply hyperbolic tangent elementwise."""
        value = np.tanh(self.data)
        out = Tensor(value, _parents=(self,), _op="tanh")

        def backward() -> None:
            if out.grad is not None:
                self._accumulate(out.grad * (1.0 - value**2))

        out._backward = backward
        out.requires_grad = self.requires_grad
        return out

    def sigmoid(self) -> Tensor:
        """Apply the logistic sigmoid elementwise."""
        positive = self.data >= 0
        value = np.empty_like(self.data)
        value[positive] = 1.0 / (1.0 + np.exp(-self.data[positive]))
        exp_x = np.exp(self.data[~positive])
        value[~positive] = exp_x / (1.0 + exp_x)
        out = Tensor(value, _parents=(self,), _op="sigmoid")

        def backward() -> None:
            if out.grad is not None:
                self._accumulate(out.grad * value * (1.0 - value))

        out._backward = backward
        out.requires_grad = self.requires_grad
        return out

    def __repr__(self) -> str:
        return f"Tensor(data={self.data!r}, grad={self.grad!r})"
