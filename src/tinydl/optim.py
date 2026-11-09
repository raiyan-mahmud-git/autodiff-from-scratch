"""Gradient-based optimizers for TinyDL parameters."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from tinydl.tensor import Tensor


class SGD:
    """Stochastic gradient descent with optional classical momentum."""

    def __init__(
        self,
        parameters: Iterable[Tensor],
        learning_rate: float = 0.01,
        momentum: float = 0.0,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self._velocity = [np.zeros_like(parameter.data) for parameter in self.parameters]

    def step(self) -> None:
        """Apply one update to each parameter that has a gradient."""
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            if self.momentum:
                self._velocity[index] = (
                    self.momentum * self._velocity[index] + parameter.grad
                )
                update = self._velocity[index]
            else:
                update = parameter.grad
            parameter.data -= self.learning_rate * update

    def zero_grad(self) -> None:
        """Clear gradients for all optimizer parameters."""
        for parameter in self.parameters:
            parameter.grad = None


class Adam:
    """Adam optimizer with first and second moments and bias correction."""

    def __init__(
        self,
        parameters: Iterable[Tensor],
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._first = [np.zeros_like(parameter.data) for parameter in self.parameters]
        self._second = [np.zeros_like(parameter.data) for parameter in self.parameters]
        self._step_count = 0

    def step(self) -> None:
        """Apply one Adam update, including moment bias correction."""
        self._step_count += 1
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            self._first[index] = self.beta1 * self._first[index] + (1.0 - self.beta1) * gradient
            self._second[index] = (
                self.beta2 * self._second[index] + (1.0 - self.beta2) * gradient**2
            )
            first_hat = self._first[index] / (1.0 - self.beta1**self._step_count)
            second_hat = self._second[index] / (1.0 - self.beta2**self._step_count)
            parameter.data -= self.learning_rate * first_hat / (np.sqrt(second_hat) + self.epsilon)

    def zero_grad(self) -> None:
        """Clear gradients for all optimizer parameters."""
        for parameter in self.parameters:
            parameter.grad = None
