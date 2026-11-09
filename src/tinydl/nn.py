"""Basic neural-network modules for TinyDL."""

from __future__ import annotations

import numpy as np

from tinydl.tensor import Tensor


class Module:
    """Base class for modules that contain trainable parameters or submodules."""

    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return this module's parameters, recursively, without duplicates."""
        found: list[Tensor] = []
        seen: set[int] = set()

        def collect(value: object) -> None:
            if isinstance(value, Tensor):
                if value.requires_grad and id(value) not in seen:
                    seen.add(id(value))
                    found.append(value)
            elif isinstance(value, Module):
                for child in vars(value).values():
                    collect(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    collect(child)

        collect(self)
        return found

    def zero_grad(self) -> None:
        """Clear gradients on every parameter."""
        for parameter in self.parameters():
            parameter.grad = None

    def train(self, mode: bool = True) -> Module:
        """Set this module and all child modules to training or evaluation mode."""
        self.training = mode
        for value in vars(self).values():
            if isinstance(value, Module):
                value.train(mode)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    if isinstance(child, Module):
                        child.train(mode)
        return self

    def eval(self) -> Module:
        """Set this module and children to evaluation mode."""
        return self.train(False)

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute this module's output; subclasses must implement it."""
        raise NotImplementedError


class Linear(Module):
    """Fully connected affine transformation ``x @ weight + bias``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        init: str = "he",
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("feature dimensions must be positive")
        generator = rng if rng is not None else np.random.default_rng()
        if init == "he":
            scale = np.sqrt(2.0 / in_features)
        elif init == "naive":
            scale = 1.0
        else:
            raise ValueError("init must be 'he' or 'naive'")
        self.weight = Tensor(generator.standard_normal((in_features, out_features)) * scale)
        self.bias = Tensor(np.zeros(out_features))

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs @ self.weight + self.bias


class ReLU(Module):
    """Rectified linear activation module."""

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.relu()


class Tanh(Module):
    """Hyperbolic tangent activation module."""

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.tanh()


class Dropout(Module):
    """Inverted dropout, disabled in evaluation mode."""

    def __init__(self, probability: float = 0.5, *, seed: int | None = None) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.probability = probability
        self._rng = np.random.default_rng(seed)

    def forward(self, inputs: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return inputs
        keep_probability = 1.0 - self.probability
        mask = self._rng.random(inputs.shape) < keep_probability
        return inputs * (mask.astype(inputs.data.dtype) / keep_probability)


class Sequential(Module):
    """Apply modules in order."""

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        if not all(isinstance(module, Module) for module in modules):
            raise TypeError("Sequential accepts only Module instances")
        self.layers: tuple[Module, ...] = tuple(modules)

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

