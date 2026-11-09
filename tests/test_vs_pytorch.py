"""Optional numerical comparisons against PyTorch as an independent reference."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip(
    "torch", reason="PyTorch comparison tests require the optional dev dependency torch"
)

from tinydl import Tensor
from tinydl.functional import cross_entropy
from tinydl.optim import SGD, Adam


def _tiny_arrays() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(5, 3)).astype(np.float64)
    weight1 = rng.normal(scale=0.3, size=(3, 4)).astype(np.float64)
    bias1 = rng.normal(scale=0.1, size=(4,)).astype(np.float64)
    weight2 = rng.normal(scale=0.3, size=(4, 3)).astype(np.float64)
    bias2 = rng.normal(scale=0.1, size=(3,)).astype(np.float64)
    target = np.array([0, 2, 1, 0, 2], dtype=np.int64)
    return x, weight1, bias1, weight2, bias2, target


def test_forward_loss_and_every_parameter_gradient_match_pytorch() -> None:
    x, weight1, bias1, weight2, bias2, target = _tiny_arrays()

    tx = Tensor(x, requires_grad=False)
    tw1, tb1 = Tensor(weight1), Tensor(bias1)
    tw2, tb2 = Tensor(weight2), Tensor(bias2)
    tiny_hidden = (tx @ tw1 + tb1).relu()
    tiny_logits = tiny_hidden @ tw2 + tb2
    tiny_loss = cross_entropy(tiny_logits, Tensor(target, requires_grad=False))
    tiny_loss.backward()

    px = torch.tensor(x, dtype=torch.float64)
    pw1 = torch.tensor(weight1, dtype=torch.float64, requires_grad=True)
    pb1 = torch.tensor(bias1, dtype=torch.float64, requires_grad=True)
    pw2 = torch.tensor(weight2, dtype=torch.float64, requires_grad=True)
    pb2 = torch.tensor(bias2, dtype=torch.float64, requires_grad=True)
    py = torch.tensor(target, dtype=torch.long)
    pytorch_hidden = torch.relu(px @ pw1 + pb1)
    pytorch_logits = pytorch_hidden @ pw2 + pb2
    pytorch_loss = torch.nn.functional.cross_entropy(pytorch_logits, py)
    pytorch_loss.backward()

    np.testing.assert_allclose(tiny_logits.data, pytorch_logits.detach().numpy(), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(tiny_loss.data, pytorch_loss.item(), rtol=1e-8, atol=1e-8)
    for tiny_parameter, torch_parameter in zip(
        (tw1, tb1, tw2, tb2), (pw1, pb1, pw2, pb2), strict=True
    ):
        np.testing.assert_allclose(
            tiny_parameter.grad, torch_parameter.grad.numpy(), rtol=1e-8, atol=1e-8
        )


@pytest.mark.parametrize("optimizer_name", ["sgd", "adam"])
def test_two_hundred_training_steps_match_pytorch(optimizer_name: str) -> None:
    rng = np.random.default_rng(73)
    x = rng.normal(size=(12, 4)).astype(np.float64)
    target = rng.integers(0, 3, size=12, dtype=np.int64)
    weight1 = rng.normal(scale=0.25, size=(4, 5)).astype(np.float64)
    bias1 = np.zeros(5, dtype=np.float64)
    weight2 = rng.normal(scale=0.25, size=(5, 3)).astype(np.float64)
    bias2 = np.zeros(3, dtype=np.float64)

    tiny_parameters = [Tensor(value) for value in (weight1, bias1, weight2, bias2)]
    tiny_optimizer = (
        SGD(tiny_parameters, learning_rate=0.03)
        if optimizer_name == "sgd"
        else Adam(tiny_parameters, learning_rate=0.01)
    )

    torch_parameters = [
        torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for value in (weight1, bias1, weight2, bias2)
    ]
    torch_optimizer = (
        torch.optim.SGD(torch_parameters, lr=0.03)
        if optimizer_name == "sgd"
        else torch.optim.Adam(torch_parameters, lr=0.01)
    )
    pytorch_x = torch.tensor(x, dtype=torch.float64)
    pytorch_target = torch.tensor(target, dtype=torch.long)
    losses_tiny: list[float] = []
    losses_torch: list[float] = []

    for _ in range(200):
        tiny_optimizer.zero_grad()
        w1, b1, w2, b2 = tiny_parameters
        tiny_logits = Tensor(x, requires_grad=False) @ w1
        tiny_logits = (tiny_logits + b1).relu() @ w2 + b2
        tiny_loss = cross_entropy(tiny_logits, Tensor(target, requires_grad=False))
        losses_tiny.append(float(tiny_loss.data))
        tiny_loss.backward()
        tiny_optimizer.step()

        torch_optimizer.zero_grad(set_to_none=True)
        torch_w1, torch_b1, torch_w2, torch_b2 = torch_parameters
        torch_logits = torch.relu(pytorch_x @ torch_w1 + torch_b1) @ torch_w2 + torch_b2
        torch_loss = torch.nn.functional.cross_entropy(torch_logits, pytorch_target)
        losses_torch.append(float(torch_loss.item()))
        torch_loss.backward()
        torch_optimizer.step()

    np.testing.assert_allclose(losses_tiny, losses_torch, rtol=1e-8, atol=1e-8)
    for tiny_parameter, torch_parameter in zip(tiny_parameters, torch_parameters, strict=True):
        np.testing.assert_allclose(tiny_parameter.data, torch_parameter.detach().numpy(), rtol=1e-8, atol=1e-8)

