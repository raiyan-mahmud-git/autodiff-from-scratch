import numpy as np
import pytest

from tinydl import Tensor
from tinydl.optim import SGD, Adam


def test_sgd_update_matches_hand_calculation() -> None:
    parameter = Tensor(np.array([1.0, -2.0]))
    parameter.grad = np.array([0.5, -1.0])
    optimizer = SGD([parameter], learning_rate=0.1)
    optimizer.step()
    np.testing.assert_allclose(parameter.data, np.array([0.95, -1.9]))


def test_momentum_sgd_updates_match_hand_calculation() -> None:
    parameter = Tensor(np.array([1.0]))
    optimizer = SGD([parameter], learning_rate=0.1, momentum=0.9)
    parameter.grad = np.array([2.0])
    optimizer.step()
    np.testing.assert_allclose(parameter.data, np.array([0.8]))
    parameter.grad = np.array([1.0])
    optimizer.step()
    # v_2 = 0.9 * 2 + 1 = 2.8; w_2 = 0.8 - 0.1 * 2.8
    np.testing.assert_allclose(parameter.data, np.array([0.52]))


def test_adam_first_update_matches_hand_calculation() -> None:
    parameter = Tensor(np.array([1.0, -2.0]))
    parameter.grad = np.array([0.5, -1.0])
    optimizer = Adam([parameter], learning_rate=0.1, epsilon=1e-8)
    optimizer.step()
    expected = np.array([0.9, -1.9])
    np.testing.assert_allclose(parameter.data, expected, atol=1e-7)


def test_adam_second_update_uses_bias_corrected_moments() -> None:
    parameter = Tensor(np.array([1.0]))
    optimizer = Adam([parameter], learning_rate=0.1, beta1=0.5, beta2=0.5, epsilon=1e-8)
    parameter.grad = np.array([2.0])
    optimizer.step()
    parameter.grad = np.array([1.0])
    optimizer.step()
    first = 0.5 * (0.5 * 2.0) + 0.5 * 1.0
    second = 0.5 * (0.5 * 4.0) + 0.5 * 1.0
    first_hat = first / (1.0 - 0.5**2)
    second_hat = second / (1.0 - 0.5**2)
    expected = 0.9 - 0.1 * first_hat / (np.sqrt(second_hat) + 1e-8)
    np.testing.assert_allclose(parameter.data, np.array([expected]), atol=1e-7)


def test_optimizer_skips_missing_gradients_and_zeroes_gradients() -> None:
    first = Tensor(np.array([1.0]))
    second = Tensor(np.array([2.0]))
    first.grad = np.array([1.0])
    optimizer = SGD([first, second], learning_rate=0.1)
    optimizer.step()
    np.testing.assert_allclose(second.data, np.array([2.0]))
    optimizer.zero_grad()
    assert first.grad is None
    assert second.grad is None


@pytest.mark.parametrize(
    "make_optimizer",
    [
        lambda parameter: SGD([parameter], learning_rate=0.1, momentum=0.9),
        lambda parameter: Adam([parameter], learning_rate=0.1),
    ],
)
def test_optimizer_zero_grad_method(make_optimizer: object) -> None:
    parameter = Tensor(np.array([1.0]))
    parameter.grad = np.array([1.0])
    optimizer = make_optimizer(parameter)  # type: ignore[operator]
    optimizer.zero_grad()
    assert parameter.grad is None


