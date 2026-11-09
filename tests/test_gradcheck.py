import numpy as np
import pytest

from tinydl import Tensor
from tinydl.gradcheck import check_gradients

SIMPLE_TOLERANCE = 1e-6
COMPOUND_TOLERANCE = 1e-5


@pytest.mark.parametrize(
    "function,values",
    [
        (lambda x: (x + 2.5).sum(), [np.array([-0.7, 0.2, 1.3])]),
        (lambda x: (x - 1.5).sum(), [np.array([-0.7, 0.2, 1.3])]),
        (lambda x: (x * 1.7).sum(), [np.array([-0.7, 0.2, 1.3])]),
        (lambda x: (x / 1.7).sum(), [np.array([-0.7, 0.2, 1.3])]),
        (lambda x: (-x).sum(), [np.array([-0.7, 0.2, 1.3])]),
        (lambda x: (x**3).sum(), [np.array([-0.7, 0.2, 1.3])]),
        (lambda x, y: (x @ y).sum(), [np.array([[0.3, 0.8]]), np.array([[1.2], [-0.4]])]),
        (lambda x: x.sum(axis=0).sum(), [np.array([[0.2, 0.8], [1.1, -0.3]])]),
        (lambda x: x.mean(axis=1).sum(), [np.array([[0.2, 0.8], [1.1, -0.3]])]),
        (lambda x: x.reshape(2, 2).sum(), [np.array([0.2, 0.8, 1.1, -0.3])]),
        (lambda x: x.transpose().sum(), [np.array([[0.2, 0.8], [1.1, -0.3]])]),
        (lambda x: x.exp().sum(), [np.array([-0.7, 0.2, 1.3])]),
        (lambda x: x.log().sum(), [np.array([0.7, 1.2, 2.3])]),
        (lambda x: x.relu().sum(), [np.array([-0.7, 0.2, 1.3])]),
        (lambda x: x.tanh().sum(), [np.array([-0.7, 0.2, 1.3])]),
        (lambda x: x.sigmoid().sum(), [np.array([-0.7, 0.2, 1.3])]),
    ],
)
def test_simple_operation_gradients(
    function: object, values: list[np.ndarray]
) -> None:
    assert check_gradients(function, values) < SIMPLE_TOLERANCE  # type: ignore[arg-type]


def test_add_broadcast_gradient() -> None:
    values = [np.arange(6.0).reshape(2, 3) / 7, np.array([0.2, 0.5, 0.9])]
    assert check_gradients(lambda x, row: (x + row).sum(), values) < SIMPLE_TOLERANCE


def test_multiply_broadcast_gradient() -> None:
    values = [np.arange(6.0).reshape(2, 3) / 7 + 0.2, np.array([0.3, 0.8, 1.1])]
    assert check_gradients(lambda x, row: (x * row).sum(), values) < SIMPLE_TOLERANCE


def test_division_gradient_with_broadcasting() -> None:
    values = [np.arange(6.0).reshape(2, 3) / 7 + 0.2, np.array([1.3, 1.8, 2.1])]
    assert check_gradients(lambda x, row: (x / row).sum(), values) < SIMPLE_TOLERANCE


def test_reused_tensor_accumulates_gradients() -> None:
    values = [np.array([0.4, 1.2, 2.0])]
    assert check_gradients(lambda x: (x * x + x).sum(), values) < SIMPLE_TOLERANCE


def test_compound_two_layer_network_gradient() -> None:
    x = np.array([[0.2, -0.3, 0.7], [0.5, 0.4, -0.2]], dtype=np.float64)
    first_weight = np.array([[0.1, -0.4, 0.6, 0.2], [0.7, 0.3, -0.5, 0.8], [-0.2, 0.9, 0.4, -0.6]])
    first_bias = np.array([0.1, 0.2, -0.1, 0.3])
    second_weight = np.array([[0.2, -0.3], [0.5, 0.4], [-0.6, 0.7], [0.3, 0.8]])
    second_bias = np.array([0.1, -0.2])

    def network(data: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor) -> Tensor:
        hidden = (data @ w1 + b1).relu()
        return ((hidden @ w2 + b2) ** 2).mean()

    values = [x, first_weight, first_bias, second_weight, second_bias]
    assert check_gradients(network, values) < COMPOUND_TOLERANCE


def test_rejects_non_scalar_function() -> None:
    with pytest.raises(ValueError, match="scalar-valued"):
        check_gradients(lambda x: x * 2.0, [np.array([0.2, 0.4])])
