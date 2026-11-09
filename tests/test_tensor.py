import numpy as np
import pytest

from tinydl import Tensor


def test_elementwise_forward_and_backward() -> None:
    x = Tensor([2.0, 3.0])
    y = (x * 2.0 + 1.0).sum()
    np.testing.assert_array_equal(y.data, 12.0)
    y.backward()
    np.testing.assert_array_equal(x.grad, [2.0, 2.0])


def test_repeated_tensor_use_accumulates_both_paths() -> None:
    x = Tensor(3.0)
    (x * x).backward()
    np.testing.assert_array_equal(x.grad, 6.0)


def test_broadcast_gradient_sums_to_input_shape() -> None:
    matrix = Tensor(np.ones((2, 3)))
    row = Tensor(np.array([2.0, 3.0, 4.0]))
    (matrix + row).sum().backward()
    np.testing.assert_array_equal(matrix.grad, np.ones((2, 3)))
    np.testing.assert_array_equal(row.grad, np.array([2.0, 2.0, 2.0]))


def test_broadcast_gradient_across_leading_and_singleton_axes() -> None:
    value = Tensor(np.ones((1, 3, 1)))
    other = Tensor(np.ones((2, 3, 4)))
    (value * other).sum().backward()
    np.testing.assert_array_equal(value.grad, np.full((1, 3, 1), 8.0))


def test_non_scalar_backward_requires_seed() -> None:
    with pytest.raises(ValueError, match="non-scalar"):
        Tensor([1.0, 2.0]).backward()


def test_sum_mean_reshape_and_transpose_gradients() -> None:
    x = Tensor(np.arange(6.0).reshape(2, 3) + 1.0)
    y = x.transpose().reshape(6).mean()
    y.backward()
    np.testing.assert_allclose(x.grad, np.full((2, 3), 1 / 6))


def test_matmul_vector_and_matrix_gradients() -> None:
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    matrix = Tensor(np.arange(6.0).reshape(3, 2))
    (x @ matrix).sum().backward()
    np.testing.assert_array_equal(x.grad, np.array([1.0, 5.0, 9.0]))
    np.testing.assert_array_equal(matrix.grad, np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))


def test_elementwise_nonlinear_gradients() -> None:
    x = Tensor(np.array([0.5, 1.0, 2.0]))
    (x.exp().log() + x.tanh() + x.sigmoid() + x.relu()).sum().backward()
    assert x.grad is not None
    assert np.all(np.isfinite(x.grad))


def test_sub_div_pow_and_neg_gradients() -> None:
    x = Tensor(np.array([2.0, 4.0]))
    y = Tensor(np.array([1.0, 2.0]))
    ((-x) - y + x / y + x**2).sum().backward()
    np.testing.assert_allclose(x.grad, np.array([4.0, 7.5]))
    np.testing.assert_allclose(y.grad, np.array([-3.0, -2.0]))


def test_batched_matmul_broadcast_gradient() -> None:
    left = Tensor(np.ones((2, 1, 3, 4)))
    right = Tensor(np.ones((5, 4, 2)))
    (left @ right).sum().backward()
    np.testing.assert_array_equal(left.grad, np.full((2, 1, 3, 4), 10.0))
    np.testing.assert_array_equal(right.grad, np.full((5, 4, 2), 6.0))

