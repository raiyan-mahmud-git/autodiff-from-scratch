import numpy as np
import pytest

from tinydl import Tensor
from tinydl.functional import cross_entropy, log_softmax
from tinydl.nn import Dropout, Linear, Module, ReLU, Sequential, Tanh


def test_log_softmax_is_normalized_and_stable_for_large_logits() -> None:
    logits = Tensor(np.array([[10000.0, 10001.0, 9999.0]], dtype=np.float64))
    result = log_softmax(logits)
    assert np.all(np.isfinite(result.data))
    np.testing.assert_allclose(np.exp(result.data).sum(axis=-1), [1.0])


def test_cross_entropy_class_indices_and_gradient() -> None:
    logits = Tensor(np.array([[2.0, 1.0, 0.1], [0.2, 0.3, 1.5]], dtype=np.float64))
    loss = cross_entropy(logits, Tensor(np.array([0, 2])))
    assert np.isfinite(loss.data)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


def test_cross_entropy_one_hot_targets() -> None:
    logits = Tensor(np.array([[2.0, 1.0], [1.0, 3.0]]))
    target = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]))
    loss = cross_entropy(logits, target)
    expected = (np.logaddexp(0.0, -1.0) + np.logaddexp(0.0, -2.0)) / 2
    np.testing.assert_allclose(loss.data, expected)


def test_cross_entropy_rejects_invalid_class_index() -> None:
    with pytest.raises(ValueError, match="class range"):
        cross_entropy(Tensor(np.zeros((2, 3))), Tensor(np.array([0, 3])))


def test_linear_shape_parameters_and_initialization() -> None:
    layer = Linear(3, 2, rng=np.random.default_rng(5))
    output = layer(Tensor(np.ones((4, 3))))
    assert output.shape == (4, 2)
    assert len(layer.parameters()) == 2
    layer.zero_grad()
    assert all(parameter.grad is None for parameter in layer.parameters())


def test_sequential_activations_and_parameter_discovery() -> None:
    model = Sequential(
        Linear(2, 3, rng=np.random.default_rng(1)),
        ReLU(),
        Linear(3, 1, rng=np.random.default_rng(2)),
        Tanh(),
    )
    output = model(Tensor(np.ones((5, 2))))
    assert output.shape == (5, 1)
    assert len(model.parameters()) == 4
    model.eval()
    assert all(not module.training for module in model.layers)


def test_dropout_inverted_scaling_and_eval_mode() -> None:
    dropout = Dropout(0.5, seed=3)
    inputs = Tensor(np.ones(1000))
    output = dropout(inputs)
    assert set(np.unique(output.data)).issubset({0.0, 2.0})
    np.testing.assert_allclose(output.data.mean(), 1.0, atol=0.15)
    dropout.eval()
    np.testing.assert_array_equal(dropout(inputs).data, inputs.data)


def test_module_base_is_abstract_by_convention() -> None:
    with pytest.raises(NotImplementedError):
        Module()(Tensor([1.0]))
