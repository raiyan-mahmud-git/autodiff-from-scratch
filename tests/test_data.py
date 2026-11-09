import numpy as np
import pytest

from tinydl.data import batches, prepare_splits


def test_prepare_splits_is_fixed_seed_and_normalizes_images() -> None:
    images = np.arange(12 * 4, dtype=np.uint8).reshape(12, 2, 2)
    labels = np.arange(12, dtype=np.int64)
    first = prepare_splits(images, labels, images[:3], labels[:3], validation_size=3, seed=7)
    second = prepare_splits(images, labels, images[:3], labels[:3], validation_size=3, seed=7)
    assert first.x_train.shape == (9, 4)
    assert first.x_val.shape == (3, 4)
    assert first.x_test.shape == (3, 4)
    assert first.x_train.dtype == np.float64
    assert first.y_train.dtype == np.int64
    np.testing.assert_array_equal(first.x_train, second.x_train)
    assert float(first.x_train.min()) >= 0.0
    assert float(first.x_train.max()) <= 1.0
    assert set(first.y_train).isdisjoint(set(first.y_val))


def test_batching_shuffles_reproducibly_and_keeps_final_partial_batch() -> None:
    images = np.arange(10, dtype=np.float64).reshape(10, 1)
    labels = np.arange(10, dtype=np.int64)
    first = list(batches(images, labels, 4, rng=np.random.default_rng(2)))
    second = list(batches(images, labels, 4, rng=np.random.default_rng(2)))
    assert [len(y) for _, y in first] == [4, 4, 2]
    np.testing.assert_array_equal(np.concatenate([y for _, y in first]), np.concatenate([y for _, y in second]))


def test_batching_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        list(batches(np.ones((2, 1)), np.ones(2, dtype=np.int64), 0, rng=np.random.default_rng(0)))
    with pytest.raises(ValueError, match="counts must match"):
        list(batches(np.ones((2, 1)), np.ones(1, dtype=np.int64), 1, rng=np.random.default_rng(0)))


def test_prepare_splits_rejects_invalid_validation_size() -> None:
    images = np.zeros((2, 2, 2), dtype=np.uint8)
    labels = np.zeros(2, dtype=np.int64)
    with pytest.raises(ValueError, match="validation_size"):
        prepare_splits(images, labels, images, labels, validation_size=2)
