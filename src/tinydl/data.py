"""MNIST download, IDX parsing, deterministic splitting, and minibatches."""

from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
Labels = npt.NDArray[np.int64]

_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


@dataclass(frozen=True)
class MNISTSplit:
    """Flattened, normalized train/validation/test arrays."""

    x_train: Array
    y_train: Labels
    x_val: Array
    y_val: Labels
    x_test: Array
    y_test: Labels


def _download_file(directory: Path, filename: str) -> Path:
    """Download one compressed MNIST file and verify its published MD5."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    expected = _FILES[filename]
    if path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == expected:
        return path

    temporary = path.with_suffix(path.suffix + ".part")
    digest = hashlib.md5()
    try:
        with (
            urllib.request.urlopen(_BASE_URL + filename, timeout=60) as response,
            temporary.open("wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"checksum mismatch for {filename}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _read_images(path: Path) -> npt.NDArray[np.uint8]:
    """Read a gzip-compressed IDX image file and check its header."""
    with gzip.open(path, "rb") as file:
        header = file.read(16)
        if len(header) != 16:
            raise ValueError(f"truncated IDX image header: {path}")
        magic, count, rows, columns = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"invalid IDX image magic number in {path}")
        values = np.frombuffer(file.read(), dtype=np.uint8)
    expected_count = count * rows * columns
    if values.size != expected_count:
        raise ValueError(f"image data size mismatch in {path}")
    return values.reshape(count, rows, columns)


def _read_labels(path: Path) -> Labels:
    """Read a gzip-compressed IDX label file and check its header."""
    with gzip.open(path, "rb") as file:
        header = file.read(8)
        if len(header) != 8:
            raise ValueError(f"truncated IDX label header: {path}")
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"invalid IDX label magic number in {path}")
        values = np.frombuffer(file.read(), dtype=np.uint8)
    if values.size != count:
        raise ValueError(f"label data size mismatch in {path}")
    return values.astype(np.int64)


def _prepare_images(images: npt.NDArray[np.uint8]) -> Array:
    """Flatten images and scale pixel values from bytes to float64 [0, 1]."""
    return images.reshape(len(images), -1).astype(np.float64) / 255.0


def prepare_splits(
    train_images: npt.NDArray[np.uint8],
    train_labels: npt.NDArray[np.integer],
    test_images: npt.NDArray[np.uint8],
    test_labels: npt.NDArray[np.integer],
    *,
    validation_size: int = 5000,
    seed: int = 0,
) -> MNISTSplit:
    """Make a fixed random train/validation split and normalize all images."""
    if len(train_images) != len(train_labels) or len(test_images) != len(test_labels):
        raise ValueError("image and label counts must match")
    if not 0 < validation_size < len(train_images):
        raise ValueError("validation_size must be between zero and the training count")
    order = np.random.default_rng(seed).permutation(len(train_images))
    val_indices = order[:validation_size]
    train_indices = order[validation_size:]
    return MNISTSplit(
        x_train=_prepare_images(train_images[train_indices]),
        y_train=np.asarray(train_labels[train_indices], dtype=np.int64),
        x_val=_prepare_images(train_images[val_indices]),
        y_val=np.asarray(train_labels[val_indices], dtype=np.int64),
        x_test=_prepare_images(test_images),
        y_test=np.asarray(test_labels, dtype=np.int64),
    )


def load_mnist(
    data_dir: str | Path = "data/mnist",
    *,
    validation_size: int = 5000,
    seed: int = 0,
) -> MNISTSplit:
    """Download, verify, parse, and split the standard MNIST dataset."""
    directory = Path(data_dir)
    files = {name: _download_file(directory, name) for name in _FILES}
    train_images = _read_images(files["train-images-idx3-ubyte.gz"])
    train_labels = _read_labels(files["train-labels-idx1-ubyte.gz"])
    test_images = _read_images(files["t10k-images-idx3-ubyte.gz"])
    test_labels = _read_labels(files["t10k-labels-idx1-ubyte.gz"])
    return prepare_splits(
        train_images,
        train_labels,
        test_images,
        test_labels,
        validation_size=validation_size,
        seed=seed,
    )


def batches(
    images: Array,
    labels: Labels,
    batch_size: int,
    *,
    rng: np.random.Generator,
    shuffle: bool = True,
) -> Iterator[tuple[Array, Labels]]:
    """Yield minibatches, using the supplied generator for reproducible shuffling."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(images) != len(labels):
        raise ValueError("image and label counts must match")
    indices = rng.permutation(len(images)) if shuffle else np.arange(len(images))
    for start in range(0, len(images), batch_size):
        selection = indices[start : start + batch_size]
        yield images[selection], labels[selection]
