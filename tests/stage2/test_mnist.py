# test_mnist.py: Tests for src/data/mnist.py — load_mnist interface and behavior.

import gzip
import os
import struct

import numpy as np
import pytest

from src.data.mnist import load_mnist


# --- synthetic data helpers ---

def make_image_gz(path, n=10, rows=28, cols=28):
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, n, rows, cols))
        f.write(np.zeros(n * rows * cols, dtype=np.uint8).tobytes())


def make_label_gz(path, n=10):
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">II", 2049, n))
        f.write(np.arange(n, dtype=np.uint8).tobytes())


# --- shape and dtype tests ---

def test_train_images_shape(tmp_path):
    make_image_gz(str(tmp_path / "train-images-idx3-ubyte.gz"), n=60)
    make_label_gz(str(tmp_path / "train-labels-idx1-ubyte.gz"), n=60)
    images, _ = load_mnist("train", dataset_dir=str(tmp_path))
    assert images.shape == (60, 28, 28)


def test_train_labels_shape(tmp_path):
    make_image_gz(str(tmp_path / "train-images-idx3-ubyte.gz"), n=60)
    make_label_gz(str(tmp_path / "train-labels-idx1-ubyte.gz"), n=60)
    _, labels = load_mnist("train", dataset_dir=str(tmp_path))
    assert labels.shape == (60,)


def test_images_dtype_uint8(tmp_path):
    make_image_gz(str(tmp_path / "train-images-idx3-ubyte.gz"))
    make_label_gz(str(tmp_path / "train-labels-idx1-ubyte.gz"))
    images, _ = load_mnist("train", dataset_dir=str(tmp_path))
    assert images.dtype == np.uint8


def test_labels_dtype_uint8(tmp_path):
    make_image_gz(str(tmp_path / "train-images-idx3-ubyte.gz"))
    make_label_gz(str(tmp_path / "train-labels-idx1-ubyte.gz"))
    _, labels = load_mnist("train", dataset_dir=str(tmp_path))
    assert labels.dtype == np.uint8


def test_test_split_shape(tmp_path):
    make_image_gz(str(tmp_path / "t10k-images-idx3-ubyte.gz"), n=20)
    make_label_gz(str(tmp_path / "t10k-labels-idx1-ubyte.gz"), n=20)
    images, labels = load_mnist("test", dataset_dir=str(tmp_path))
    assert images.shape == (20, 28, 28)
    assert labels.shape == (20,)


# --- value range tests ---

def test_image_pixel_range(tmp_path):
    make_image_gz(str(tmp_path / "train-images-idx3-ubyte.gz"))
    make_label_gz(str(tmp_path / "train-labels-idx1-ubyte.gz"))
    images, _ = load_mnist("train", dataset_dir=str(tmp_path))
    assert images.min() >= 0
    assert images.max() <= 255


def test_label_values(tmp_path):
    make_image_gz(str(tmp_path / "train-images-idx3-ubyte.gz"), n=10)
    make_label_gz(str(tmp_path / "train-labels-idx1-ubyte.gz"), n=10)
    _, labels = load_mnist("train", dataset_dir=str(tmp_path))
    assert set(labels.tolist()) == set(range(10))


# --- error handling tests ---

def test_invalid_split_raises(tmp_path):
    with pytest.raises(ValueError):
        load_mnist("valid", dataset_dir=str(tmp_path))


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mnist("train", dataset_dir=str(tmp_path))


# --- real data integration tests ---

MNIST_DIR = "/mnt/d/datasets/mnist"


@pytest.mark.skipif(not os.path.exists(MNIST_DIR), reason="MNIST dataset not available")
def test_real_train_shape():
    images, labels = load_mnist("train")
    assert images.shape == (60000, 28, 28)
    assert labels.shape == (60000,)


@pytest.mark.skipif(not os.path.exists(MNIST_DIR), reason="MNIST dataset not available")
def test_real_test_shape():
    images, labels = load_mnist("test")
    assert images.shape == (10000, 28, 28)
    assert labels.shape == (10000,)


@pytest.mark.skipif(not os.path.exists(MNIST_DIR), reason="MNIST dataset not available")
def test_real_train_dtypes():
    images, labels = load_mnist("train")
    assert images.dtype == np.uint8
    assert labels.dtype == np.uint8


@pytest.mark.skipif(not os.path.exists(MNIST_DIR), reason="MNIST dataset not available")
def test_real_label_range():
    _, labels = load_mnist("train")
    assert labels.min() == 0
    assert labels.max() == 9
