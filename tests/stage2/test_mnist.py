# test_mnist.py: Tests for src/data/mnist.py load_images and load_labels interface.

import gzip
import os
import struct

import numpy as np
import pytest

from src.data.mnist import load_images, load_labels


# --- synthetic data helpers ---

def make_image_gz(path, n=10, rows=28, cols=28):
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, n, rows, cols))
        f.write(np.zeros(n * rows * cols, dtype=np.uint8).tobytes())


def make_label_gz(path, n=10):
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">II", 2049, n))
        f.write(np.arange(n, dtype=np.uint8).tobytes())


# --- real data skip marker ---

_MNIST_DIR = "/mnt/d/datasets/mnist"
_skip_no_real_data = pytest.mark.skipif(
    not os.path.exists(_MNIST_DIR), reason="MNIST dataset not available"
)


# --- synthetic data tests ---

class TestLoadImages:
    def test_train_shape(self, tmp_path):
        make_image_gz(str(tmp_path / "train-images-idx3-ubyte.gz"), n=60)
        make_label_gz(str(tmp_path / "train-labels-idx1-ubyte.gz"), n=60)
        images = load_images("train", dataset_dir=str(tmp_path))
        assert images.shape == (60, 28, 28)

    def test_dtype_uint8(self, tmp_path):
        make_image_gz(str(tmp_path / "train-images-idx3-ubyte.gz"))
        make_label_gz(str(tmp_path / "train-labels-idx1-ubyte.gz"))
        images = load_images("train", dataset_dir=str(tmp_path))
        assert images.dtype == np.uint8

    def test_pixel_range(self, tmp_path):
        make_image_gz(str(tmp_path / "train-images-idx3-ubyte.gz"))
        make_label_gz(str(tmp_path / "train-labels-idx1-ubyte.gz"))
        images = load_images("train", dataset_dir=str(tmp_path))
        assert images.min() >= 0
        assert images.max() <= 255

    def test_test_split_shape(self, tmp_path):
        make_image_gz(str(tmp_path / "t10k-images-idx3-ubyte.gz"), n=20)
        make_label_gz(str(tmp_path / "t10k-labels-idx1-ubyte.gz"), n=20)
        images = load_images("test", dataset_dir=str(tmp_path))
        assert images.shape == (20, 28, 28)

    def test_invalid_split_raises(self, tmp_path):
        with pytest.raises(ValueError):
            load_images("valid", dataset_dir=str(tmp_path))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_images("train", dataset_dir=str(tmp_path))


class TestLoadLabels:
    def test_train_shape(self, tmp_path):
        make_image_gz(str(tmp_path / "train-images-idx3-ubyte.gz"), n=10)
        make_label_gz(str(tmp_path / "train-labels-idx1-ubyte.gz"), n=10)
        labels = load_labels("train", dataset_dir=str(tmp_path))
        assert labels.shape == (10,)

    def test_dtype_uint8(self, tmp_path):
        make_image_gz(str(tmp_path / "train-images-idx3-ubyte.gz"))
        make_label_gz(str(tmp_path / "train-labels-idx1-ubyte.gz"))
        labels = load_labels("train", dataset_dir=str(tmp_path))
        assert labels.dtype == np.uint8

    def test_label_values(self, tmp_path):
        make_image_gz(str(tmp_path / "train-images-idx3-ubyte.gz"), n=10)
        make_label_gz(str(tmp_path / "train-labels-idx1-ubyte.gz"), n=10)
        labels = load_labels("train", dataset_dir=str(tmp_path))
        assert set(labels.tolist()) == set(range(10))

    def test_test_split_shape(self, tmp_path):
        make_image_gz(str(tmp_path / "t10k-images-idx3-ubyte.gz"), n=20)
        make_label_gz(str(tmp_path / "t10k-labels-idx1-ubyte.gz"), n=20)
        labels = load_labels("test", dataset_dir=str(tmp_path))
        assert labels.shape == (20,)

    def test_invalid_split_raises(self, tmp_path):
        with pytest.raises(ValueError):
            load_labels("valid", dataset_dir=str(tmp_path))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_labels("train", dataset_dir=str(tmp_path))


# --- real data integration tests ---

class TestLoadMnistReal:
    @_skip_no_real_data
    def test_train_images_shape(self):
        images = load_images("train")
        assert images.shape == (60000, 28, 28)
        assert images.dtype == np.uint8

    @_skip_no_real_data
    def test_test_images_shape(self):
        images = load_images("test")
        assert images.shape == (10000, 28, 28)

    @_skip_no_real_data
    def test_train_labels_shape(self):
        labels = load_labels("train")
        assert labels.shape == (60000,)
        assert labels.dtype == np.uint8

    @_skip_no_real_data
    def test_label_range(self):
        labels = load_labels("train")
        assert labels.min() == 0
        assert labels.max() == 9
