# test_dataset.py: Tests for MNISTDataset, MulticlassDataset, BinaryDataset, RegressionDataset.

import gzip
import os
import struct

import numpy as np
import pytest

from src.data.datasets import MNISTDataset, MulticlassDataset, BinaryDataset, RegressionDataset
from src.data import transforms as T


# --- synthetic data helpers ---

def make_image_gz(path, n=20, rows=28, cols=28):
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, n, rows, cols))
        rng = np.random.default_rng(0)
        f.write(rng.integers(0, 256, n * rows * cols, dtype=np.uint8).tobytes())


def make_label_gz(path, n=20):
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">II", 2049, n))
        labels = np.tile(np.arange(10, dtype=np.uint8), 2)[:n]
        f.write(labels.tobytes())


@pytest.fixture
def mnist_dir(tmp_path):
    make_image_gz(str(tmp_path / "train-images-idx3-ubyte.gz"), n=20)
    make_label_gz(str(tmp_path / "train-labels-idx1-ubyte.gz"), n=20)
    make_image_gz(str(tmp_path / "t10k-images-idx3-ubyte.gz"), n=20)
    make_label_gz(str(tmp_path / "t10k-labels-idx1-ubyte.gz"), n=20)
    return str(tmp_path)


# --- real data skip marker ---

_MNIST_DIR = "/mnt/d/datasets/mnist"
_skip_no_real_data = pytest.mark.skipif(
    not os.path.exists(_MNIST_DIR), reason="MNIST dataset not available"
)


# --- MNISTDataset base class tests ---

class TestMNISTDataset:
    def test_no_transform_returns_raw(self, mnist_dir):
        ds = MNISTDataset("train", dataset_dir=mnist_dir)
        assert ds.images.shape == (20, 28, 28)
        assert ds.images.dtype == np.uint8

    def test_custom_transform(self, mnist_dir):
        ds = MNISTDataset("train", transform=lambda x: T.to_flat(T.normalize(x)), dataset_dir=mnist_dir)
        assert ds.images.shape == (20, 784)
        assert ds.images.dtype == np.float32

    def test_custom_target_transform(self, mnist_dir):
        ds = MNISTDataset("train", target_transform=T.one_hot, dataset_dir=mnist_dir)
        assert ds.targets.shape == (20, 10)

    def test_len(self, mnist_dir):
        ds = MNISTDataset("train", dataset_dir=mnist_dir)
        assert len(ds) == 20

    def test_getitem_returns_tuple(self, mnist_dir):
        ds = MNISTDataset("train", dataset_dir=mnist_dir)
        item = ds[0]
        assert isinstance(item, tuple) and len(item) == 2

    def test_invalid_split_raises(self, mnist_dir):
        with pytest.raises(ValueError):
            MNISTDataset("valid", dataset_dir=mnist_dir)


# --- MulticlassDataset tests ---

class TestMulticlassDataset:
    def test_len(self, mnist_dir):
        ds = MulticlassDataset("train", dataset_dir=mnist_dir)
        assert len(ds) == 20

    def test_images_shape(self, mnist_dir):
        ds = MulticlassDataset("train", dataset_dir=mnist_dir)
        assert ds.images.shape == (20, 784)

    def test_images_dtype(self, mnist_dir):
        ds = MulticlassDataset("train", dataset_dir=mnist_dir)
        assert ds.images.dtype == np.float32

    def test_images_normalized(self, mnist_dir):
        ds = MulticlassDataset("train", dataset_dir=mnist_dir)
        assert ds.images.min() >= 0.0
        assert ds.images.max() <= 1.0

    def test_targets_shape(self, mnist_dir):
        ds = MulticlassDataset("train", dataset_dir=mnist_dir)
        assert ds.targets.shape == (20, 10)

    def test_targets_dtype(self, mnist_dir):
        ds = MulticlassDataset("train", dataset_dir=mnist_dir)
        assert ds.targets.dtype == np.float32

    def test_targets_onehot(self, mnist_dir):
        ds = MulticlassDataset("train", dataset_dir=mnist_dir)
        assert np.all(ds.targets.sum(axis=1) == 1.0)
        assert set(np.unique(ds.targets)).issubset({0.0, 1.0})

    def test_getitem_shapes(self, mnist_dir):
        ds = MulticlassDataset("train", dataset_dir=mnist_dir)
        image, target = ds[0]
        assert image.shape == (784,)
        assert target.shape == (10,)

    def test_custom_transform_override(self, mnist_dir):
        ds = MulticlassDataset("train", transform=lambda x: x.reshape(len(x), -1), dataset_dir=mnist_dir)
        assert ds.images.dtype == np.uint8


# --- BinaryDataset tests ---

class TestBinaryDataset:
    def test_targets_shape(self, mnist_dir):
        ds = BinaryDataset("train", dataset_dir=mnist_dir)
        assert ds.targets.shape == (20, 1)

    def test_targets_dtype(self, mnist_dir):
        ds = BinaryDataset("train", dataset_dir=mnist_dir)
        assert ds.targets.dtype == np.float32

    def test_targets_binary_values(self, mnist_dir):
        ds = BinaryDataset("train", dataset_dir=mnist_dir)
        assert set(np.unique(ds.targets)).issubset({0.0, 1.0})

    def test_targets_odd_is_one(self, mnist_dir):
        ds = BinaryDataset("train", dataset_dir=mnist_dir)
        assert ds.targets[1, 0] == 1.0
        assert ds.targets[0, 0] == 0.0

    def test_getitem_target_shape(self, mnist_dir):
        ds = BinaryDataset("train", dataset_dir=mnist_dir)
        _, target = ds[0]
        assert target.shape == (1,)


# --- RegressionDataset tests ---

class TestRegressionDataset:
    def test_targets_shape(self, mnist_dir):
        ds = RegressionDataset("train", dataset_dir=mnist_dir)
        assert ds.targets.shape == (20, 1)

    def test_targets_dtype(self, mnist_dir):
        ds = RegressionDataset("train", dataset_dir=mnist_dir)
        assert ds.targets.dtype == np.float32

    def test_targets_range(self, mnist_dir):
        ds = RegressionDataset("train", dataset_dir=mnist_dir)
        assert ds.targets.min() >= 0.0
        assert ds.targets.max() <= 1.0

    def test_targets_values(self, mnist_dir):
        ds = RegressionDataset("train", dataset_dir=mnist_dir)
        assert pytest.approx(ds.targets[9, 0], abs=1e-6) == 1.0
        assert pytest.approx(ds.targets[0, 0], abs=1e-6) == 0.0

    def test_getitem_target_shape(self, mnist_dir):
        ds = RegressionDataset("train", dataset_dir=mnist_dir)
        _, target = ds[0]
        assert target.shape == (1,)


# --- real data integration tests ---

class TestDatasetsReal:
    @_skip_no_real_data
    def test_multiclass_train_shapes(self):
        ds = MulticlassDataset("train")
        assert ds.images.shape == (60000, 784)
        assert ds.targets.shape == (60000, 10)

    @_skip_no_real_data
    def test_binary_test_shapes(self):
        ds = BinaryDataset("test")
        assert ds.images.shape == (10000, 784)
        assert ds.targets.shape == (10000, 1)

    @_skip_no_real_data
    def test_regression_test_shapes(self):
        ds = RegressionDataset("test")
        assert ds.images.shape == (10000, 784)
        assert ds.targets.shape == (10000, 1)
