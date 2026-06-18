# test_dataset.py: Tests for MnistDataset in src/data/mnist.py.

import gzip
import os
import struct

import numpy as np
import pytest

from src.data.mnist import MnistDataset


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


# --- synthetic data tests ---

class TestMnistDataset:
    # __len__
    def test_len_train(self, mnist_dir):
        ds = MnistDataset("train", "multiclass", dataset_dir=mnist_dir)
        assert len(ds) == 20

    def test_len_test(self, mnist_dir):
        ds = MnistDataset("test", "multiclass", dataset_dir=mnist_dir)
        assert len(ds) == 20

    # images: shape and dtype
    def test_images_shape_multiclass(self, mnist_dir):
        ds = MnistDataset("train", "multiclass", dataset_dir=mnist_dir)
        assert ds.images.shape == (20, 784)

    def test_images_dtype_float32(self, mnist_dir):
        ds = MnistDataset("train", "multiclass", dataset_dir=mnist_dir)
        assert ds.images.dtype == np.float32

    def test_images_normalized(self, mnist_dir):
        ds = MnistDataset("train", "multiclass", dataset_dir=mnist_dir)
        assert ds.images.min() >= 0.0
        assert ds.images.max() <= 1.0

    # targets: multiclass
    def test_targets_shape_multiclass(self, mnist_dir):
        ds = MnistDataset("train", "multiclass", dataset_dir=mnist_dir)
        assert ds.targets.shape == (20, 10)

    def test_targets_dtype_multiclass(self, mnist_dir):
        ds = MnistDataset("train", "multiclass", dataset_dir=mnist_dir)
        assert ds.targets.dtype == np.float32

    def test_targets_onehot_multiclass(self, mnist_dir):
        ds = MnistDataset("train", "multiclass", dataset_dir=mnist_dir)
        assert np.all(ds.targets.sum(axis=1) == 1.0)
        assert set(np.unique(ds.targets)).issubset({0.0, 1.0})

    # targets: binary
    def test_targets_shape_binary(self, mnist_dir):
        ds = MnistDataset("train", "binary", dataset_dir=mnist_dir)
        assert ds.targets.shape == (20, 1)

    def test_targets_dtype_binary(self, mnist_dir):
        ds = MnistDataset("train", "binary", dataset_dir=mnist_dir)
        assert ds.targets.dtype == np.float32

    def test_targets_values_binary(self, mnist_dir):
        ds = MnistDataset("train", "binary", dataset_dir=mnist_dir)
        assert set(np.unique(ds.targets)).issubset({0.0, 1.0})

    def test_targets_odd_is_one_binary(self, mnist_dir):
        # labels are 0..9 repeated — odd labels → 1, even → 0
        ds = MnistDataset("train", "binary", dataset_dir=mnist_dir)
        assert ds.targets[1, 0] == 1.0
        assert ds.targets[0, 0] == 0.0

    # targets: regression
    def test_targets_shape_regression(self, mnist_dir):
        ds = MnistDataset("train", "regression", dataset_dir=mnist_dir)
        assert ds.targets.shape == (20, 1)

    def test_targets_dtype_regression(self, mnist_dir):
        ds = MnistDataset("train", "regression", dataset_dir=mnist_dir)
        assert ds.targets.dtype == np.float32

    def test_targets_range_regression(self, mnist_dir):
        ds = MnistDataset("train", "regression", dataset_dir=mnist_dir)
        assert ds.targets.min() >= 0.0
        assert ds.targets.max() <= 1.0

    def test_targets_values_regression(self, mnist_dir):
        ds = MnistDataset("train", "regression", dataset_dir=mnist_dir)
        # label 9 (index 9) → 9/9 = 1.0
        assert pytest.approx(ds.targets[9, 0], abs=1e-6) == 1.0
        # label 0 (index 0) → 0/9 = 0.0
        assert pytest.approx(ds.targets[0, 0], abs=1e-6) == 0.0

    # __getitem__
    def test_getitem_returns_tuple(self, mnist_dir):
        ds = MnistDataset("train", "multiclass", dataset_dir=mnist_dir)
        item = ds[0]
        assert isinstance(item, tuple) and len(item) == 2

    def test_getitem_image_shape(self, mnist_dir):
        ds = MnistDataset("train", "multiclass", dataset_dir=mnist_dir)
        image, _ = ds[0]
        assert image.shape == (784,)

    def test_getitem_target_shape_multiclass(self, mnist_dir):
        ds = MnistDataset("train", "multiclass", dataset_dir=mnist_dir)
        _, target = ds[0]
        assert target.shape == (10,)

    def test_getitem_target_shape_binary(self, mnist_dir):
        ds = MnistDataset("train", "binary", dataset_dir=mnist_dir)
        _, target = ds[0]
        assert target.shape == (1,)

    # task_spec
    def test_task_spec_keys(self, mnist_dir):
        ds = MnistDataset("train", "multiclass", dataset_dir=mnist_dir)
        for key in ("task", "output_dim", "target_dtype", "prediction_mode"):
            assert key in ds.task_spec

    def test_task_spec_multiclass(self, mnist_dir):
        ds = MnistDataset("train", "multiclass", dataset_dir=mnist_dir)
        assert ds.task_spec["output_dim"] == 10
        assert ds.task_spec["prediction_mode"] == "argmax"

    def test_task_spec_binary(self, mnist_dir):
        ds = MnistDataset("train", "binary", dataset_dir=mnist_dir)
        assert ds.task_spec["output_dim"] == 1
        assert ds.task_spec["prediction_mode"] == "threshold"

    def test_task_spec_regression(self, mnist_dir):
        ds = MnistDataset("train", "regression", dataset_dir=mnist_dir)
        assert ds.task_spec["output_dim"] == 1
        assert ds.task_spec["prediction_mode"] == "round_clip"

    # error handling
    def test_invalid_split_raises(self, mnist_dir):
        with pytest.raises(ValueError):
            MnistDataset("valid", "multiclass", dataset_dir=mnist_dir)

    def test_invalid_task_raises(self, mnist_dir):
        with pytest.raises(ValueError):
            MnistDataset("train", "unknown", dataset_dir=mnist_dir)


# --- real data integration tests ---

class TestMnistDatasetReal:
    @_skip_no_real_data
    def test_train_multiclass_shapes(self):
        ds = MnistDataset("train", "multiclass")
        assert ds.images.shape == (60000, 784)
        assert ds.targets.shape == (60000, 10)

    @_skip_no_real_data
    def test_test_binary_shapes(self):
        ds = MnistDataset("test", "binary")
        assert ds.images.shape == (10000, 784)
        assert ds.targets.shape == (10000, 1)
