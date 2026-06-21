# test_transforms.py: Tests for src/data/transforms.py image and target transform functions.

import numpy as np
import pytest

from src.data.transforms import normalize, to_flat, one_hot, binarize, to_regression


# --- synthetic data helpers ---

def make_images(n=10, rows=28, cols=28):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (n, rows, cols), dtype=np.uint8)


def make_labels(n=10):
    return np.tile(np.arange(10, dtype=np.uint8), 2)[:n]


# --- normalize ---

class TestNormalize:
    def test_output_dtype(self):
        images = make_images()
        out = normalize(images)
        assert out.dtype == np.float32

    def test_output_shape(self):
        images = make_images(n=5)
        out = normalize(images)
        assert out.shape == (5, 28, 28)

    def test_range_min(self):
        images = make_images()
        out = normalize(images)
        assert out.min() >= 0.0

    def test_range_max(self):
        images = make_images()
        out = normalize(images)
        assert out.max() <= 1.0

    def test_zero_pixel_maps_to_zero(self):
        images = np.zeros((2, 28, 28), dtype=np.uint8)
        out = normalize(images)
        assert np.all(out == 0.0)

    def test_255_pixel_maps_to_one(self):
        images = np.full((2, 28, 28), 255, dtype=np.uint8)
        out = normalize(images)
        assert pytest.approx(out.max(), abs=1e-6) == 1.0


# --- to_flat ---

class TestToFlat:
    def test_output_shape(self):
        images = make_images(n=5)
        out = to_flat(images)
        assert out.shape == (5, 784)

    def test_dtype_preserved(self):
        images = make_images().astype(np.float32)
        out = to_flat(images)
        assert out.dtype == np.float32

    def test_values_preserved(self):
        images = np.arange(28 * 28, dtype=np.float32).reshape(1, 28, 28)
        out = to_flat(images)
        assert np.array_equal(out[0], np.arange(784, dtype=np.float32))

    def test_compose_with_normalize(self):
        images = make_images(n=3)
        out = to_flat(normalize(images))
        assert out.shape == (3, 784)
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# --- one_hot ---

class TestOneHot:
    def test_output_shape(self):
        labels = make_labels(n=20)
        out = one_hot(labels)
        assert out.shape == (20, 10)

    def test_output_dtype(self):
        labels = make_labels()
        out = one_hot(labels)
        assert out.dtype == np.float32

    def test_row_sum_is_one(self):
        labels = make_labels()
        out = one_hot(labels)
        assert np.all(out.sum(axis=1) == 1.0)

    def test_values_are_binary(self):
        labels = make_labels()
        out = one_hot(labels)
        assert set(np.unique(out)).issubset({0.0, 1.0})

    def test_correct_position(self):
        labels = np.array([3, 7], dtype=np.uint8)
        out = one_hot(labels)
        assert out[0, 3] == 1.0
        assert out[1, 7] == 1.0
        assert out[0].sum() == 1.0

    def test_label_zero(self):
        labels = np.array([0], dtype=np.uint8)
        out = one_hot(labels)
        assert out[0, 0] == 1.0

    def test_label_nine(self):
        labels = np.array([9], dtype=np.uint8)
        out = one_hot(labels)
        assert out[0, 9] == 1.0


# --- binarize ---

class TestBinarize:
    def test_output_shape(self):
        labels = make_labels(n=10)
        out = binarize(labels)
        assert out.shape == (10, 1)

    def test_output_dtype(self):
        labels = make_labels()
        out = binarize(labels)
        assert out.dtype == np.float32

    def test_values_are_binary(self):
        labels = make_labels()
        out = binarize(labels)
        assert set(np.unique(out)).issubset({0.0, 1.0})

    def test_odd_label_maps_to_one(self):
        labels = np.array([1, 3, 5, 7, 9], dtype=np.uint8)
        out = binarize(labels)
        assert np.all(out == 1.0)

    def test_even_label_maps_to_zero(self):
        labels = np.array([0, 2, 4, 6, 8], dtype=np.uint8)
        out = binarize(labels)
        assert np.all(out == 0.0)


# --- to_regression ---

class TestToRegression:
    def test_output_shape(self):
        labels = make_labels(n=10)
        out = to_regression(labels)
        assert out.shape == (10, 1)

    def test_output_dtype(self):
        labels = make_labels()
        out = to_regression(labels)
        assert out.dtype == np.float32

    def test_range_min(self):
        labels = make_labels()
        out = to_regression(labels)
        assert out.min() >= 0.0

    def test_range_max(self):
        labels = make_labels()
        out = to_regression(labels)
        assert out.max() <= 1.0

    def test_label_zero_maps_to_zero(self):
        labels = np.array([0], dtype=np.uint8)
        out = to_regression(labels)
        assert pytest.approx(out[0, 0], abs=1e-6) == 0.0

    def test_label_nine_maps_to_one(self):
        labels = np.array([9], dtype=np.uint8)
        out = to_regression(labels)
        assert pytest.approx(out[0, 0], abs=1e-6) == 1.0

    def test_label_values(self):
        labels = np.arange(10, dtype=np.uint8)
        out = to_regression(labels)
        expected = np.arange(10, dtype=np.float32).reshape(-1, 1) / 9.0
        assert np.allclose(out, expected, atol=1e-6)
