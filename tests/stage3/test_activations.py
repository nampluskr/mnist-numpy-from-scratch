# test_activations.py: 활성화 함수 출력 형태, 범위, 수치 정확도 테스트

import numpy as np
import pytest
from src.nn.activations import sigmoid, softmax, relu, identity


@pytest.fixture
def x1d():
    return np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)


@pytest.fixture
def x2d():
    return np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]], dtype=np.float32)


class TestSigmoid:
    def test_output_shape(self, x1d):
        assert sigmoid(x1d).shape == x1d.shape

    def test_range(self, x1d):
        out = sigmoid(x1d)
        assert np.all(out > 0.0) and np.all(out < 1.0)

    def test_midpoint(self):
        np.testing.assert_allclose(sigmoid(np.array([0.0])), [0.5], atol=1e-6)

    def test_symmetry(self, x1d):
        # sigmoid(-x) == 1 - sigmoid(x)
        np.testing.assert_allclose(sigmoid(-x1d), 1.0 - sigmoid(x1d), atol=1e-6)

    def test_numerical_stability_large_negative(self):
        out = sigmoid(np.array([-500.0], dtype=np.float32))
        assert np.isfinite(out).all()


class TestSoftmax:
    def test_output_shape(self, x2d):
        assert softmax(x2d).shape == x2d.shape

    def test_sums_to_one(self, x2d):
        sums = softmax(x2d).sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(2), atol=1e-6)

    def test_all_positive(self, x2d):
        assert np.all(softmax(x2d) > 0.0)

    def test_numerical_stability_large_values(self):
        x = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
        out = softmax(x)
        assert np.isfinite(out).all()
        np.testing.assert_allclose(out.sum(), 1.0, atol=1e-6)


class TestReLU:
    def test_output_shape(self, x1d):
        assert relu(x1d).shape == x1d.shape

    def test_negative_zeroed(self, x1d):
        out = relu(x1d)
        assert np.all(out[x1d < 0] == 0.0)

    def test_positive_unchanged(self, x1d):
        out = relu(x1d)
        np.testing.assert_array_equal(out[x1d > 0], x1d[x1d > 0])


class TestIdentity:
    def test_passthrough(self, x2d):
        np.testing.assert_array_equal(identity(x2d), x2d)
