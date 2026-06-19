# test_layers.py: Unit tests for Module, Linear, Sigmoid, ReLU, and Sequential.

import numpy as np
import pytest
from src.nn.layers import Linear, Sigmoid, ReLU, Sequential


@pytest.fixture
def x():
    return np.random.default_rng(0).random((4, 8)).astype(np.float32)


class TestLinear:
    def test_forward_shape(self, x):
        layer = Linear(8, 5, seed=0)
        assert layer.forward(x).shape == (4, 5)

    def test_params_count(self):
        layer = Linear(8, 5, seed=0)
        assert len(layer.params) == 2  # w, b
        assert len(layer.grads) == 2

    def test_weight_shape(self):
        layer = Linear(8, 5, seed=0)
        assert layer.w.shape == (8, 5)
        assert layer.b.shape == (5,)

    def test_backward_input_grad_shape(self, x):
        layer = Linear(8, 5, seed=0)
        layer.forward(x)
        dout = np.ones((4, 5), dtype=np.float32)
        grad_x = layer.backward(dout)
        assert grad_x.shape == x.shape

    def test_backward_updates_grads_inplace(self, x):
        layer = Linear(8, 5, seed=0)
        layer.forward(x)
        dout = np.ones((4, 5), dtype=np.float32)
        layer.backward(dout)
        assert not np.all(layer.grad_w == 0)

    def test_seed_reproducibility(self):
        l1 = Linear(8, 5, seed=42)
        l2 = Linear(8, 5, seed=42)
        np.testing.assert_array_equal(l1.w, l2.w)

    def test_he_init_scale(self):
        layer = Linear(784, 256, seed=0)
        expected_std = np.sqrt(2.0 / 784)
        assert abs(layer.w.std() - expected_std) < 0.01


class TestSigmoid:
    def test_forward_shape(self, x):
        layer = Sigmoid()
        assert layer.forward(x).shape == x.shape

    def test_forward_range(self, x):
        out = Sigmoid().forward(x)
        assert np.all(out > 0) and np.all(out < 1)

    def test_backward_shape(self, x):
        layer = Sigmoid()
        out = layer.forward(x)
        dout = np.ones_like(out)
        assert layer.backward(dout).shape == x.shape

    def test_backward_value(self):
        # sigmoid'(x) = s(x)(1-s(x)); at x=0: s=0.5, grad=0.25
        x = np.array([[0.0]], dtype=np.float32)
        layer = Sigmoid()
        layer.forward(x)
        grad = layer.backward(np.ones((1, 1), dtype=np.float32))
        np.testing.assert_allclose(grad, [[0.25]], atol=1e-6)

    def test_no_params(self):
        layer = Sigmoid()
        assert layer.params == [] and layer.grads == []


class TestReLU:
    def test_forward_zeros_negative(self):
        x = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        out = ReLU().forward(x)
        np.testing.assert_array_equal(out, [[0.0, 0.0, 1.0]])

    def test_backward_masks_negative(self):
        x = np.array([[-1.0, 0.5, 2.0]], dtype=np.float32)
        layer = ReLU()
        layer.forward(x)
        grad = layer.backward(np.ones((1, 3), dtype=np.float32))
        np.testing.assert_array_equal(grad, [[0.0, 1.0, 1.0]])


class TestSequential:
    def test_forward_shape(self, x):
        net = Sequential(Linear(8, 5, seed=0), Sigmoid(), Linear(5, 3, seed=1))
        assert net.forward(x).shape == (4, 3)

    def test_params_aggregated(self):
        net = Sequential(Linear(8, 5, seed=0), Sigmoid(), Linear(5, 3, seed=1))
        # Linear(8,5): w+b=2, Sigmoid: 0, Linear(5,3): w+b=2 -> total 4
        assert len(net.params) == 4

    def test_backward_shape(self, x):
        net = Sequential(Linear(8, 5, seed=0), Sigmoid(), Linear(5, 3, seed=1))
        net.forward(x)
        dout = np.ones((4, 3), dtype=np.float32)
        grad_x = net.backward(dout)
        assert grad_x.shape == x.shape

    def test_params_are_references(self):
        # params entries should reference the layer's internal arrays.
        linear = Linear(4, 3, seed=0)
        net = Sequential(linear)
        assert net.params[0] is linear.w
        assert net.params[1] is linear.b
