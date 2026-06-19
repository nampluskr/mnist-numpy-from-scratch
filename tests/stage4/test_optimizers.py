# test_optimizers.py: Unit tests for SGD and Adam parameter updates.

import numpy as np
import pytest
from src.core.optimizers import SGD, Adam


class SimpleModel:
    """Minimal model stub with params and grads lists."""
    def __init__(self, param_vals, grad_vals):
        self.params = [np.array(p, dtype=np.float32) for p in param_vals]
        self.grads = [np.array(g, dtype=np.float32) for g in grad_vals]


@pytest.fixture
def model():
    return SimpleModel(
        param_vals=[[1.0, 2.0, 3.0], [0.5]],
        grad_vals=[[0.1, 0.2, 0.3], [1.0]],
    )


class TestSGD:
    def test_step_updates_params(self, model):
        lr = 0.1
        expected = [
            np.array([1.0 - 0.1 * 0.1, 2.0 - 0.1 * 0.2, 3.0 - 0.1 * 0.3], dtype=np.float32),
            np.array([0.5 - 0.1 * 1.0], dtype=np.float32),
        ]
        SGD(model, lr=lr).step()
        for param, exp in zip(model.params, expected):
            np.testing.assert_allclose(param, exp, rtol=1e-6)

    def test_step_is_inplace(self, model):
        refs = [p for p in model.params]
        SGD(model, lr=0.01).step()
        for ref, param in zip(refs, model.params):
            assert ref is param

    def test_step_returns_none(self, model):
        assert SGD(model, lr=0.01).step() is None

    def test_zero_lr_leaves_params_unchanged(self, model):
        original = [p.copy() for p in model.params]
        SGD(model, lr=0.0).step()
        for param, orig in zip(model.params, original):
            np.testing.assert_array_equal(param, orig)

    def test_multiple_steps_accumulate(self, model):
        lr = 0.1
        p0_init = model.params[0].copy()
        g0 = model.grads[0].copy()
        opt = SGD(model, lr=lr)
        opt.step()
        opt.step()
        np.testing.assert_allclose(model.params[0], p0_init - 2 * lr * g0, rtol=1e-6)


class TestAdam:
    def test_step_updates_params(self, model):
        original = [p.copy() for p in model.params]
        Adam(model, lr=0.001).step()
        for param, orig in zip(model.params, original):
            assert not np.array_equal(param, orig)

    def test_step_is_inplace(self, model):
        refs = [p for p in model.params]
        Adam(model, lr=0.001).step()
        for ref, param in zip(refs, model.params):
            assert ref is param

    def test_step_returns_none(self, model):
        assert Adam(model, lr=0.001).step() is None

    def test_iter_increments(self, model):
        opt = Adam(model, lr=0.001)
        assert opt.iter == 0
        opt.step()
        assert opt.iter == 1
        opt.step()
        assert opt.iter == 2

    def test_moment_shapes(self, model):
        opt = Adam(model, lr=0.001)
        opt.step()
        for m, v, p in zip(opt.ms, opt.vs, model.params):
            assert m.shape == p.shape
            assert v.shape == p.shape

    def test_updates_in_negative_grad_direction(self, model):
        original = [p.copy() for p in model.params]
        Adam(model, lr=0.001).step()
        for param, orig, grad in zip(model.params, original, model.grads):
            # gradient is positive, so params should decrease
            assert np.all(param < orig)

    def test_default_betas(self, model):
        opt = Adam(model, lr=0.001)
        assert opt.beta1 == 0.9
        assert opt.beta2 == 0.999
        assert opt.eps == 1e-8
