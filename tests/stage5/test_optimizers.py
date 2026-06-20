# test_optimizers.py: Unit tests for SGD and Adam optimizers.

import numpy as np
import pytest

from src.core.optimizers import SGD, Adam


class FakeModel:
    def __init__(self, params):
        self.params = params
        self.grads = [np.ones_like(p) for p in params]


def make_model():
    return FakeModel([np.array([1.0, 2.0]), np.array([[0.5], [-0.5]])])


class TestSGD:
    def test_params_change_after_step(self):
        model = make_model()
        before = [p.copy() for p in model.params]
        SGD(model, lr=0.1).step()
        for p, b in zip(model.params, before):
            assert not np.allclose(p, b)

    def test_lr_zero_params_unchanged(self):
        model = make_model()
        before = [p.copy() for p in model.params]
        SGD(model, lr=0.0).step()
        for p, b in zip(model.params, before):
            np.testing.assert_array_equal(p, b)

    def test_inplace_update(self):
        model = make_model()
        ids_before = [id(p) for p in model.params]
        SGD(model, lr=0.1).step()
        ids_after = [id(p) for p in model.params]
        assert ids_before == ids_after


class TestAdam:
    def test_params_change_after_step(self):
        model = make_model()
        before = [p.copy() for p in model.params]
        Adam(model, lr=0.001).step()
        for p, b in zip(model.params, before):
            assert not np.allclose(p, b)

    def test_t_increments(self):
        model = make_model()
        opt = Adam(model, lr=0.001)
        opt.step()
        assert opt.iter == 1
        opt.step()
        assert opt.iter == 2

    def test_mv_shape_matches_params(self):
        model = make_model()
        opt = Adam(model, lr=0.001)
        for m, v, p in zip(opt.ms, opt.vs, model.params):
            assert m.shape == p.shape
            assert v.shape == p.shape

    def test_step_stable_with_zero_grad(self):
        model = FakeModel([np.ones((4,))])
        model.grads = [np.zeros((4,))]
        opt = Adam(model, lr=0.001)
        for _ in range(5):
            opt.step()
        assert np.all(np.isfinite(model.params[0]))
