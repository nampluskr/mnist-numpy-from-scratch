# test_mlp.py: MLP 생성, forward shape, backward/update 흐름 테스트

import numpy as np
import pytest
from src.models.mlp import MLP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batch():
    rng = np.random.default_rng(0)
    x = rng.random((8, 784), dtype=np.float32)
    return x


@pytest.fixture
def mlp_multiclass():
    return MLP(task="multiclass")


@pytest.fixture
def mlp_binary():
    return MLP(task="binary")


@pytest.fixture
def mlp_regression():
    return MLP(task="regression")


# ---------------------------------------------------------------------------
# 생성
# ---------------------------------------------------------------------------

class TestMLPInit:
    def test_multiclass_output_dim(self, mlp_multiclass):
        assert mlp_multiclass.output_dim == 10

    def test_binary_output_dim(self, mlp_binary):
        assert mlp_binary.output_dim == 1

    def test_regression_output_dim(self, mlp_regression):
        assert mlp_regression.output_dim == 1

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            MLP(task="invalid")

    def test_weight_shapes(self, mlp_multiclass):
        # 784->256, 256->128, 128->10
        assert mlp_multiclass.params["W1"].shape == (784, 256)
        assert mlp_multiclass.params["b1"].shape == (256,)
        assert mlp_multiclass.params["W2"].shape == (256, 128)
        assert mlp_multiclass.params["b2"].shape == (128,)
        assert mlp_multiclass.params["W3"].shape == (128, 10)
        assert mlp_multiclass.params["b3"].shape == (10,)


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------

class TestMLPForward:
    def test_multiclass_output_shape(self, mlp_multiclass, batch):
        preds = mlp_multiclass.forward(batch)
        assert preds.shape == (8, 10)

    def test_binary_output_shape(self, mlp_binary, batch):
        preds = mlp_binary.forward(batch)
        assert preds.shape == (8, 1)

    def test_regression_output_shape(self, mlp_regression, batch):
        preds = mlp_regression.forward(batch)
        assert preds.shape == (8, 1)

    def test_multiclass_softmax_sums_to_one(self, mlp_multiclass, batch):
        preds = mlp_multiclass.forward(batch)
        sums = preds.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(8), atol=1e-5)

    def test_binary_output_in_range(self, mlp_binary, batch):
        preds = mlp_binary.forward(batch)
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

    def test_output_dtype_float32(self, mlp_multiclass, batch):
        preds = mlp_multiclass.forward(batch)
        assert preds.dtype == np.float32


# ---------------------------------------------------------------------------
# backward + update
# ---------------------------------------------------------------------------

class TestMLPBackwardUpdate:
    def test_backward_returns_grad_input(self, mlp_multiclass, batch):
        mlp_multiclass.forward(batch)
        y = np.zeros((8, 10), dtype=np.float32)
        y[np.arange(8), np.arange(8) % 10] = 1.0
        grad_out = (mlp_multiclass.forward(batch) - y) / 8
        grad_x = mlp_multiclass.backward(grad_out)
        assert grad_x.shape == batch.shape

    def test_update_changes_weights(self, mlp_multiclass, batch):
        w1_before = mlp_multiclass.params["W1"].copy()
        preds = mlp_multiclass.forward(batch)
        y = np.zeros((8, 10), dtype=np.float32)
        y[np.arange(8), np.arange(8) % 10] = 1.0
        grad_out = (preds - y) / 8
        mlp_multiclass.backward(grad_out)
        mlp_multiclass.update(lr=0.01)
        assert not np.allclose(mlp_multiclass.params["W1"], w1_before)

    def test_repeated_updates_reduce_loss(self):
        rng = np.random.default_rng(1)
        x = rng.random((32, 784), dtype=np.float32)
        y = np.zeros((32, 10), dtype=np.float32)
        y[np.arange(32), rng.integers(0, 10, 32)] = 1.0

        mlp = MLP(task="multiclass")
        losses = []
        for _ in range(20):
            preds = mlp.forward(x)
            preds_clipped = np.clip(preds, 1e-7, 1.0)
            loss = -np.mean(np.sum(y * np.log(preds_clipped), axis=1))
            losses.append(loss)
            grad_out = (preds - y) / 32
            mlp.backward(grad_out)
            mlp.update(lr=0.1)

        assert losses[-1] < losses[0]
