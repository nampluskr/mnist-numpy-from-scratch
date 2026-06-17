# test_mlp.py: MLP 생성, forward shape, backward/update 흐름 테스트

import numpy as np
import pytest
from src.models.mlp import MLP
from src.nn.losses import cross_entropy_grad, binary_cross_entropy_grad, mse_grad


@pytest.fixture
def batch():
    return np.random.default_rng(0).random((8, 784)).astype(np.float32)


@pytest.fixture
def mlp_multiclass():
    return MLP(task="multiclass", seed=0)


@pytest.fixture
def mlp_binary():
    return MLP(task="binary", seed=0)


@pytest.fixture
def mlp_regression():
    return MLP(task="regression", seed=0)


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

    def test_params_count(self, mlp_multiclass):
        # Linear(784,256): w+b, Linear(256,128): w+b, Linear(128,10): w+b → 6
        assert len(mlp_multiclass.params) == 6

    def test_grads_count(self, mlp_multiclass):
        assert len(mlp_multiclass.grads) == 6

    def test_first_weight_shape(self, mlp_multiclass):
        assert mlp_multiclass.params[0].shape == (784, 256)

    def test_seed_reproducibility(self):
        m1 = MLP(task="multiclass", seed=42)
        m2 = MLP(task="multiclass", seed=42)
        np.testing.assert_array_equal(m1.params[0], m2.params[0])


class TestMLPForward:
    def test_multiclass_output_shape(self, mlp_multiclass, batch):
        logits = mlp_multiclass.forward(batch)
        assert logits.shape == (8, 10)

    def test_binary_output_shape(self, mlp_binary, batch):
        logits = mlp_binary.forward(batch)
        assert logits.shape == (8, 1)

    def test_regression_output_shape(self, mlp_regression, batch):
        logits = mlp_regression.forward(batch)
        assert logits.shape == (8, 1)

    def test_output_dtype_float32(self, mlp_multiclass, batch):
        assert mlp_multiclass.forward(batch).dtype == np.float32

    def test_logits_not_probability(self, mlp_multiclass, batch):
        # raw logit 이므로 row sum ≠ 1
        logits = mlp_multiclass.forward(batch)
        sums = logits.sum(axis=1)
        assert not np.allclose(sums, np.ones(8), atol=1e-3)


class TestMLPBackward:
    def test_backward_returns_grad_input_shape(self, mlp_multiclass, batch):
        mlp_multiclass.forward(batch)
        targets = np.zeros((8, 10), dtype=np.float32)
        targets[np.arange(8), np.arange(8) % 10] = 1.0
        logits = mlp_multiclass.forward(batch)
        grad_out = cross_entropy_grad(logits, targets)
        grad_x = mlp_multiclass.backward(grad_out)
        assert grad_x.shape == batch.shape

    def test_grads_populated_after_backward(self, mlp_multiclass, batch):
        logits = mlp_multiclass.forward(batch)
        targets = np.zeros((8, 10), dtype=np.float32)
        targets[np.arange(8), np.arange(8) % 10] = 1.0
        grad_out = cross_entropy_grad(logits, targets)
        mlp_multiclass.backward(grad_out)
        assert not np.all(mlp_multiclass.grads[0] == 0)

    def test_grads_are_references_to_layer_grads(self, mlp_multiclass, batch):
        # MLP.grads 리스트 원소가 레이어 내부 grad 배열과 동일 객체여야 한다
        logits = mlp_multiclass.forward(batch)
        targets = np.zeros((8, 10), dtype=np.float32)
        targets[np.arange(8), np.arange(8) % 10] = 1.0
        grad_out = cross_entropy_grad(logits, targets)
        mlp_multiclass.backward(grad_out)
        # Sequential.grads[0] 은 첫 Linear 의 grad_w 와 동일
        assert mlp_multiclass.grads[0] is mlp_multiclass.net.layers[0].grad_w


class TestMLPTrainingLoop:
    def test_loss_decreases_over_steps(self):
        rng = np.random.default_rng(1)
        x = rng.random((32, 784)).astype(np.float32)
        targets = np.zeros((32, 10), dtype=np.float32)
        targets[np.arange(32), rng.integers(0, 10, 32)] = 1.0

        from src.nn.losses import cross_entropy
        mlp = MLP(task="multiclass", seed=0)
        lr = 0.1
        losses = []

        for _ in range(20):
            logits = mlp.forward(x)
            losses.append(cross_entropy(logits, targets))
            grad_out = cross_entropy_grad(logits, targets)
            mlp.backward(grad_out)
            for param, grad in zip(mlp.params, mlp.grads):
                param -= lr * grad

        assert losses[-1] < losses[0]
