# test_losses.py: 손실 함수 및 평가 지표 테스트 (logit 입력 기준)

import numpy as np
import pytest
from src.nn.losses import (
    cross_entropy, cross_entropy_grad,
    binary_cross_entropy, binary_cross_entropy_grad,
    mse, mse_grad,
    accuracy, binary_accuracy, r2_score,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestCrossEntropy:
    def test_returns_scalar(self, rng):
        logits = rng.standard_normal((8, 10)).astype(np.float32)
        targets = np.zeros((8, 10), dtype=np.float32)
        targets[np.arange(8), rng.integers(0, 10, 8)] = 1.0
        loss = cross_entropy(logits, targets)
        assert loss.ndim == 0

    def test_non_negative(self, rng):
        logits = rng.standard_normal((8, 10)).astype(np.float32)
        targets = np.zeros((8, 10), dtype=np.float32)
        targets[np.arange(8), rng.integers(0, 10, 8)] = 1.0
        assert cross_entropy(logits, targets) >= 0.0

    def test_perfect_prediction_low_loss(self):
        # 정답 클래스에 매우 큰 logit → loss ≈ 0
        logits = np.array([[10.0, -10.0], [-10.0, 10.0]], dtype=np.float32)
        targets = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert cross_entropy(logits, targets) < 0.01


class TestCrossEntropyGrad:
    def test_grad_shape(self, rng):
        logits = rng.standard_normal((8, 10)).astype(np.float32)
        targets = np.zeros((8, 10), dtype=np.float32)
        targets[np.arange(8), rng.integers(0, 10, 8)] = 1.0
        grad = cross_entropy_grad(logits, targets)
        assert grad.shape == logits.shape

    def test_grad_sums_to_zero(self, rng):
        # softmax rows sum to 1, targets rows sum to 1 → row sums of grad = 0
        logits = rng.standard_normal((8, 10)).astype(np.float32)
        targets = np.zeros((8, 10), dtype=np.float32)
        targets[np.arange(8), rng.integers(0, 10, 8)] = 1.0
        grad = cross_entropy_grad(logits, targets)
        np.testing.assert_allclose(grad.sum(axis=1), np.zeros(8), atol=1e-6)

    def test_grad_scale_by_batch(self, rng):
        # grad 의 스케일이 1/N
        N = 16
        logits = rng.standard_normal((N, 5)).astype(np.float32)
        targets = np.zeros((N, 5), dtype=np.float32)
        targets[np.arange(N), rng.integers(0, 5, N)] = 1.0
        grad = cross_entropy_grad(logits, targets)
        assert abs(grad.sum()) < 1.0  # 합이 상대적으로 작음


class TestBinaryCrossEntropy:
    def test_returns_scalar(self, rng):
        logits = rng.standard_normal((8, 1)).astype(np.float32)
        targets = rng.integers(0, 2, (8, 1)).astype(np.float32)
        loss = binary_cross_entropy(logits, targets)
        assert loss.ndim == 0

    def test_non_negative(self, rng):
        logits = rng.standard_normal((8, 1)).astype(np.float32)
        targets = rng.integers(0, 2, (8, 1)).astype(np.float32)
        assert binary_cross_entropy(logits, targets) >= 0.0

    def test_perfect_prediction_low_loss(self):
        logits = np.array([[10.0], [-10.0]], dtype=np.float32)
        targets = np.array([[1.0], [0.0]], dtype=np.float32)
        assert binary_cross_entropy(logits, targets) < 0.01


class TestBinaryCrossEntropyGrad:
    def test_grad_shape(self, rng):
        logits = rng.standard_normal((8, 1)).astype(np.float32)
        targets = rng.integers(0, 2, (8, 1)).astype(np.float32)
        assert binary_cross_entropy_grad(logits, targets).shape == (8, 1)

    def test_positive_logit_positive_target_negative_grad(self):
        # logit > 0 → sigmoid > 0.5, target=1 → grad < 0 불가, target=0 → grad > 0
        logits = np.array([[2.0]], dtype=np.float32)
        targets = np.array([[0.0]], dtype=np.float32)
        grad = binary_cross_entropy_grad(logits, targets)
        assert grad[0, 0] > 0


class TestMSE:
    def test_returns_scalar(self, rng):
        preds = rng.random((8, 1)).astype(np.float32)
        targets = rng.random((8, 1)).astype(np.float32)
        assert mse(preds, targets).ndim == 0

    def test_zero_on_perfect(self):
        x = np.array([[1.0], [2.0]], dtype=np.float32)
        np.testing.assert_allclose(mse(x, x), 0.0, atol=1e-7)

    def test_known_value(self):
        preds = np.array([[1.0], [3.0]], dtype=np.float32)
        targets = np.array([[0.0], [0.0]], dtype=np.float32)
        np.testing.assert_allclose(mse(preds, targets), 5.0, atol=1e-6)


class TestMSEGrad:
    def test_grad_shape(self, rng):
        preds = rng.random((8, 1)).astype(np.float32)
        targets = rng.random((8, 1)).astype(np.float32)
        assert mse_grad(preds, targets).shape == (8, 1)

    def test_known_value(self):
        preds = np.array([[1.0], [3.0]], dtype=np.float32)
        targets = np.array([[0.0], [0.0]], dtype=np.float32)
        # 2 * (preds - targets) / N = 2 * [[1],[3]] / 2 = [[1],[3]]
        np.testing.assert_allclose(mse_grad(preds, targets), [[1.0], [3.0]], atol=1e-6)


class TestMetrics:
    def test_accuracy_perfect(self):
        logits = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=np.float32)
        targets = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(accuracy(logits, targets), 1.0)

    def test_accuracy_zero(self):
        logits = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=np.float32)
        targets = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(accuracy(logits, targets), 0.0)

    def test_binary_accuracy_positive_logit(self):
        # logit >= 0 → pred=1
        logits = np.array([[1.0], [-1.0]], dtype=np.float32)
        targets = np.array([[1.0], [0.0]], dtype=np.float32)
        np.testing.assert_allclose(binary_accuracy(logits, targets), 1.0)

    def test_r2_perfect(self):
        x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        np.testing.assert_allclose(r2_score(x, x), 1.0, atol=1e-6)

    def test_r2_range(self, rng):
        preds = rng.random((20, 1)).astype(np.float32)
        targets = rng.random((20, 1)).astype(np.float32)
        score = r2_score(preds, targets)
        assert score <= 1.0
