# test_metrics.py: 평가 지표 함수 테스트

import numpy as np
import pytest
from src.nn.metrics import accuracy, binary_accuracy, r2_score


@pytest.fixture
def rng():
	return np.random.default_rng(0)


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
