# test_predictor.py: Unit tests for task-specific Predictor.predict() post-processing.

import numpy as np
import pytest
from src.core.predictor import Predictor
from src.data.mnist import get_task_spec


class FixedModel:
    """Returns preset logits regardless of input."""
    def __init__(self, logits):
        self._logits = logits
        self.params = []
        self.grads = []

    def forward(self, x):
        return self._logits


class TestPredictorMulticlass:
    def _make(self, logits):
        return Predictor(FixedModel(logits), get_task_spec("multiclass"))

    def test_returns_required_keys(self):
        logits = np.zeros((4, 10), dtype=np.float32)
        result = self._make(logits).predict(np.zeros((4, 784), dtype=np.float32))
        assert set(result.keys()) == {"logits", "predictions"}

    def test_predictions_are_argmax(self):
        logits = np.zeros((3, 10), dtype=np.float32)
        logits[0, 3] = 1.0
        logits[1, 7] = 1.0
        logits[2, 0] = 1.0
        result = self._make(logits).predict(np.zeros((3, 784), dtype=np.float32))
        np.testing.assert_array_equal(result["predictions"], [3, 7, 0])

    def test_predictions_dtype_int32(self):
        logits = np.zeros((2, 10), dtype=np.float32)
        result = self._make(logits).predict(np.zeros((2, 784), dtype=np.float32))
        assert result["predictions"].dtype == np.int32

    def test_predictions_shape(self):
        logits = np.zeros((5, 10), dtype=np.float32)
        result = self._make(logits).predict(np.zeros((5, 784), dtype=np.float32))
        assert result["predictions"].shape == (5,)

    def test_logits_passthrough(self):
        logits = np.ones((2, 10), dtype=np.float32)
        result = self._make(logits).predict(np.zeros((2, 784), dtype=np.float32))
        np.testing.assert_array_equal(result["logits"], logits)


class TestPredictorBinary:
    def _make(self, logits):
        return Predictor(FixedModel(logits), get_task_spec("binary"))

    def test_positive_logit_gives_one(self):
        logits = np.array([[1.0], [2.0]], dtype=np.float32)
        result = self._make(logits).predict(np.zeros((2, 784), dtype=np.float32))
        np.testing.assert_array_equal(result["predictions"], [1, 1])

    def test_negative_logit_gives_zero(self):
        logits = np.array([[-1.0], [-2.0]], dtype=np.float32)
        result = self._make(logits).predict(np.zeros((2, 784), dtype=np.float32))
        np.testing.assert_array_equal(result["predictions"], [0, 0])

    def test_zero_logit_gives_one(self):
        # sigmoid(0) = 0.5 >= 0.5 gives 1.
        logits = np.array([[0.0]], dtype=np.float32)
        result = self._make(logits).predict(np.zeros((1, 784), dtype=np.float32))
        np.testing.assert_array_equal(result["predictions"], [1])

    def test_predictions_shape_flat(self):
        logits = np.zeros((4, 1), dtype=np.float32)
        result = self._make(logits).predict(np.zeros((4, 784), dtype=np.float32))
        assert result["predictions"].shape == (4,)

    def test_predictions_dtype_int32(self):
        logits = np.zeros((3, 1), dtype=np.float32)
        result = self._make(logits).predict(np.zeros((3, 784), dtype=np.float32))
        assert result["predictions"].dtype == np.int32


class TestPredictorRegression:
    def _make(self, logits):
        return Predictor(FixedModel(logits), get_task_spec("regression"))

    def test_maps_to_digit_range(self):
        # logit=0.5 gives 0.5*9=4.5, then Python rounding decides 4 or 5.
        logits = np.array([[0.0], [1.0 / 9.0], [9.0 / 9.0]], dtype=np.float32)
        result = self._make(logits).predict(np.zeros((3, 784), dtype=np.float32))
        assert all(0 <= p <= 9 for p in result["predictions"])

    def test_clips_below_zero(self):
        logits = np.array([[-1.0]], dtype=np.float32)
        result = self._make(logits).predict(np.zeros((1, 784), dtype=np.float32))
        assert result["predictions"][0] == 0

    def test_clips_above_nine(self):
        logits = np.array([[2.0]], dtype=np.float32)
        result = self._make(logits).predict(np.zeros((1, 784), dtype=np.float32))
        assert result["predictions"][0] == 9

    def test_predictions_shape_flat(self):
        logits = np.zeros((4, 1), dtype=np.float32)
        result = self._make(logits).predict(np.zeros((4, 784), dtype=np.float32))
        assert result["predictions"].shape == (4,)

    def test_predictions_dtype_int32(self):
        logits = np.zeros((3, 1), dtype=np.float32)
        result = self._make(logits).predict(np.zeros((3, 784), dtype=np.float32))
        assert result["predictions"].dtype == np.int32
