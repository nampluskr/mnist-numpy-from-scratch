# test_task.py: Unit tests for get_task_spec() and transform_targets().

import numpy as np
import pytest

from src.task import get_task_spec, transform_targets


LABELS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint8)


# ---------------------------------------------------------------------------
# get_task_spec
# ---------------------------------------------------------------------------

class TestGetTaskSpec:
    def test_returns_dict(self):
        assert isinstance(get_task_spec("multiclass"), dict)

    def test_required_keys(self):
        required = {"task", "output_dim", "target_dtype", "prediction_mode"}
        for task in ("multiclass", "binary", "regression"):
            assert required <= set(get_task_spec(task).keys())

    def test_multiclass(self):
        spec = get_task_spec("multiclass")
        assert spec["task"] == "multiclass"
        assert spec["output_dim"] == 10
        assert spec["target_dtype"] == "float32"
        assert spec["prediction_mode"] == "argmax"

    def test_binary(self):
        spec = get_task_spec("binary")
        assert spec["task"] == "binary"
        assert spec["output_dim"] == 1
        assert spec["target_dtype"] == "float32"
        assert spec["prediction_mode"] == "threshold"

    def test_regression(self):
        spec = get_task_spec("regression")
        assert spec["task"] == "regression"
        assert spec["output_dim"] == 1
        assert spec["target_dtype"] == "float32"
        assert spec["prediction_mode"] == "round_clip"

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            get_task_spec("unknown")


# ---------------------------------------------------------------------------
# transform_targets
# ---------------------------------------------------------------------------

class TestTransformTargets:
    def test_multiclass_shape(self):
        targets = transform_targets(LABELS, "multiclass")
        assert targets.shape == (10, 10)

    def test_multiclass_dtype(self):
        targets = transform_targets(LABELS, "multiclass")
        assert targets.dtype == np.float32

    def test_multiclass_one_hot(self):
        targets = transform_targets(LABELS, "multiclass")
        assert np.all(targets.sum(axis=1) == 1.0)
        for i in range(10):
            assert targets[i, i] == 1.0

    def test_binary_shape(self):
        targets = transform_targets(LABELS, "binary")
        assert targets.shape == (10, 1)

    def test_binary_dtype(self):
        targets = transform_targets(LABELS, "binary")
        assert targets.dtype == np.float32

    def test_binary_values(self):
        targets = transform_targets(LABELS, "binary")
        # Odd labels map to 1, even labels map to 0.
        expected = np.array([[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]], dtype=np.float32)
        np.testing.assert_array_equal(targets, expected)

    def test_regression_shape(self):
        targets = transform_targets(LABELS, "regression")
        assert targets.shape == (10, 1)

    def test_regression_dtype(self):
        targets = transform_targets(LABELS, "regression")
        assert targets.dtype == np.float32

    def test_regression_values(self):
        targets = transform_targets(LABELS, "regression")
        expected = (LABELS.astype(np.float32) / 9.0).reshape(-1, 1)
        np.testing.assert_allclose(targets, expected, rtol=1e-6)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            transform_targets(LABELS, "unknown")
