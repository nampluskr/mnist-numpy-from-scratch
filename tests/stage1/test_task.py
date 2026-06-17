# test_task.py: Unit tests for get_task_spec() and transform_targets().

import numpy as np
import pytest

from src.task import get_task_spec, transform_targets


# ---------------------------------------------------------------------------
# get_task_spec
# ---------------------------------------------------------------------------

def test_get_task_spec_returns_dict():
    assert isinstance(get_task_spec("multiclass"), dict)


def test_get_task_spec_required_keys():
    required = {"task", "output_dim", "target_dtype", "prediction_mode"}
    for task in ("multiclass", "binary", "regression"):
        assert required <= set(get_task_spec(task).keys())


def test_get_task_spec_multiclass():
    spec = get_task_spec("multiclass")
    assert spec["task"] == "multiclass"
    assert spec["output_dim"] == 10
    assert spec["target_dtype"] == "float32"
    assert spec["prediction_mode"] == "argmax"


def test_get_task_spec_binary():
    spec = get_task_spec("binary")
    assert spec["task"] == "binary"
    assert spec["output_dim"] == 1
    assert spec["target_dtype"] == "float32"
    assert spec["prediction_mode"] == "threshold"


def test_get_task_spec_regression():
    spec = get_task_spec("regression")
    assert spec["task"] == "regression"
    assert spec["output_dim"] == 1
    assert spec["target_dtype"] == "float32"
    assert spec["prediction_mode"] == "round_clip"


def test_get_task_spec_invalid_raises():
    with pytest.raises(ValueError):
        get_task_spec("unknown")


# ---------------------------------------------------------------------------
# transform_targets
# ---------------------------------------------------------------------------

LABELS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint8)


def test_transform_targets_multiclass_shape():
    targets = transform_targets(LABELS, "multiclass")
    assert targets.shape == (10, 10)


def test_transform_targets_multiclass_dtype():
    targets = transform_targets(LABELS, "multiclass")
    assert targets.dtype == np.float32


def test_transform_targets_multiclass_one_hot():
    targets = transform_targets(LABELS, "multiclass")
    assert np.all(targets.sum(axis=1) == 1.0)
    for i in range(10):
        assert targets[i, i] == 1.0


def test_transform_targets_binary_shape():
    targets = transform_targets(LABELS, "binary")
    assert targets.shape == (10, 1)


def test_transform_targets_binary_dtype():
    targets = transform_targets(LABELS, "binary")
    assert targets.dtype == np.float32


def test_transform_targets_binary_values():
    targets = transform_targets(LABELS, "binary")
    # odd labels → 1, even labels → 0
    expected = np.array([[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]], dtype=np.float32)
    np.testing.assert_array_equal(targets, expected)


def test_transform_targets_regression_shape():
    targets = transform_targets(LABELS, "regression")
    assert targets.shape == (10, 1)


def test_transform_targets_regression_dtype():
    targets = transform_targets(LABELS, "regression")
    assert targets.dtype == np.float32


def test_transform_targets_regression_values():
    targets = transform_targets(LABELS, "regression")
    expected = (LABELS.astype(np.float32) / 9.0).reshape(-1, 1)
    np.testing.assert_allclose(targets, expected, rtol=1e-6)


def test_transform_targets_invalid_raises():
    with pytest.raises(ValueError):
        transform_targets(LABELS, "unknown")
