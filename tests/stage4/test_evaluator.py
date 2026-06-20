# test_evaluator.py: Unit tests for Evaluator.evaluate() results.

import numpy as np
import pytest
from src.core.evaluator import Evaluator
from src.data.mnist import get_task_spec


class TinyModel:
    """3-param linear model: logits = x @ w + b."""
    def __init__(self, in_dim, out_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.standard_normal((in_dim, out_dim)).astype(np.float32) * 0.1
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.params = [self.w, self.b]
        self.grads = [np.zeros_like(self.w), np.zeros_like(self.b)]

    def forward(self, x):
        return x @ self.w + self.b


class SimpleLoader:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


def make_loader(task, n_batches=3, batch_size=8, seed=0):
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(n_batches):
        x = rng.standard_normal((batch_size, 4)).astype(np.float32)
        if task == "multiclass":
            labels = rng.integers(0, 10, size=batch_size)
            y = np.zeros((batch_size, 10), dtype=np.float32)
            y[np.arange(batch_size), labels] = 1.0
        elif task == "binary":
            y = rng.integers(0, 2, size=(batch_size, 1)).astype(np.float32)
        else:
            y = rng.random((batch_size, 1)).astype(np.float32)
        batches.append((x, y))
    return SimpleLoader(batches)


@pytest.fixture(params=["multiclass", "binary", "regression"])
def task(request):
    return request.param


@pytest.fixture
def out_dim_map():
    return {"multiclass": 10, "binary": 1, "regression": 1}


class TestEvaluatorEvaluate:
    def test_returns_dict_with_required_keys(self, task, out_dim_map):
        model = TinyModel(4, out_dim_map[task])
        evaluator = Evaluator(model, get_task_spec(task))
        result = evaluator.evaluate(make_loader(task))
        assert set(result.keys()) == {"loss", "metric", "num_samples"}

    def test_num_samples_correct(self, task, out_dim_map):
        model = TinyModel(4, out_dim_map[task])
        evaluator = Evaluator(model, get_task_spec(task))
        result = evaluator.evaluate(make_loader(task, n_batches=3, batch_size=8))
        assert result["num_samples"] == 24

    def test_loss_is_float(self, task, out_dim_map):
        model = TinyModel(4, out_dim_map[task])
        evaluator = Evaluator(model, get_task_spec(task))
        result = evaluator.evaluate(make_loader(task))
        assert isinstance(result["loss"], float)

    def test_metric_is_float(self, task, out_dim_map):
        model = TinyModel(4, out_dim_map[task])
        evaluator = Evaluator(model, get_task_spec(task))
        result = evaluator.evaluate(make_loader(task))
        assert isinstance(result["metric"], float)

    def test_params_not_modified(self, task, out_dim_map):
        model = TinyModel(4, out_dim_map[task])
        original_w = model.w.copy()
        evaluator = Evaluator(model, get_task_spec(task))
        evaluator.evaluate(make_loader(task))
        np.testing.assert_array_equal(model.w, original_w)

    def test_partial_last_batch(self, out_dim_map):
        task = "multiclass"
        batches = [
            (np.ones((8, 4), dtype=np.float32),
             np.eye(10, dtype=np.float32)[[0] * 8]),
            (np.ones((3, 4), dtype=np.float32),
             np.eye(10, dtype=np.float32)[[0] * 3]),
        ]
        model = TinyModel(4, 10)
        evaluator = Evaluator(model, get_task_spec(task))
        result = evaluator.evaluate(SimpleLoader(batches))
        assert result["num_samples"] == 11
