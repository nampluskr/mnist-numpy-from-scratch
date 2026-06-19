# test_trainer.py: Unit tests for Trainer.fit() training loop results.

import numpy as np
import pytest
from src.core.trainer import Trainer
from src.task import get_task_spec


class TinyModel:
    """3-param linear model: logits = x @ w + b."""
    def __init__(self, in_dim, out_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.standard_normal((in_dim, out_dim)).astype(np.float32) * 0.1
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)
        self.params = [self.w, self.b]
        self.grads = [self.grad_w, self.grad_b]
        self._x = None

    def forward(self, x):
        self._x = x
        return x @ self.w + self.b

    def backward(self, dout):
        self.grad_w[...] = self._x.T @ dout
        self.grad_b[...] = dout.sum(axis=0)


class NoOpOptimizer:
    def __init__(self, model):
        self.params = model.params
        self.grads = model.grads

    def step(self):
        pass


class SimpleLoader:
    """Yields fixed (x, y) batches."""
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


class TestTrainerFit:
    def test_returns_dict_with_required_keys(self, task, out_dim_map):
        model = TinyModel(4, out_dim_map[task])
        opt = NoOpOptimizer(model)
        trainer = Trainer(model, opt, get_task_spec(task))
        result = trainer.fit(make_loader(task))
        assert set(result.keys()) == {"loss", "metric", "num_samples"}

    def test_num_samples_correct(self, task, out_dim_map):
        model = TinyModel(4, out_dim_map[task])
        opt = NoOpOptimizer(model)
        trainer = Trainer(model, opt, get_task_spec(task))
        result = trainer.fit(make_loader(task, n_batches=3, batch_size=8))
        assert result["num_samples"] == 24

    def test_loss_is_float(self, task, out_dim_map):
        model = TinyModel(4, out_dim_map[task])
        opt = NoOpOptimizer(model)
        trainer = Trainer(model, opt, get_task_spec(task))
        result = trainer.fit(make_loader(task))
        assert isinstance(result["loss"], float)

    def test_metric_is_float(self, task, out_dim_map):
        model = TinyModel(4, out_dim_map[task])
        opt = NoOpOptimizer(model)
        trainer = Trainer(model, opt, get_task_spec(task))
        result = trainer.fit(make_loader(task))
        assert isinstance(result["metric"], float)

    def test_params_updated_with_nonzero_lr(self, task, out_dim_map):
        from src.core.optimizers import SGD
        model = TinyModel(4, out_dim_map[task])
        original_w = model.w.copy()
        opt = SGD(model, lr=0.01)
        trainer = Trainer(model, opt, get_task_spec(task))
        trainer.fit(make_loader(task))
        assert not np.array_equal(model.w, original_w)

    def test_partial_last_batch(self, out_dim_map):
        task = "multiclass"
        batches = [
            (np.ones((8, 4), dtype=np.float32),
             np.eye(10, dtype=np.float32)[[0] * 8]),
            (np.ones((3, 4), dtype=np.float32),
             np.eye(10, dtype=np.float32)[[0] * 3]),
        ]
        model = TinyModel(4, 10)
        opt = NoOpOptimizer(model)
        trainer = Trainer(model, opt, get_task_spec(task))
        result = trainer.fit(SimpleLoader(batches))
        assert result["num_samples"] == 11
