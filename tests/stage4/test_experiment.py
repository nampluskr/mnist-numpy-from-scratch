# test_experiment.py: Unit tests for Experiment assembly and run() results.

import gzip
import struct

import numpy as np
import pytest

from src.core.experiment import Experiment
from src.core.trainer import Trainer
from src.core.evaluator import Evaluator
from src.core.predictor import Predictor
from src.models.mlp import MLP


# --- synthetic MNIST gz helpers ---

def make_image_gz(path, n=40):
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, n, 28, 28))
        rng = np.random.default_rng(0)
        f.write(rng.integers(0, 256, n * 28 * 28, dtype=np.uint8).tobytes())


def make_label_gz(path, n=40):
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">II", 2049, n))
        labels = np.tile(np.arange(10, dtype=np.uint8), 4)[:n]
        f.write(labels.tobytes())


@pytest.fixture
def mnist_dir(tmp_path):
    make_image_gz(str(tmp_path / "train-images-idx3-ubyte.gz"), n=40)
    make_label_gz(str(tmp_path / "train-labels-idx1-ubyte.gz"), n=40)
    make_image_gz(str(tmp_path / "t10k-images-idx3-ubyte.gz"), n=20)
    make_label_gz(str(tmp_path / "t10k-labels-idx1-ubyte.gz"), n=20)
    return str(tmp_path)


def small_config(mnist_dir, task="multiclass"):
    return {
        "dataset_dir": mnist_dir,
        "task": task,
        "batch_size": 8,
        "num_epochs": 2,
        "seed": 0,
        "lr": 0.01,
    }


# --- assembly tests ---

class TestExperimentAssembly:
    def test_has_train_loader(self, mnist_dir):
        exp = Experiment(small_config(mnist_dir))
        assert exp.train_loader is not None

    def test_has_test_loader(self, mnist_dir):
        exp = Experiment(small_config(mnist_dir))
        assert exp.test_loader is not None

    def test_model_is_mlp(self, mnist_dir):
        exp = Experiment(small_config(mnist_dir))
        assert isinstance(exp.model, MLP)

    def test_trainer_is_trainer(self, mnist_dir):
        exp = Experiment(small_config(mnist_dir))
        assert isinstance(exp.trainer, Trainer)

    def test_evaluator_is_evaluator(self, mnist_dir):
        exp = Experiment(small_config(mnist_dir))
        assert isinstance(exp.evaluator, Evaluator)

    def test_predictor_is_predictor(self, mnist_dir):
        exp = Experiment(small_config(mnist_dir))
        assert isinstance(exp.predictor, Predictor)


# --- run() tests ---

class TestExperimentRun:
    @pytest.fixture(params=["multiclass", "binary", "regression"])
    def task(self, request):
        return request.param

    def test_run_returns_list(self, mnist_dir, task):
        exp = Experiment(small_config(mnist_dir, task))
        logs = exp.run()
        assert isinstance(logs, list)

    def test_run_length_equals_num_epochs(self, mnist_dir, task):
        exp = Experiment(small_config(mnist_dir, task))
        logs = exp.run()
        assert len(logs) == 2

    def test_log_has_epoch_key(self, mnist_dir, task):
        exp = Experiment(small_config(mnist_dir, task))
        logs = exp.run()
        assert logs[0]["epoch"] == 1
        assert logs[1]["epoch"] == 2

    def test_log_has_train_and_test(self, mnist_dir, task):
        exp = Experiment(small_config(mnist_dir, task))
        logs = exp.run()
        assert "train" in logs[0]
        assert "test" in logs[0]

    def test_train_log_keys(self, mnist_dir, task):
        exp = Experiment(small_config(mnist_dir, task))
        logs = exp.run()
        assert set(logs[0]["train"].keys()) == {"loss", "metric", "num_samples"}

    def test_test_log_keys(self, mnist_dir, task):
        exp = Experiment(small_config(mnist_dir, task))
        logs = exp.run()
        assert set(logs[0]["test"].keys()) == {"loss", "metric", "num_samples"}
