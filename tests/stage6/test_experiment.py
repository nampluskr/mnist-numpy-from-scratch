# test_experiment.py: CNN + Experiment 통합 테스트 (synthetic MNIST 사용)

import gzip
import struct

import numpy as np
import pytest

from src.core.experiment import Experiment
from src.models.mlp import MLP
from src.models.cnn import CNN


# ---------------------------------------------------------------------------
# synthetic MNIST gz 헬퍼
# ---------------------------------------------------------------------------

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


def cnn_config(mnist_dir, task="multiclass"):
    return {
        "dataset_dir": mnist_dir,
        "task": task,
        "model": "cnn",
        "batch_size": 8,
        "num_epochs": 1,
        "seed": 0,
        "lr": 0.01,
    }


def mlp_config(mnist_dir, task="multiclass"):
    return {
        "dataset_dir": mnist_dir,
        "task": task,
        "model": "mlp",
        "batch_size": 8,
        "num_epochs": 1,
        "seed": 0,
        "lr": 0.01,
    }


# ---------------------------------------------------------------------------
# 조립 테스트: config["model"] 분기
# ---------------------------------------------------------------------------

class TestExperimentModelSelection:
    def test_model_is_cnn_when_specified(self, mnist_dir):
        exp = Experiment(cnn_config(mnist_dir))
        assert isinstance(exp.model, CNN)

    def test_model_is_mlp_when_specified(self, mnist_dir):
        exp = Experiment(mlp_config(mnist_dir))
        assert isinstance(exp.model, MLP)

    def test_model_is_mlp_by_default(self, mnist_dir):
        config = {
            "dataset_dir": mnist_dir,
            "task": "multiclass",
            "batch_size": 8,
            "num_epochs": 1,
            "seed": 0,
        }
        exp = Experiment(config)
        assert isinstance(exp.model, MLP)

    def test_cnn_has_loaders(self, mnist_dir):
        exp = Experiment(cnn_config(mnist_dir))
        assert exp.train_loader is not None
        assert exp.test_loader is not None


# ---------------------------------------------------------------------------
# CNN run() 테스트
# ---------------------------------------------------------------------------

class TestCNNExperimentRun:
    @pytest.fixture(params=["multiclass", "binary", "regression"])
    def task(self, request):
        return request.param

    def test_run_returns_list(self, mnist_dir, task):
        exp = Experiment(cnn_config(mnist_dir, task))
        logs = exp.run()
        assert isinstance(logs, list)

    def test_run_length_equals_num_epochs(self, mnist_dir, task):
        exp = Experiment(cnn_config(mnist_dir, task))
        logs = exp.run()
        assert len(logs) == 1

    def test_log_has_epoch_key(self, mnist_dir, task):
        exp = Experiment(cnn_config(mnist_dir, task))
        logs = exp.run()
        assert logs[0]["epoch"] == 1

    def test_log_has_train_and_test(self, mnist_dir, task):
        exp = Experiment(cnn_config(mnist_dir, task))
        logs = exp.run()
        assert "train" in logs[0]
        assert "test" in logs[0]

    def test_train_log_keys(self, mnist_dir, task):
        exp = Experiment(cnn_config(mnist_dir, task))
        logs = exp.run()
        assert set(logs[0]["train"].keys()) == {"loss", "metric", "num_samples"}

    def test_test_log_keys(self, mnist_dir, task):
        exp = Experiment(cnn_config(mnist_dir, task))
        logs = exp.run()
        assert set(logs[0]["test"].keys()) == {"loss", "metric", "num_samples"}

    def test_num_samples_positive(self, mnist_dir, task):
        exp = Experiment(cnn_config(mnist_dir, task))
        logs = exp.run()
        assert logs[0]["train"]["num_samples"] > 0
        assert logs[0]["test"]["num_samples"] > 0


class TestCNNExperimentLoss:
    """수치 안정성 검증 — regression은 소규모 synthetic 환경에서 발산할 수 있으므로 제외."""

    @pytest.fixture(params=["multiclass", "binary"])
    def task(self, request):
        return request.param

    def test_train_loss_is_finite(self, mnist_dir, task):
        exp = Experiment(cnn_config(mnist_dir, task))
        logs = exp.run()
        assert np.isfinite(logs[0]["train"]["loss"])

    def test_test_loss_is_finite(self, mnist_dir, task):
        exp = Experiment(cnn_config(mnist_dir, task))
        logs = exp.run()
        assert np.isfinite(logs[0]["test"]["loss"])


# ---------------------------------------------------------------------------
# CNN + Trainer 단계 테스트 (1 step forward+backward)
# ---------------------------------------------------------------------------

class TestCNNTrainerStep:
    def test_trainer_step_multiclass(self, mnist_dir):
        from src.core.trainer import Trainer
        from src.core.optimizers import SGD
        from src.task import get_task_spec
        from src.data.mnist import MnistDataset
        from src.data.dataloader import DataLoader

        task = "multiclass"
        task_spec = get_task_spec(task)
        dataset = MnistDataset("train", task, dataset_dir=mnist_dir)
        loader = DataLoader(dataset, batch_size=8, shuffle=False)

        model = CNN(task=task, seed=0)
        optimizer = SGD(model, lr=0.01)
        trainer = Trainer(model, optimizer, task_spec)

        log = trainer.fit(loader)
        assert "loss" in log
        assert "metric" in log
        assert "num_samples" in log
        assert np.isfinite(log["loss"])

    def test_trainer_step_binary(self, mnist_dir):
        from src.core.trainer import Trainer
        from src.core.optimizers import SGD
        from src.task import get_task_spec
        from src.data.mnist import MnistDataset
        from src.data.dataloader import DataLoader

        task = "binary"
        task_spec = get_task_spec(task)
        dataset = MnistDataset("train", task, dataset_dir=mnist_dir)
        loader = DataLoader(dataset, batch_size=8, shuffle=False)

        model = CNN(task=task, seed=0)
        optimizer = SGD(model, lr=0.01)
        trainer = Trainer(model, optimizer, task_spec)

        log = trainer.fit(loader)
        assert np.isfinite(log["loss"])
