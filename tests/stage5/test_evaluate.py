# test_evaluate.py: scripts/evaluate.py main() 반환값 및 체크포인트 로딩 테스트

import argparse
import gzip
import struct

import numpy as np
import pytest

from scripts.evaluate import main, build_config
from scripts.train import main as train_main


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


def make_args(mnist_dir, task="multiclass", checkpoint=None):
    return argparse.Namespace(
        task=task,
        batch_size=8,
        seed=0,
        dataset_dir=mnist_dir,
        checkpoint=checkpoint,
    )


def make_train_args(mnist_dir, task="multiclass", checkpoint=None):
    return argparse.Namespace(
        task=task,
        epochs=1,
        batch_size=8,
        lr=0.01,
        seed=0,
        dataset_dir=mnist_dir,
        checkpoint=checkpoint,
    )


# --- build_config ---

class TestBuildConfig:
    def test_keys_present(self, mnist_dir):
        args = make_args(mnist_dir)
        config = build_config(args)
        assert {"dataset_dir", "task", "batch_size", "num_epochs", "seed"} <= set(config.keys())

    def test_num_epochs_zero(self, mnist_dir):
        args = make_args(mnist_dir)
        config = build_config(args)
        assert config["num_epochs"] == 0

    def test_task_mapped(self, mnist_dir):
        args = make_args(mnist_dir, task="regression")
        config = build_config(args)
        assert config["task"] == "regression"


# --- main() 반환값 ---

class TestEvaluateMain:
    @pytest.fixture(params=["multiclass", "binary", "regression"])
    def task(self, request):
        return request.param

    def test_returns_dict(self, mnist_dir, task):
        result = main(make_args(mnist_dir, task=task))
        assert isinstance(result, dict)

    def test_required_keys(self, mnist_dir, task):
        result = main(make_args(mnist_dir, task=task))
        assert set(result.keys()) == {"loss", "metric", "num_samples"}

    def test_loss_is_float(self, mnist_dir, task):
        result = main(make_args(mnist_dir, task=task))
        assert isinstance(result["loss"], float)

    def test_metric_is_float(self, mnist_dir, task):
        result = main(make_args(mnist_dir, task=task))
        assert isinstance(result["metric"], float)

    def test_num_samples_positive(self, mnist_dir, task):
        result = main(make_args(mnist_dir, task=task))
        assert result["num_samples"] > 0


# --- 체크포인트 로딩 ---

class TestEvaluateCheckpoint:
    def test_loads_checkpoint_without_error(self, mnist_dir, tmp_path):
        ckpt_path = str(tmp_path / "model")
        train_main(make_train_args(mnist_dir, checkpoint=ckpt_path))
        result = main(make_args(mnist_dir, checkpoint=ckpt_path))
        assert isinstance(result, dict)
