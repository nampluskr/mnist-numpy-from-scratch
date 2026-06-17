# test_train.py: scripts/train.py main() 반환값 및 체크포인트 저장 테스트

import argparse
import gzip
import struct

import numpy as np
import pytest

from scripts.train import main, build_config


def _cupy_gpu_available():
    try:
        import cupy
        cupy.pad(cupy.ones((2, 2), dtype=cupy.float32), 1)
        return True
    except Exception:
        return False


_skip_no_gpu = pytest.mark.skipif(not _cupy_gpu_available(), reason="CuPy GPU not available")


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


def make_args(mnist_dir, task="multiclass", epochs=2, checkpoint=None, model="mlp"):
    return argparse.Namespace(
        task=task,
        model=model,
        epochs=epochs,
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
        assert {"dataset_dir", "task", "batch_size", "num_epochs", "seed", "lr"} <= set(config.keys())

    def test_num_epochs_mapped(self, mnist_dir):
        args = make_args(mnist_dir, epochs=3)
        config = build_config(args)
        assert config["num_epochs"] == 3

    def test_task_mapped(self, mnist_dir):
        args = make_args(mnist_dir, task="binary")
        config = build_config(args)
        assert config["task"] == "binary"


# --- main() 반환값 ---

class TestTrainMain:
    @pytest.fixture(params=["multiclass", "binary", "regression"])
    def task(self, request):
        return request.param

    def test_returns_list(self, mnist_dir, task):
        logs = main(make_args(mnist_dir, task=task))
        assert isinstance(logs, list)

    def test_length_equals_epochs(self, mnist_dir, task):
        logs = main(make_args(mnist_dir, task=task, epochs=2))
        assert len(logs) == 2

    def test_log_has_epoch_key(self, mnist_dir, task):
        logs = main(make_args(mnist_dir, task=task, epochs=2))
        assert logs[0]["epoch"] == 1
        assert logs[1]["epoch"] == 2

    def test_log_has_train_test(self, mnist_dir, task):
        logs = main(make_args(mnist_dir, task=task))
        assert "train" in logs[0] and "test" in logs[0]

    def test_train_log_keys(self, mnist_dir, task):
        logs = main(make_args(mnist_dir, task=task))
        assert set(logs[0]["train"].keys()) == {"loss", "metric", "num_samples"}

    def test_test_log_keys(self, mnist_dir, task):
        logs = main(make_args(mnist_dir, task=task))
        assert set(logs[0]["test"].keys()) == {"loss", "metric", "num_samples"}


# --- --model 플래그 ---

class TestTrainModel:
    def test_model_key_in_config(self, mnist_dir):
        args = make_args(mnist_dir, model="mlp")
        config = build_config(args)
        assert config["model"] == "mlp"

    def test_cnn_model_key_in_config(self, mnist_dir):
        args = make_args(mnist_dir, model="cnn")
        config = build_config(args)
        assert config["model"] == "cnn"

    @_skip_no_gpu
    def test_cnn_returns_list(self, mnist_dir):
        logs = main(make_args(mnist_dir, model="cnn", epochs=1))
        assert isinstance(logs, list)

    @_skip_no_gpu
    def test_cnn_log_has_train_test(self, mnist_dir):
        logs = main(make_args(mnist_dir, model="cnn", epochs=1))
        assert "train" in logs[0] and "test" in logs[0]


# --- 체크포인트 저장 ---

class TestTrainCheckpoint:
    def test_checkpoint_file_created(self, mnist_dir, tmp_path):
        ckpt_path = str(tmp_path / "model")
        main(make_args(mnist_dir, epochs=1, checkpoint=ckpt_path))
        assert (tmp_path / "model.npz").exists()

    def test_no_checkpoint_when_none(self, mnist_dir, tmp_path):
        main(make_args(mnist_dir, epochs=1, checkpoint=None))
        assert not any((tmp_path).glob("*.npz"))
