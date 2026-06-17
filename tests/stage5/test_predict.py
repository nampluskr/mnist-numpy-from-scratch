# test_predict.py: scripts/predict.py main() 반환값 및 체크포인트 로딩 테스트

import argparse
import gzip
import struct

import numpy as np
import pytest

from scripts.predict import main, build_config
from scripts.train import main as train_main


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


def make_args(mnist_dir, task="multiclass", n=8, checkpoint=None, model="mlp"):
    return argparse.Namespace(
        task=task,
        model=model,
        seed=0,
        dataset_dir=mnist_dir,
        checkpoint=checkpoint,
        n=n,
    )


def make_train_args(mnist_dir, task="multiclass", checkpoint=None, model="mlp"):
    return argparse.Namespace(
        task=task,
        model=model,
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

    def test_batch_size_equals_n(self, mnist_dir):
        args = make_args(mnist_dir, n=5)
        config = build_config(args)
        assert config["batch_size"] == 5

    def test_num_epochs_zero(self, mnist_dir):
        args = make_args(mnist_dir)
        config = build_config(args)
        assert config["num_epochs"] == 0


# --- main() 반환값 ---

class TestPredictMain:
    @pytest.fixture(params=["multiclass", "binary", "regression"])
    def task(self, request):
        return request.param

    def test_returns_dict(self, mnist_dir, task):
        result = main(make_args(mnist_dir, task=task))
        assert isinstance(result, dict)

    def test_required_keys(self, mnist_dir, task):
        result = main(make_args(mnist_dir, task=task))
        assert set(result.keys()) == {"logits", "predictions"}

    def test_predictions_count_equals_n(self, mnist_dir, task):
        result = main(make_args(mnist_dir, task=task, n=5))
        assert len(result["predictions"]) == 5

    def test_predictions_dtype_int32(self, mnist_dir, task):
        result = main(make_args(mnist_dir, task=task))
        assert result["predictions"].dtype == np.int32

    def test_n_larger_than_dataset_clips(self, mnist_dir, task):
        # test set has 20 samples, requesting 100 should clip to 20
        result = main(make_args(mnist_dir, task=task, n=100))
        assert len(result["predictions"]) == 20


# --- --model 플래그 ---

class TestPredictModel:
    def test_model_key_in_config(self, mnist_dir):
        args = make_args(mnist_dir, model="mlp")
        config = build_config(args)
        assert config["model"] == "mlp"

    def test_cnn_model_key_in_config(self, mnist_dir):
        args = make_args(mnist_dir, model="cnn")
        config = build_config(args)
        assert config["model"] == "cnn"

    @_skip_no_gpu
    def test_cnn_returns_dict(self, mnist_dir):
        result = main(make_args(mnist_dir, model="cnn"))
        assert isinstance(result, dict)

    @_skip_no_gpu
    def test_cnn_predictions_count(self, mnist_dir):
        result = main(make_args(mnist_dir, model="cnn", n=5))
        assert len(result["predictions"]) == 5


# --- 체크포인트 로딩 ---

class TestPredictCheckpoint:
    def test_loads_checkpoint_without_error(self, mnist_dir, tmp_path):
        ckpt_path = str(tmp_path / "model")
        train_main(make_train_args(mnist_dir, checkpoint=ckpt_path))
        result = main(make_args(mnist_dir, checkpoint=ckpt_path))
        assert isinstance(result, dict)
