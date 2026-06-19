# test_visualize.py: Unit tests for scripts/visualize.py main() and file creation.

import argparse
import gzip
import struct

import numpy as np
import pytest

from scripts.visualize import main, build_config, _decode_labels


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


def make_args(mnist_dir, task="multiclass", epochs=1, output_dir=None, n_samples=8, model="mlp"):
    return argparse.Namespace(
        task=task,
        model=model,
        epochs=epochs,
        batch_size=8,
        lr=0.01,
        seed=0,
        dataset_dir=mnist_dir,
        output_dir=output_dir or mnist_dir,
        n_samples=n_samples,
    )


# --- build_config ---

class TestBuildConfig:
    def test_keys_present(self, mnist_dir):
        args = make_args(mnist_dir)
        config = build_config(args)
        assert {"dataset_dir", "task", "batch_size", "num_epochs", "seed", "lr"} <= set(config.keys())

    def test_epochs_mapped(self, mnist_dir):
        args = make_args(mnist_dir, epochs=3)
        config = build_config(args)
        assert config["num_epochs"] == 3


# --- _decode_labels ---

class TestDecodeLabels:
    def test_multiclass_returns_argmax(self):
        one_hot = [np.eye(10)[i] for i in [3, 7, 0]]
        labels = _decode_labels(one_hot, "multiclass")
        np.testing.assert_array_equal(labels, [3, 7, 0])

    def test_binary_returns_int32(self):
        raw = [np.array([1.0]), np.array([0.0])]
        labels = _decode_labels(raw, "binary")
        assert labels.dtype == np.int32
        np.testing.assert_array_equal(labels, [1, 0])

    def test_regression_maps_to_digit(self):
        raw = [np.array([0.0]), np.array([1.0])]
        labels = _decode_labels(raw, "regression")
        assert labels[0] == 0
        assert labels[1] == 9


# --- --model flag ---

class TestVisualizeModel:
    def test_model_key_in_config(self, mnist_dir):
        args = make_args(mnist_dir, model="mlp")
        config = build_config(args)
        assert config["model"] == "mlp"

    def test_cnn_model_key_in_config(self, mnist_dir):
        args = make_args(mnist_dir, model="cnn")
        config = build_config(args)
        assert config["model"] == "cnn"

    @_skip_no_gpu
    def test_cnn_returns_dict(self, mnist_dir, tmp_path):
        result = main(make_args(mnist_dir, model="cnn", output_dir=str(tmp_path)))
        assert isinstance(result, dict)

    @_skip_no_gpu
    def test_cnn_files_created(self, mnist_dir, tmp_path):
        main(make_args(mnist_dir, model="cnn", output_dir=str(tmp_path)))
        assert (tmp_path / "training_log.png").exists()
        assert (tmp_path / "predictions.png").exists()


# --- main() return value ---

class TestVisualizeMain:
    @pytest.fixture(params=["multiclass", "binary", "regression"])
    def task(self, request):
        return request.param

    def test_returns_dict(self, mnist_dir, task, tmp_path):
        result = main(make_args(mnist_dir, task=task, output_dir=str(tmp_path)))
        assert isinstance(result, dict)

    def test_required_keys(self, mnist_dir, task, tmp_path):
        result = main(make_args(mnist_dir, task=task, output_dir=str(tmp_path)))
        assert {"logs", "log_path", "pred_path"} <= set(result.keys())

    def test_training_log_file_created(self, mnist_dir, task, tmp_path):
        main(make_args(mnist_dir, task=task, output_dir=str(tmp_path)))
        assert (tmp_path / "training_log.png").exists()

    def test_predictions_file_created(self, mnist_dir, task, tmp_path):
        main(make_args(mnist_dir, task=task, output_dir=str(tmp_path)))
        assert (tmp_path / "predictions.png").exists()

    def test_logs_length_equals_epochs(self, mnist_dir, task, tmp_path):
        result = main(make_args(mnist_dir, task=task, epochs=2, output_dir=str(tmp_path)))
        assert len(result["logs"]) == 2

    def test_log_path_is_string(self, mnist_dir, task, tmp_path):
        result = main(make_args(mnist_dir, task=task, output_dir=str(tmp_path)))
        assert isinstance(result["log_path"], str)

    def test_pred_path_is_string(self, mnist_dir, task, tmp_path):
        result = main(make_args(mnist_dir, task=task, output_dir=str(tmp_path)))
        assert isinstance(result["pred_path"], str)
