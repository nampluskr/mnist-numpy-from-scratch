# test_visualizer.py: Visualizer 파일 생성 및 반환 경로 테스트

import numpy as np
import pytest
from src.core.visualizer import Visualizer


def make_logs(n_epochs=3):
    return [
        {
            "epoch": i,
            "train": {"loss": 1.0 - i * 0.1, "metric": 0.5 + i * 0.05, "num_samples": 40},
            "test": {"loss": 1.1 - i * 0.1, "metric": 0.4 + i * 0.05, "num_samples": 20},
        }
        for i in range(1, n_epochs + 1)
    ]


def make_images_labels(n=16):
    rng = np.random.default_rng(0)
    images = rng.random((n, 784)).astype(np.float32)
    labels = rng.integers(0, 10, size=n)
    predictions = rng.integers(0, 10, size=n)
    return images, labels, predictions


class TestPlotTrainingLog:
    def test_creates_file(self, tmp_path):
        viz = Visualizer(output_dir=str(tmp_path))
        viz.plot_training_log(make_logs())
        assert (tmp_path / "training_log.png").exists()

    def test_file_is_nonempty(self, tmp_path):
        viz = Visualizer(output_dir=str(tmp_path))
        viz.plot_training_log(make_logs())
        assert (tmp_path / "training_log.png").stat().st_size > 0

    def test_returns_path_string(self, tmp_path):
        viz = Visualizer(output_dir=str(tmp_path))
        result = viz.plot_training_log(make_logs())
        assert isinstance(result, str)
        assert result.endswith("training_log.png")

    def test_custom_filename(self, tmp_path):
        viz = Visualizer(output_dir=str(tmp_path))
        viz.plot_training_log(make_logs(), filename="custom.png")
        assert (tmp_path / "custom.png").exists()

    def test_single_epoch_log(self, tmp_path):
        viz = Visualizer(output_dir=str(tmp_path))
        viz.plot_training_log(make_logs(n_epochs=1))
        assert (tmp_path / "training_log.png").exists()


class TestPlotPredictions:
    def test_creates_file(self, tmp_path):
        viz = Visualizer(output_dir=str(tmp_path))
        images, labels, predictions = make_images_labels()
        viz.plot_predictions(images, labels, predictions)
        assert (tmp_path / "predictions.png").exists()

    def test_file_is_nonempty(self, tmp_path):
        viz = Visualizer(output_dir=str(tmp_path))
        images, labels, predictions = make_images_labels()
        viz.plot_predictions(images, labels, predictions)
        assert (tmp_path / "predictions.png").stat().st_size > 0

    def test_returns_path_string(self, tmp_path):
        viz = Visualizer(output_dir=str(tmp_path))
        images, labels, predictions = make_images_labels()
        result = viz.plot_predictions(images, labels, predictions)
        assert isinstance(result, str)
        assert result.endswith("predictions.png")

    def test_custom_filename(self, tmp_path):
        viz = Visualizer(output_dir=str(tmp_path))
        images, labels, predictions = make_images_labels()
        viz.plot_predictions(images, labels, predictions, filename="pred_out.png")
        assert (tmp_path / "pred_out.png").exists()

    def test_n_larger_than_samples(self, tmp_path):
        viz = Visualizer(output_dir=str(tmp_path))
        images, labels, predictions = make_images_labels(n=4)
        viz.plot_predictions(images, labels, predictions, n=16)
        assert (tmp_path / "predictions.png").exists()


class TestVisualizerInit:
    def test_creates_output_dir(self, tmp_path):
        new_dir = tmp_path / "nested" / "outputs"
        Visualizer(output_dir=str(new_dir))
        assert new_dir.exists()
