# test_visualizer.py: Unit tests for prediction visualization.

import numpy as np
from src.core.visualizer import Visualizer


def make_images_labels(n=16):
    rng = np.random.default_rng(0)
    images = rng.random((n, 784)).astype(np.float32)
    labels = rng.integers(0, 10, size=n)
    predictions = rng.integers(0, 10, size=n)
    return images, labels, predictions


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
