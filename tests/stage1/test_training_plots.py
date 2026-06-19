# test_training_plots.py: Unit tests for training log plot utility.

from src.utils.training_plots import plot_training_log


def make_logs(n_epochs=3):
    return [
        {
            "epoch": i,
            "train": {"loss": 1.0 - i * 0.1, "metric": 0.5 + i * 0.05, "num_samples": 40},
            "test": {"loss": 1.1 - i * 0.1, "metric": 0.4 + i * 0.05, "num_samples": 20},
        }
        for i in range(1, n_epochs + 1)
    ]


class TestPlotTrainingLog:
    def test_creates_file(self, tmp_path):
        plot_training_log(make_logs(), output_dir=str(tmp_path))
        assert (tmp_path / "training_log.png").exists()

    def test_file_is_nonempty(self, tmp_path):
        plot_training_log(make_logs(), output_dir=str(tmp_path))
        assert (tmp_path / "training_log.png").stat().st_size > 0

    def test_returns_path_string(self, tmp_path):
        result = plot_training_log(make_logs(), output_dir=str(tmp_path))
        assert isinstance(result, str)
        assert result.endswith("training_log.png")

    def test_custom_filename(self, tmp_path):
        plot_training_log(make_logs(), output_dir=str(tmp_path), filename="custom.png")
        assert (tmp_path / "custom.png").exists()

    def test_single_epoch_log(self, tmp_path):
        plot_training_log(make_logs(n_epochs=1), output_dir=str(tmp_path))
        assert (tmp_path / "training_log.png").exists()
