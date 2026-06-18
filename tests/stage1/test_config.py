# test_config.py: Unit tests for get_default_config().

from src.config import get_default_config


class TestGetDefaultConfig:
    def test_returns_dict(self):
        config = get_default_config()
        assert isinstance(config, dict)

    def test_required_keys(self):
        config = get_default_config()
        required = {"dataset_dir", "seed", "batch_size", "num_epochs", "task", "split"}
        assert required == set(config.keys())

    def test_dataset_dir(self):
        config = get_default_config()
        assert config["dataset_dir"] == "/mnt/d/datasets/mnist"

    def test_seed(self):
        config = get_default_config()
        assert config["seed"] == 42

    def test_batch_size(self):
        config = get_default_config()
        assert config["batch_size"] == 64

    def test_num_epochs(self):
        config = get_default_config()
        assert config["num_epochs"] == 10

    def test_task(self):
        config = get_default_config()
        assert config["task"] == "multiclass"

    def test_split(self):
        config = get_default_config()
        assert config["split"] == "train"
