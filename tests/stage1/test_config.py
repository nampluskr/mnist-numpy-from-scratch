# test_config.py: Unit tests for get_default_config().

from src.config import get_default_config


def test_returns_dict():
    config = get_default_config()
    assert isinstance(config, dict)


def test_required_keys():
    config = get_default_config()
    required = {"dataset_dir", "seed", "batch_size", "num_epochs", "task", "split"}
    assert required == set(config.keys())


def test_dataset_dir():
    config = get_default_config()
    assert config["dataset_dir"] == "/mnt/d/datasets/mnist"


def test_seed():
    config = get_default_config()
    assert config["seed"] == 42


def test_batch_size():
    config = get_default_config()
    assert config["batch_size"] == 64


def test_num_epochs():
    config = get_default_config()
    assert config["num_epochs"] == 10


def test_task():
    config = get_default_config()
    assert config["task"] == "multiclass"


def test_split():
    config = get_default_config()
    assert config["split"] == "train"
