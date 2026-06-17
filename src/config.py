# config.py: Default configuration values for dataset path, training, and task settings.


def get_default_config():
    return {
        "dataset_dir": "/mnt/d/datasets/mnist",
        "seed": 42,
        "batch_size": 64,
        "num_epochs": 10,
        "task": "multiclass",
        "split": "train",
    }
