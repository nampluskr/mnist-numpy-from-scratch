# mnist.py: MNIST dataset loader and dataset class from local gzip files.

import gzip
import os
import struct

import numpy as np

_DATASET_DIR = "/mnt/d/datasets/mnist"

_SPLIT_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}

_VALID_TASKS = ("multiclass", "binary", "regression")

_TASK_SPECS = {
    "multiclass": {
        "task": "multiclass",
        "output_dim": 10,
        "target_dtype": "float32",
        "prediction_mode": "argmax",
    },
    "binary": {
        "task": "binary",
        "output_dim": 1,
        "target_dtype": "float32",
        "prediction_mode": "threshold",
    },
    "regression": {
        "task": "regression",
        "output_dim": 1,
        "target_dtype": "float32",
        "prediction_mode": "round_clip",
    },
}


def get_task_spec(task):
    if task not in _VALID_TASKS:
        raise ValueError(f"Unknown task: {task!r}. Must be one of {_VALID_TASKS}.")
    return dict(_TASK_SPECS[task])


def transform_targets(labels, task):
    if task not in _VALID_TASKS:
        raise ValueError(f"Unknown task: {task!r}. Must be one of {_VALID_TASKS}.")

    if task == "multiclass":
        n = len(labels)
        targets = np.zeros((n, 10), dtype=np.float32)
        targets[np.arange(n), labels] = 1.0
        return targets
    elif task == "binary":
        return (labels % 2).astype(np.float32).reshape(-1, 1)
    else:  # regression
        return labels.astype(np.float32).reshape(-1, 1) / 9.0


def _load_images(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    with gzip.open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def _load_labels(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label file not found: {path}")
    with gzip.open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_mnist(split, dataset_dir=None):
    if split not in _SPLIT_FILES:
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")
    if dataset_dir is None:
        dataset_dir = _DATASET_DIR

    img_file, lbl_file = _SPLIT_FILES[split]
    images = _load_images(os.path.join(dataset_dir, img_file))
    labels = _load_labels(os.path.join(dataset_dir, lbl_file))
    return images, labels


class MNISTDataset:
    def __init__(self, split, task, dataset_dir=None):
        images, labels = load_mnist(split, dataset_dir=dataset_dir)
        self.images = images.reshape(-1, 784).astype(np.float32) / 255.0
        self.targets = transform_targets(labels, task)
        self.task_spec = get_task_spec(task)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]
