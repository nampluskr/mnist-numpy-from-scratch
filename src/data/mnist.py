# mnist.py: MNIST dataset loader from local gzip files.

import gzip
import os
import struct

import numpy as np

from src.config import get_default_config

_SPLIT_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


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
        dataset_dir = get_default_config()["dataset_dir"]

    img_file, lbl_file = _SPLIT_FILES[split]
    images = _load_images(os.path.join(dataset_dir, img_file))
    labels = _load_labels(os.path.join(dataset_dir, lbl_file))
    return images, labels
