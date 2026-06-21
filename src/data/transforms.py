# transforms.py: Image and target transform functions for MNIST datasets.

import numpy as np


def normalize(images):
    return images.astype(np.float32) / 255.0


def to_flat(images):
    return images.reshape(len(images), -1)


def one_hot(labels, num_classes=10):
    n = len(labels)
    out = np.zeros((n, num_classes), dtype=np.float32)
    out[np.arange(n), labels] = 1.0
    return out


def binarize(labels):
    return (labels % 2).astype(np.float32).reshape(-1, 1)


def to_regression(labels):
    return labels.astype(np.float32).reshape(-1, 1) / 9.0
