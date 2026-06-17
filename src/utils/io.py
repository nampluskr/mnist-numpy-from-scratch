# io.py: File I/O utilities for saving and loading numpy parameter arrays.

import numpy as np


def save_params(params, path):
    np.savez(path, **params)


def load_params(path):
    data = np.load(path)
    return dict(data)
