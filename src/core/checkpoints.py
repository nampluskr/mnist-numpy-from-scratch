# checkpoints.py: Save and load model.params as indexed .npz arrays.

import numpy as np


def save(model, path):
    """Save model.params to an .npz file as param_0, param_1, ..."""
    arrays = {f"param_{i}": p for i, p in enumerate(model.params)}
    np.savez(path, **arrays)


def load(model, path):
    """Restore model.params in-place from an .npz file."""
    npz_path = path if str(path).endswith(".npz") else str(path) + ".npz"
    data = np.load(npz_path)
    for i, param in enumerate(model.params):
        param[...] = data[f"param_{i}"]
