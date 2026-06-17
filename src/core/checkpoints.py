# checkpoints.py: Save and load model.params as indexed .npz arrays.

import numpy as np


def _to_numpy(array):
    """Return a NumPy copy for NumPy or CuPy-like arrays."""
    return array.get() if hasattr(array, "get") else np.asarray(array)


def _to_param_array(param, array):
    """Convert a saved NumPy array to the same array module as param."""
    module = type(param).__module__.split(".")[0]
    if module == "cupy":
        import cupy as cp
        return cp.asarray(array)
    return np.asarray(array)


def save(model, path):
    """Save model.params to an .npz file as param_0, param_1, ..."""
    arrays = {f"param_{i}": _to_numpy(p) for i, p in enumerate(model.params)}
    np.savez(path, **arrays)


def load(model, path):
    """Restore model.params in-place from an .npz file."""
    npz_path = path if str(path).endswith(".npz") else str(path) + ".npz"
    data = np.load(npz_path)
    for i, param in enumerate(model.params):
        param[...] = _to_param_array(param, data[f"param_{i}"])
