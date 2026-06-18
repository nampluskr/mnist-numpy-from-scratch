# test_io.py: Unit tests for save_params() and load_params().

import os
import tempfile

import numpy as np

from src.utils.io import save_params, load_params


PARAMS = {
    "w1": np.random.randn(4, 3).astype(np.float32),
    "b1": np.zeros(3, dtype=np.float32),
}


class TestSaveParams:
    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "params.npz")
            save_params(PARAMS, path)
            assert os.path.exists(path)


class TestLoadParams:
    def test_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "params.npz")
            save_params(PARAMS, path)
            loaded = load_params(path)
            assert isinstance(loaded, dict)

    def test_roundtrip_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "params.npz")
            save_params(PARAMS, path)
            loaded = load_params(path)
            assert set(loaded.keys()) == set(PARAMS.keys())

    def test_roundtrip_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "params.npz")
            save_params(PARAMS, path)
            loaded = load_params(path)
            for key in PARAMS:
                np.testing.assert_array_equal(loaded[key], PARAMS[key])
