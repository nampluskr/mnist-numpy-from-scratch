# test_checkpoints.py: Unit tests for checkpoints.save() and checkpoints.load().

import os
import tempfile

import numpy as np

from src.utils import checkpoints


class _SimpleModel:
    def __init__(self):
        self.params = [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([[5.0], [6.0]], dtype=np.float32),
            np.array([0.3], dtype=np.float32),
        ]


class TestCheckpointsSave:
    def test_creates_npz_file(self):
        model = _SimpleModel()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model")
            checkpoints.save(model, path)
            assert os.path.exists(path + ".npz")

    def test_creates_file_with_npz_extension(self):
        model = _SimpleModel()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.npz")
            checkpoints.save(model, path)
            assert os.path.exists(path)


class TestCheckpointsLoad:
    def test_roundtrip_values(self):
        model = _SimpleModel()
        original = [p.copy() for p in model.params]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model")
            checkpoints.save(model, path)
            for p in model.params:
                p[...] = 0.0
            checkpoints.load(model, path)
            for orig, restored in zip(original, model.params):
                np.testing.assert_array_equal(orig, restored)

    def test_inplace_restore(self):
        model = _SimpleModel()
        original_ids = [id(p) for p in model.params]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model")
            checkpoints.save(model, path)
            checkpoints.load(model, path)
            assert [id(p) for p in model.params] == original_ids

    def test_load_with_npz_extension_in_path(self):
        model = _SimpleModel()
        original = [p.copy() for p in model.params]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model")
            checkpoints.save(model, path)
            for p in model.params:
                p[...] = 0.0
            checkpoints.load(model, path + ".npz")
            for orig, restored in zip(original, model.params):
                np.testing.assert_array_equal(orig, restored)
