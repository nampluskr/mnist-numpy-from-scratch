# test_checkpoints.py: Unit tests for saving and loading parameters.

import numpy as np
import pytest
from src.core.checkpoints import save, load


class SimpleModel:
    def __init__(self, param_vals):
        self.params = [np.array(p, dtype=np.float32) for p in param_vals]


@pytest.fixture
def model():
    return SimpleModel([[1.0, 2.0, 3.0], [4.0, 5.0]])


class TestSave:
    def test_creates_npz_file(self, tmp_path, model):
        path = tmp_path / "ckpt"
        save(model, str(path))
        assert (tmp_path / "ckpt.npz").exists()

    def test_file_contains_all_params(self, tmp_path, model):
        path = tmp_path / "ckpt"
        save(model, str(path))
        data = np.load(str(path) + ".npz")
        assert "param_0" in data
        assert "param_1" in data

    def test_saved_values_match(self, tmp_path, model):
        path = tmp_path / "ckpt"
        save(model, str(path))
        data = np.load(str(path) + ".npz")
        np.testing.assert_array_equal(data["param_0"], model.params[0])
        np.testing.assert_array_equal(data["param_1"], model.params[1])


class TestLoad:
    def test_restores_param_values(self, tmp_path, model):
        path = str(tmp_path / "ckpt")
        original = [p.copy() for p in model.params]
        save(model, path)

        for p in model.params:
            p[...] = 0.0

        load(model, path)
        for param, orig in zip(model.params, original):
            np.testing.assert_array_equal(param, orig)

    def test_load_is_inplace(self, tmp_path, model):
        path = str(tmp_path / "ckpt")
        save(model, path)
        refs = [p for p in model.params]
        load(model, path)
        for ref, param in zip(refs, model.params):
            assert ref is param

    def test_load_with_npz_extension(self, tmp_path, model):
        path = str(tmp_path / "ckpt")
        save(model, path)
        for p in model.params:
            p[...] = 0.0
        load(model, path + ".npz")
        assert not np.all(model.params[0] == 0.0)

    def test_load_without_npz_extension(self, tmp_path, model):
        path = str(tmp_path / "ckpt")
        save(model, path)
        for p in model.params:
            p[...] = 0.0
        load(model, path)
        assert not np.all(model.params[0] == 0.0)
