# mlp.py: NumPy 기반 3층 MLP (784->256->128->output_dim), manual backward + SGD update.

import numpy as np
from src.task import get_task_spec


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))


def _sigmoid_grad(s):
    return s * (1.0 - s)


def _softmax(x):
    shifted = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(shifted, dtype=np.float32)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


class MLP:
    """3-layer MLP: 784 -> 256 -> 128 -> output_dim."""

    def __init__(self, task="multiclass", seed=None):
        spec = get_task_spec(task)
        self.task = task
        self.output_dim = spec["output_dim"]
        self.prediction_mode = spec["prediction_mode"]

        rng = np.random.default_rng(seed)

        def he(fan_in, fan_out):
            return rng.standard_normal((fan_in, fan_out)).astype(np.float32) * np.sqrt(2.0 / fan_in)

        self.params = {
            "W1": he(784, 256), "b1": np.zeros(256, dtype=np.float32),
            "W2": he(256, 128), "b2": np.zeros(128, dtype=np.float32),
            "W3": he(128, self.output_dim), "b3": np.zeros(self.output_dim, dtype=np.float32),
        }
        self._cache = {}
        self._grads = {}

    def forward(self, x):
        p = self.params
        z1 = x @ p["W1"] + p["b1"]
        a1 = _sigmoid(z1)
        z2 = a1 @ p["W2"] + p["b2"]
        a2 = _sigmoid(z2)
        z3 = a2 @ p["W3"] + p["b3"]

        if self.task == "multiclass":
            out = _softmax(z3)
        elif self.task == "binary":
            out = _sigmoid(z3)
        else:  # regression
            out = z3.astype(np.float32)

        self._cache = {"x": x, "a1": a1, "a2": a2, "out": out}
        return out

    def backward(self, grad_out):
        p = self.params
        c = self._cache
        g = self._grads

        # layer 3
        if self.task in ("multiclass", "binary"):
            dz3 = grad_out  # softmax/sigmoid gradient already folded into grad_out
        else:
            dz3 = grad_out

        g["W3"] = c["a2"].T @ dz3
        g["b3"] = dz3.sum(axis=0)

        # layer 2
        da2 = dz3 @ p["W3"].T
        dz2 = da2 * _sigmoid_grad(c["a2"])
        g["W2"] = c["a1"].T @ dz2
        g["b2"] = dz2.sum(axis=0)

        # layer 1
        da1 = dz2 @ p["W2"].T
        dz1 = da1 * _sigmoid_grad(c["a1"])
        g["W1"] = c["x"].T @ dz1
        g["b1"] = dz1.sum(axis=0)

        grad_x = dz1 @ p["W1"].T
        return grad_x

    def update(self, lr=0.01):
        for key in self.params:
            self.params[key] -= lr * self._grads[key]
