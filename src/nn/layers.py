# layers.py: Layer modules with forward/backward and params/grads interfaces.

import numpy as np
from .activations import sigmoid


class Module:
    def __init__(self):
        self.params = []
        self.grads = []
        self.training = True

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class Linear(Module):
    def __init__(self, in_features, out_features, seed=None):
        super().__init__()
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / in_features)  # He init
        self.w = (rng.standard_normal((in_features, out_features)) * scale).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)
        self.params = [self.w, self.b]
        self.grads = [self.grad_w, self.grad_b]
        self._x = None

    def forward(self, x):
        self._x = x
        return x @ self.w + self.b

    def backward(self, dout):
        self.grad_w[...] = self._x.T @ dout
        self.grad_b[...] = dout.sum(axis=0)
        return dout @ self.w.T


class Sigmoid(Module):
    def forward(self, x):
        self._out = sigmoid(x)
        return self._out

    def backward(self, dout):
        return dout * self._out * (1.0 - self._out)


class ReLU(Module):
    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout):
        return dout * self._mask


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        for layer in self.layers:
            self.params.extend(layer.params)
            self.grads.extend(layer.grads)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def train(self):
        self.training = True
        for layer in self.layers:
            layer.train()

    def eval(self):
        self.training = False
        for layer in self.layers:
            layer.eval()
