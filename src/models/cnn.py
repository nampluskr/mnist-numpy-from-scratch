# cnn.py: CuPy-based CNN model with the same public interface as MLP.

import numpy as np

try:
    import cupy as _xp
except ImportError:
    import numpy as _xp

from src.nn.conv import Conv2d, MaxPool2d, Flatten, Dropout
from src.nn.layers import Linear, ReLU, Sequential
from src.data.mnist import get_task_spec


class CNN:
    """CuPy-based two-convolution CNN with the same interface as MLP.

    Structure: Conv->ReLU->Pool -> Conv->ReLU->Pool -> Flatten -> Linear->ReLU->Dropout -> Linear.
    forward(): (N, 784) numpy input -> (N, output_dim) numpy raw logits.
    The CuPy/numpy boundary is handled explicitly after Flatten.
    """

    def __init__(self, task="multiclass", seed=None):
        spec = get_task_spec(task)
        self.task = task
        self.output_dim = spec["output_dim"]
        self._xp = _xp

        rng = np.random.default_rng(seed)
        conv_seeds = rng.integers(0, 2 ** 31, size=2).tolist()
        fc_seeds = rng.integers(0, 2 ** 31, size=2).tolist()

        # Conv path runs on CuPy arrays, or numpy arrays when CuPy is unavailable.
        self.conv_net = Sequential(
            Conv2d(1, 32, 3, padding=1, seed=conv_seeds[0], xp=_xp),
            ReLU(),
            MaxPool2d(2, 2, xp=_xp),
            Conv2d(32, 64, 3, padding=1, seed=conv_seeds[1], xp=_xp),
            ReLU(),
            MaxPool2d(2, 2, xp=_xp),
        )
        self.flatten = Flatten()

        # FC path runs on numpy arrays after Flatten conversion.
        self.dropout = Dropout(0.5)
        self.fc_net = Sequential(
            Linear(3136, 256, seed=fc_seeds[0]),
            ReLU(),
            Linear(256, self.output_dim, seed=fc_seeds[1]),
        )

    @property
    def params(self):
        return self.conv_net.params + self.fc_net.params

    @property
    def grads(self):
        return self.conv_net.grads + self.fc_net.grads

    def forward(self, x):
        """x: (N, 784) numpy float32 -> logits (N, output_dim) numpy float32."""
        x_xp = self._xp.asarray(x).reshape(-1, 1, 28, 28)
        x_xp = self.conv_net(x_xp)
        x_xp = self.flatten(x_xp)
        x_np = x_xp.get() if hasattr(x_xp, "get") else np.asarray(x_xp)  # CuPy -> numpy boundary
        x_np = self.dropout(x_np)
        return self.fc_net(x_np)

    def backward(self, grad_out):
        """grad_out: (N, output_dim) numpy float32 from losses.*_grad."""
        grad = self.fc_net.backward(grad_out)
        grad = self.dropout.backward(grad)
        grad_xp = self._xp.asarray(grad)  # numpy -> CuPy boundary
        grad_xp = self.flatten.backward(grad_xp)
        self.conv_net.backward(grad_xp)

    def train(self):
        self.conv_net.train()
        self.dropout.train()
        self.fc_net.train()

    def eval(self):
        self.conv_net.eval()
        self.dropout.eval()
        self.fc_net.eval()
