# mlp.py: Three-layer MLP assembled from src.nn modules.

from src.nn.layers import Linear, Sigmoid, Sequential
from src.task import get_task_spec


class MLP:
    """3-layer MLP: Sequential(Linear, Sigmoid, Linear, Sigmoid, Linear).

    forward() returns raw logits.
    src.nn.losses handles activations and gradients.
    """

    def __init__(self, task="multiclass", seed=None):
        spec = get_task_spec(task)
        self.task = task
        self.output_dim = spec["output_dim"]

        # Derive per-layer seeds for reproducibility.
        import numpy as np
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 2**31, size=3)

        self.net = Sequential(
            Linear(784, 256, seed=int(seeds[0])),
            Sigmoid(),
            Linear(256, 128, seed=int(seeds[1])),
            Sigmoid(),
            Linear(128, self.output_dim, seed=int(seeds[2])),
        )

    @property
    def params(self):
        return self.net.params

    @property
    def grads(self):
        return self.net.grads

    def forward(self, x):
        """Returns raw logits (N, output_dim)."""
        return self.net(x)

    def backward(self, grad_out):
        """grad_out: d(loss)/d(logits), computed by src.nn.losses *_grad functions."""
        return self.net.backward(grad_out)
