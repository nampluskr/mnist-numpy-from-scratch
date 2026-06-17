# mlp.py: src.nn 모듈을 조립한 3층 MLP (784→256→128→output_dim), PyTorch 방식

from src.nn.layers import Linear, Sigmoid, Sequential
from src.task import get_task_spec


class MLP:
    """3-layer MLP: Sequential(Linear, Sigmoid, Linear, Sigmoid, Linear).

    forward()는 raw logit을 반환한다.
    activation과 gradient는 src.nn.losses 함수가 처리한다.
    """

    def __init__(self, task="multiclass", seed=None):
        spec = get_task_spec(task)
        self.task = task
        self.output_dim = spec["output_dim"]

        # 재현성을 위해 seed에서 레이어별 seed를 파생
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
        """grad_out: d(loss)/d(logits), src.nn.losses의 *_grad 함수로 계산."""
        return self.net.backward(grad_out)
