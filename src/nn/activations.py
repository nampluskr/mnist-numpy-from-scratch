# activations.py: 함수형 활성화 함수 (순전파 전용, torch.nn.functional 대응)

import numpy as np


def sigmoid(x):
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    out[~pos] = np.exp(x[~pos]) / (1.0 + np.exp(x[~pos]))
    return out


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def relu(x):
    return np.maximum(0.0, x)


def identity(x):
    return x
