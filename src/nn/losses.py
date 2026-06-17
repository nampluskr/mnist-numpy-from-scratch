# losses.py: 손실 함수 및 평가 지표 (torch.nn / torch.nn.functional 대응)
# cross_entropy / binary_cross_entropy 는 logit 을 입력으로 받아 activation 을 내부 처리

import numpy as np
from .activations import softmax, sigmoid


def cross_entropy(logits, targets):
    """logit → softmax → NLL. targets: one-hot (N, C)."""
    preds = softmax(logits)
    return -np.mean(np.sum(targets * np.log(preds + 1e-8), axis=1))


def cross_entropy_grad(logits, targets):
    """d(CE+softmax)/d(logits) = (softmax(logits) - targets) / N."""
    return (softmax(logits) - targets) / len(logits)


def binary_cross_entropy(logits, targets):
    """logit → sigmoid → BCE. targets: (N, 1)."""
    preds = np.clip(sigmoid(logits), 1e-8, 1 - 1e-8)
    return -np.mean(targets * np.log(preds) + (1 - targets) * np.log(1 - preds))


def binary_cross_entropy_grad(logits, targets):
    """d(BCE+sigmoid)/d(logits) = (sigmoid(logits) - targets) / N."""
    return (sigmoid(logits) - targets) / len(logits)


def mse(preds, targets):
    return np.mean((preds - targets) ** 2)


def mse_grad(preds, targets):
    return 2.0 * (preds - targets) / len(preds)


def accuracy(logits, targets):
    """targets: one-hot (N, C). softmax 불필요 — argmax 는 단조 변환에 불변."""
    return (logits.argmax(axis=1) == targets.argmax(axis=1)).mean()


def binary_accuracy(logits, targets):
    """sigmoid(x) >= 0.5 ↔ x >= 0."""
    return ((logits >= 0.0) == targets.astype(bool)).mean()


def r2_score(preds, targets):
    ss_res = np.sum((preds - targets) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-8)
