# losses.py: Loss functions and gradients for raw logits.
# cross_entropy and binary_cross_entropy apply activations internally.

import numpy as np
from .activations import softmax, sigmoid


def cross_entropy(logits, targets):
    """logit -> softmax -> NLL. targets: one-hot (N, C)."""
    preds = softmax(logits)
    return -np.mean(np.sum(targets * np.log(preds + 1e-8), axis=1))


def cross_entropy_grad(logits, targets):
    """d(CE+softmax)/d(logits) = (softmax(logits) - targets) / N."""
    return (softmax(logits) - targets) / len(logits)


def binary_cross_entropy(logits, targets):
    """logit -> sigmoid -> BCE. targets: (N, 1)."""
    preds = np.clip(sigmoid(logits), 1e-8, 1 - 1e-8)
    return -np.mean(targets * np.log(preds) + (1 - targets) * np.log(1 - preds))


def binary_cross_entropy_grad(logits, targets):
    """d(BCE+sigmoid)/d(logits) = (sigmoid(logits) - targets) / N."""
    return (sigmoid(logits) - targets) / len(logits)


def mse(preds, targets):
    return np.mean((preds - targets) ** 2)


def mse_grad(preds, targets):
    return 2.0 * (preds - targets) / len(preds)
