# metrics.py: Metric functions for classification and regression tasks.

import numpy as np


def accuracy(logits, targets):
    """targets: one-hot (N, C). Softmax is unnecessary before argmax."""
    return (logits.argmax(axis=1) == targets.argmax(axis=1)).mean()


def binary_accuracy(logits, targets):
    """sigmoid(x) >= 0.5 is equivalent to x >= 0."""
    return ((logits >= 0.0) == targets.astype(bool)).mean()


def r2_score(preds, targets):
    ss_res = np.sum((preds - targets) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-8)
