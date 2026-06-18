# metrics.py: 평가 지표 함수 (정확도, R² 등)

import numpy as np


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
