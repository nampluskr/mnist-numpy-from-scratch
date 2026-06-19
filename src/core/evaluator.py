# evaluator.py: Evaluation loop for one pass over a DataLoader.

from src.nn.losses import (
    cross_entropy,
    binary_cross_entropy,
    mse,
)
from src.nn.metrics import accuracy, binary_accuracy, r2_score

_TASK_FNS = {
    "multiclass": (cross_entropy, accuracy),
    "binary": (binary_cross_entropy, binary_accuracy),
    "regression": (mse, r2_score),
}


class Evaluator:
    def __init__(self, model, task_spec):
        self.model = model
        task = task_spec["task"]
        self.loss_fn, self.metric_fn = _TASK_FNS[task]

    def evaluate(self, test_loader):
        """Run one pass. Returns {"loss", "metric", "num_samples"}."""
        total_loss = 0.0
        total_metric = 0.0
        total_samples = 0

        for x, y in test_loader:
            n = len(x)
            logits = self.model.forward(x)
            loss = self.loss_fn(logits, y)
            metric = self.metric_fn(logits, y)
            total_loss += float(loss) * n
            total_metric += float(metric) * n
            total_samples += n

        return {
            "loss": total_loss / total_samples,
            "metric": total_metric / total_samples,
            "num_samples": total_samples,
        }
