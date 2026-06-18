# trainer.py: Training loop — one epoch over a DataLoader, returns loss/metric summary.

from src.nn.losses import (
    cross_entropy, cross_entropy_grad,
    binary_cross_entropy, binary_cross_entropy_grad,
    mse, mse_grad,
)
from src.nn.metrics import accuracy, binary_accuracy, r2_score

_TASK_FNS = {
    "multiclass": (cross_entropy, cross_entropy_grad, accuracy),
    "binary": (binary_cross_entropy, binary_cross_entropy_grad, binary_accuracy),
    "regression": (mse, mse_grad, r2_score),
}


class Trainer:
    def __init__(self, model, optimizer, task_spec):
        self.model = model
        self.optimizer = optimizer
        task = task_spec["task"]
        self.loss_fn, self.grad_fn, self.metric_fn = _TASK_FNS[task]

    def fit(self, train_loader):
        """Run one epoch. Returns {"loss", "metric", "num_samples"}."""
        total_loss = 0.0
        total_metric = 0.0
        total_samples = 0

        for x, y in train_loader:
            n = len(x)
            logits = self.model.forward(x)
            loss = self.loss_fn(logits, y)
            metric = self.metric_fn(logits, y)
            grad = self.grad_fn(logits, y)
            self.model.backward(grad)
            self.optimizer.step()
            total_loss += float(loss) * n
            total_metric += float(metric) * n
            total_samples += n

        return {
            "loss": total_loss / total_samples,
            "metric": total_metric / total_samples,
            "num_samples": total_samples,
        }
