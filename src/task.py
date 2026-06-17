# task.py: Task specification and target transformation for multiclass, binary, and regression.

import numpy as np


_VALID_TASKS = ("multiclass", "binary", "regression")

_TASK_SPECS = {
    "multiclass": {
        "task": "multiclass",
        "output_dim": 10,
        "target_dtype": "float32",
        "prediction_mode": "argmax",
    },
    "binary": {
        "task": "binary",
        "output_dim": 1,
        "target_dtype": "float32",
        "prediction_mode": "threshold",
    },
    "regression": {
        "task": "regression",
        "output_dim": 1,
        "target_dtype": "float32",
        "prediction_mode": "round_clip",
    },
}


def get_task_spec(task):
    if task not in _VALID_TASKS:
        raise ValueError(f"Unknown task: {task!r}. Must be one of {_VALID_TASKS}.")
    return dict(_TASK_SPECS[task])


def transform_targets(labels, task):
    if task not in _VALID_TASKS:
        raise ValueError(f"Unknown task: {task!r}. Must be one of {_VALID_TASKS}.")

    if task == "multiclass":
        n = len(labels)
        targets = np.zeros((n, 10), dtype=np.float32)
        targets[np.arange(n), labels] = 1.0
        return targets
    elif task == "binary":
        return (labels % 2).astype(np.float32).reshape(-1, 1)
    else:  # regression
        return labels.astype(np.float32).reshape(-1, 1) / 9.0
