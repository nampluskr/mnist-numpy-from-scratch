# task.py: Task specification registry for multiclass, binary, and regression tasks.

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
