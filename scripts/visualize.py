# visualize.py: CLI entry point for training MLP/CNN and visualizing training log and predictions.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np

from src.data.mnist import MnistDataset, get_task_spec
from src.data.dataloader import DataLoader
from src.models.mlp import MLP
from src.models.cnn import CNN
from src.core.optimizers import SGD
from src.core.trainer import Trainer
from src.core.evaluator import Evaluator
from src.core.predictor import Predictor
from src.core.visualizer import Visualizer
from src.utils.training_plots import plot_training_log


_DEFAULTS = {
    "dataset_dir": "/mnt/d/datasets/mnist",
    "task": "multiclass",
    "model": "mlp",
    "batch_size": 64,
    "num_epochs": 10,
    "seed": 42,
    "lr": 0.01,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP/CNN and visualize results")
    parser.add_argument("--task", default=_DEFAULTS["task"],
                        choices=["multiclass", "binary", "regression"])
    parser.add_argument("--model", default=_DEFAULTS["model"], choices=["mlp", "cnn"])
    parser.add_argument("--epochs", type=int, default=_DEFAULTS["num_epochs"])
    parser.add_argument("--batch_size", type=int, default=_DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=_DEFAULTS["lr"])
    parser.add_argument("--seed", type=int, default=_DEFAULTS["seed"])
    parser.add_argument("--dataset_dir", default=_DEFAULTS["dataset_dir"])
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--n_samples", type=int, default=16,
                        help="Number of test samples for prediction plot")
    return parser.parse_args()


def build_config(args):
    return {
        "dataset_dir": args.dataset_dir,
        "task": args.task,
        "model": args.model,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "seed": args.seed,
        "lr": args.lr,
    }


def _decode_labels(raw_labels, task):
    """Convert task-specific target arrays to integer digit labels for display."""
    if task == "multiclass":
        return np.argmax(np.stack(raw_labels), axis=1)
    elif task == "binary":
        return np.array(raw_labels, dtype=np.int32).ravel()
    else:  # regression
        return np.round(np.array(raw_labels, dtype=np.float32).ravel() * 9).astype(np.int32)


def main(args=None):
    if args is None:
        args = parse_args()
    config = build_config(args)

    task = config["task"]
    task_spec = get_task_spec(task)
    dataset_dir = config["dataset_dir"]
    batch_size = config["batch_size"]

    train_dataset = MnistDataset("train", task, dataset_dir=dataset_dir)
    test_dataset = MnistDataset("test", task, dataset_dir=dataset_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if config.get("model") == "cnn":
        model = CNN(task=task, seed=config["seed"])
    else:
        model = MLP(task=task, seed=config["seed"])

    optimizer = SGD(model, lr=config["lr"])
    trainer = Trainer(model, optimizer, task_spec)
    evaluator = Evaluator(model, task_spec)
    predictor = Predictor(model, task_spec)

    logs = []
    for epoch in range(1, config["num_epochs"] + 1):
        train_log = trainer.fit(train_loader)
        test_log = evaluator.evaluate(test_loader)
        logs.append({"epoch": epoch, "train": train_log, "test": test_log})

    log_path = plot_training_log(logs, output_dir=args.output_dir)
    print(f"Training log saved: {log_path}")

    viz = Visualizer(output_dir=args.output_dir)
    n = min(args.n_samples, len(test_dataset))
    images = np.stack([test_dataset[i][0] for i in range(n)])
    raw_labels = [test_dataset[i][1] for i in range(n)]
    labels = _decode_labels(raw_labels, task)

    result = predictor.predict(images)
    pred_path = viz.plot_predictions(images, labels, result["predictions"])
    print(f"Predictions saved: {pred_path}")

    return {"logs": logs, "log_path": log_path, "pred_path": pred_path}


if __name__ == "__main__":
    main()
