# visualize.py: CLI entry point for training MLP/CNN and visualizing training log and predictions.

import argparse

import numpy as np

from src.config import get_default_config
from src.core.experiment import Experiment
from src.core.visualizer import Visualizer
from src.data.mnist import MnistDataset
from src.utils.training_plots import plot_training_log


def parse_args():
    defaults = get_default_config()
    parser = argparse.ArgumentParser(description="Train MLP/CNN and visualize results")
    parser.add_argument("--task", default=defaults["task"],
                        choices=["multiclass", "binary", "regression"])
    parser.add_argument("--model", default="mlp", choices=["mlp", "cnn"])
    parser.add_argument("--epochs", type=int, default=defaults["num_epochs"])
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--dataset_dir", default=defaults["dataset_dir"])
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
    exp = Experiment(config)
    logs = exp.run()

    log_path = plot_training_log(logs, output_dir=args.output_dir)
    print(f"Training log saved: {log_path}")

    viz = Visualizer(output_dir=args.output_dir)
    dataset = MnistDataset("test", args.task, dataset_dir=args.dataset_dir)
    n = min(args.n_samples, len(dataset))
    images = np.stack([dataset[i][0] for i in range(n)])
    raw_labels = [dataset[i][1] for i in range(n)]
    labels = _decode_labels(raw_labels, args.task)

    result = exp.predictor.predict(images)
    pred_path = viz.plot_predictions(images, labels, result["predictions"])
    print(f"Predictions saved: {pred_path}")

    return {"logs": logs, "log_path": log_path, "pred_path": pred_path}


if __name__ == "__main__":
    main()
