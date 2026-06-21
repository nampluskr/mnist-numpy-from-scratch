# predict.py: CLI entry point for running predictions with MLP/CNN on MNIST test samples.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np

from src.data.datasets import MulticlassDataset, BinaryDataset, RegressionDataset
from src.task import get_task_spec
from src.models.mlp import MLP
from src.models.cnn import CNN
from src.core.predictor import Predictor
from src.utils import checkpoints


_DEFAULTS = {
    "dataset_dir": "/mnt/d/datasets/mnist",
    "task": "multiclass",
    "model": "mlp",
    "seed": 42,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with MLP/CNN on MNIST test samples")
    parser.add_argument("--task", default=_DEFAULTS["task"],
                        choices=["multiclass", "binary", "regression"])
    parser.add_argument("--model", default=_DEFAULTS["model"], choices=["mlp", "cnn"])
    parser.add_argument("--seed", type=int, default=_DEFAULTS["seed"])
    parser.add_argument("--dataset_dir", default=_DEFAULTS["dataset_dir"])
    parser.add_argument("--checkpoint", default=None,
                        help="Path to load model parameters before prediction")
    parser.add_argument("--n", type=int, default=16,
                        help="Number of test samples to predict")
    return parser.parse_args()


def build_config(args):
    return {
        "dataset_dir": args.dataset_dir,
        "task": args.task,
        "model": args.model,
        "batch_size": args.n,
        "num_epochs": 0,
        "seed": args.seed,
    }


def main(args=None):
    if args is None:
        args = parse_args()
    config = build_config(args)

    task = config["task"]
    task_spec = get_task_spec(task)
    dataset_dir = config["dataset_dir"]

    if config.get("model") == "cnn":
        model = CNN(task=task, seed=config["seed"])
    else:
        model = MLP(task=task, seed=config["seed"])

    predictor = Predictor(model, task_spec)

    if args.checkpoint:
        checkpoints.load(model, args.checkpoint)

    _dataset_cls = {"multiclass": MulticlassDataset, "binary": BinaryDataset, "regression": RegressionDataset}[task]
    dataset = _dataset_cls("test", dataset_dir=dataset_dir)
    n = min(args.n, len(dataset))
    images = np.stack([dataset[i][0] for i in range(n)])
    result = predictor.predict(images)
    for i, pred in enumerate(result["predictions"]):
        print(f"[{i:2d}] prediction={pred}")
    return result


if __name__ == "__main__":
    main()
