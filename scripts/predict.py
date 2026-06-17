# predict.py: CLI entry point for running predictions on MNIST test samples.

import argparse

import numpy as np

from src.config import get_default_config
from src.core.experiment import Experiment
from src.core import checkpoints
from src.data.mnist import MnistDataset


def parse_args():
    defaults = get_default_config()
    parser = argparse.ArgumentParser(description="Predict with MLP on MNIST test samples")
    parser.add_argument("--task", default=defaults["task"],
                        choices=["multiclass", "binary", "regression"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--dataset_dir", default=defaults["dataset_dir"])
    parser.add_argument("--checkpoint", default=None,
                        help="Path to load model parameters before prediction")
    parser.add_argument("--n", type=int, default=16,
                        help="Number of test samples to predict")
    return parser.parse_args()


def build_config(args):
    return {
        "dataset_dir": args.dataset_dir,
        "task": args.task,
        "batch_size": args.n,
        "num_epochs": 0,
        "seed": args.seed,
    }


def main(args=None):
    if args is None:
        args = parse_args()
    config = build_config(args)
    exp = Experiment(config)
    if args.checkpoint:
        checkpoints.load(exp.model, args.checkpoint)
    dataset = MnistDataset("test", args.task, dataset_dir=args.dataset_dir)
    n = min(args.n, len(dataset))
    images = np.stack([dataset[i][0] for i in range(n)])
    result = exp.predictor.predict(images)
    for i, pred in enumerate(result["predictions"]):
        print(f"[{i:2d}] prediction={pred}")
    return result


if __name__ == "__main__":
    main()
