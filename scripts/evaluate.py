# evaluate.py: CLI entry point for evaluating MLP/CNN on MNIST test set.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from src.config import get_default_config
from src.core.experiment import Experiment
from src.core import checkpoints


def parse_args():
    defaults = get_default_config()
    parser = argparse.ArgumentParser(description="Evaluate MLP/CNN on MNIST test set")
    parser.add_argument("--task", default=defaults["task"],
                        choices=["multiclass", "binary", "regression"])
    parser.add_argument("--model", default="mlp", choices=["mlp", "cnn"])
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--dataset_dir", default=defaults["dataset_dir"])
    parser.add_argument("--checkpoint", default=None,
                        help="Path to load model parameters before evaluation")
    return parser.parse_args()


def build_config(args):
    return {
        "dataset_dir": args.dataset_dir,
        "task": args.task,
        "model": args.model,
        "batch_size": args.batch_size,
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
    result = exp.evaluator.evaluate(exp.test_loader)
    print(f"loss={result['loss']:.4f}  metric={result['metric']:.4f}  samples={result['num_samples']}")
    return result


if __name__ == "__main__":
    main()
