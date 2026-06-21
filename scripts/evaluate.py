# evaluate.py: CLI entry point for evaluating MLP/CNN on MNIST test set.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from src.data.datasets import MulticlassDataset, BinaryDataset, RegressionDataset
from src.task import get_task_spec
from src.data.dataloader import Dataloader
from src.models.mlp import MLP
from src.models.cnn import CNN
from src.core.evaluator import Evaluator
from src.utils import checkpoints


_DEFAULTS = {
    "dataset_dir": "/mnt/d/datasets/mnist",
    "task": "multiclass",
    "model": "mlp",
    "batch_size": 64,
    "seed": 42,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MLP/CNN on MNIST test set")
    parser.add_argument("--task", default=_DEFAULTS["task"],
                        choices=["multiclass", "binary", "regression"])
    parser.add_argument("--model", default=_DEFAULTS["model"], choices=["mlp", "cnn"])
    parser.add_argument("--batch_size", type=int, default=_DEFAULTS["batch_size"])
    parser.add_argument("--seed", type=int, default=_DEFAULTS["seed"])
    parser.add_argument("--dataset_dir", default=_DEFAULTS["dataset_dir"])
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

    task = config["task"]
    task_spec = get_task_spec(task)
    dataset_dir = config["dataset_dir"]
    batch_size = config["batch_size"]

    _dataset_cls = {"multiclass": MulticlassDataset, "binary": BinaryDataset, "regression": RegressionDataset}[task]
    test_dataset = _dataset_cls("test", dataset_dir=dataset_dir)
    test_loader = Dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    if config.get("model") == "cnn":
        model = CNN(task=task, seed=config["seed"])
    else:
        model = MLP(task=task, seed=config["seed"])

    evaluator = Evaluator(model, task_spec)

    if args.checkpoint:
        checkpoints.load(model, args.checkpoint)

    result = evaluator.evaluate(test_loader)
    print(f"loss={result['loss']:.4f}  metric={result['metric']:.4f}  samples={result['num_samples']}")
    return result


if __name__ == "__main__":
    main()
