# train.py: CLI entry point for training MLP/CNN - parses args, assembles Experiment, runs training loop.

import argparse

from src.config import get_default_config
from src.core.experiment import Experiment
from src.core import checkpoints


def parse_args():
    defaults = get_default_config()
    parser = argparse.ArgumentParser(description="Train MLP/CNN on MNIST")
    parser.add_argument("--task", default=defaults["task"],
                        choices=["multiclass", "binary", "regression"])
    parser.add_argument("--model", default="mlp", choices=["mlp", "cnn"])
    parser.add_argument("--epochs", type=int, default=defaults["num_epochs"])
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--dataset_dir", default=defaults["dataset_dir"])
    parser.add_argument("--checkpoint", default=None,
                        help="Path to save model parameters after training")
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


def main(args=None):
    if args is None:
        args = parse_args()
    config = build_config(args)
    exp = Experiment(config)
    logs = exp.run()
    for log in logs:
        e = log["epoch"]
        tl = log["train"]["loss"]
        tm = log["train"]["metric"]
        vl = log["test"]["loss"]
        vm = log["test"]["metric"]
        print(f"Epoch {e:3d} | train loss={tl:.4f} metric={tm:.4f} | test loss={vl:.4f} metric={vm:.4f}")
    if args.checkpoint:
        checkpoints.save(exp.model, args.checkpoint)
        print(f"Checkpoint saved: {args.checkpoint}")
    return logs


if __name__ == "__main__":
    main()
