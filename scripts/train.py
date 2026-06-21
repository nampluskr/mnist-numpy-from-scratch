# train.py: CLI entry point for training MLP/CNN on MNIST.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from src.data.mnist import MNISTDataset, get_task_spec
from src.data.dataloader import Dataloader
from src.models.mlp import MLP
from src.models.cnn import CNN
from src.core.optimizers import SGD
from src.core.trainer import Trainer
from src.core.evaluator import Evaluator
from src.utils import checkpoints


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
    parser = argparse.ArgumentParser(description="Train MLP/CNN on MNIST")
    parser.add_argument("--task", default=_DEFAULTS["task"],
                        choices=["multiclass", "binary", "regression"])
    parser.add_argument("--model", default=_DEFAULTS["model"], choices=["mlp", "cnn"])
    parser.add_argument("--epochs", type=int, default=_DEFAULTS["num_epochs"])
    parser.add_argument("--batch_size", type=int, default=_DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=_DEFAULTS["lr"])
    parser.add_argument("--seed", type=int, default=_DEFAULTS["seed"])
    parser.add_argument("--dataset_dir", default=_DEFAULTS["dataset_dir"])
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

    task = config["task"]
    task_spec = get_task_spec(task)
    dataset_dir = config["dataset_dir"]
    batch_size = config["batch_size"]

    train_dataset = MNISTDataset("train", task, dataset_dir=dataset_dir)
    test_dataset = MNISTDataset("test", task, dataset_dir=dataset_dir)
    train_loader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = Dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    if config.get("model") == "cnn":
        model = CNN(task=task, seed=config["seed"])
    else:
        model = MLP(task=task, seed=config["seed"])

    optimizer = SGD(model, lr=config["lr"])
    trainer = Trainer(model, optimizer, task_spec)
    evaluator = Evaluator(model, task_spec)

    logs = []
    for epoch in range(1, config["num_epochs"] + 1):
        train_log = trainer.fit(train_loader)
        test_log = evaluator.evaluate(test_loader)
        logs.append({"epoch": epoch, "train": train_log, "test": test_log})
        tl = train_log["loss"]
        tm = train_log["metric"]
        vl = test_log["loss"]
        vm = test_log["metric"]
        print(f"Epoch {epoch:3d} | train loss={tl:.4f} metric={tm:.4f} | test loss={vl:.4f} metric={vm:.4f}")

    if args.checkpoint:
        checkpoints.save(model, args.checkpoint)
        print(f"Checkpoint saved: {args.checkpoint}")
    return logs


if __name__ == "__main__":
    main()
