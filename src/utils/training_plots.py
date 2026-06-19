# training_plots.py: Utility functions for plotting training logs.

import os

import matplotlib.pyplot as plt


def plot_training_log(logs, output_dir="outputs", filename="training_log.png"):
    epochs = [log["epoch"] for log in logs]
    train_losses = [log["train"]["loss"] for log in logs]
    test_losses = [log["test"]["loss"] for log in logs]
    train_metrics = [log["train"]["metric"] for log in logs]
    test_metrics = [log["test"]["metric"] for log in logs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="train")
    ax1.plot(epochs, test_losses, label="test")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(epochs, train_metrics, label="train")
    ax2.plot(epochs, test_metrics, label="test")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric")
    ax2.set_title("Metric")
    ax2.legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close(fig)
    return path
