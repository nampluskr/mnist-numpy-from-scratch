# visualizer.py: Training log and prediction visualization; saves figures to output_dir.

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_log(self, logs, filename="training_log.png"):
        """Plot train/test loss and metric curves. logs: list of per-epoch dicts from Experiment.run()."""
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
        path = self.output_dir / filename
        plt.savefig(path)
        plt.close()
        return str(path)

    def plot_predictions(self, images, labels, predictions, filename="predictions.png", n=16):
        """Plot n sample images with true labels and predicted labels.

        images:      (N, 784) float32
        labels:      (N,)     int — decoded true labels
        predictions: (N,)     int — decoded predictions from Predictor
        """
        n = min(n, len(images))
        cols = min(8, n)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))
        axes = np.array(axes).ravel()

        for i in range(n):
            img = images[i].reshape(28, 28)
            axes[i].imshow(img, cmap="gray")
            color = "green" if labels[i] == predictions[i] else "red"
            axes[i].set_title(f"T:{int(labels[i])} P:{int(predictions[i])}", fontsize=8, color=color)
            axes[i].axis("off")

        for i in range(n, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path)
        plt.close()
        return str(path)
