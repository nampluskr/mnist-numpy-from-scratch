# visualizer.py: Prediction visualization class for saving sample image grids.

import os
import math

import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_predictions(self, images, labels, predictions, filename="predictions.png", n=16):
        """Plot n sample images with true labels and predicted labels.

        images:      (N, 784) float32
        labels:      (N,)     int - decoded true labels
        predictions: (N,)     int - decoded predictions from Predictor
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
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path)
        plt.close(fig)
        return path
