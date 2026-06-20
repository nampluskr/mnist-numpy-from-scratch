# run_all.py: Run full pipeline (train -> evaluate -> predict -> visualize) for all configs.

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.train import main as train_main
from experiments.evaluate import main as evaluate_main
from experiments.predict import main as predict_main
from experiments.visualize import main as visualize_main

DATASET_DIR = "/mnt/d/datasets/mnist"
SEED = 42
CONFIGS = [
    {"task": "multiclass", "model": "mlp", "epochs": 10, "lr": 0.01, "batch_size": 64},
    {"task": "multiclass", "model": "cnn", "epochs": 10, "lr": 0.001, "batch_size": 32},
    {"task": "binary", "model": "mlp", "epochs": 10, "lr": 0.01, "batch_size": 64},
    {"task": "binary", "model": "cnn", "epochs": 10, "lr": 0.001, "batch_size": 32},
    {"task": "regression", "model": "mlp", "epochs": 10, "lr": 0.01, "batch_size": 64},
    {"task": "regression", "model": "cnn", "epochs": 10, "lr": 0.001, "batch_size": 32},
]


def main():
    print("=== train ===")
    train_main(CONFIGS, DATASET_DIR, SEED)

    print("\n=== evaluate ===")
    evaluate_main(CONFIGS, DATASET_DIR, SEED)

    print("\n=== predict ===")
    predict_main(CONFIGS, DATASET_DIR, SEED)

    print("\n=== visualize ===")
    visualize_main(CONFIGS, DATASET_DIR, SEED)


if __name__ == "__main__":
    main()
