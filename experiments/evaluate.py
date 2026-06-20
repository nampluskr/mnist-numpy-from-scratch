# evaluate.py: Batch evaluation for all configs via subprocess.

import os
import subprocess
import sys


def exp_name(cfg):
    return f"{cfg['task']}_{cfg['model']}_ep{cfg['epochs']}_lr{cfg['lr']}_bs{cfg['batch_size']}"


def main(configs, dataset_dir, seed):
    total = len(configs)
    results = []

    for i, cfg in enumerate(configs, 1):
        name = exp_name(cfg)
        checkpoint = os.path.join("outputs", name, "model.npz")
        print(f"\n[{i}/{total}] evaluate | {name}")
        try:
            subprocess.run(
                [sys.executable, "scripts/evaluate.py",
                 "--task", cfg["task"],
                 "--model", cfg["model"],
                 "--batch_size", str(cfg["batch_size"]),
                 "--seed", str(seed),
                 "--dataset_dir", dataset_dir,
                 "--checkpoint", checkpoint],
                check=True,
            )
            results.append({"name": name, "success": True, "error": None})
            print(f"[OK] {name}")
        except subprocess.CalledProcessError as e:
            results.append({"name": name, "success": False, "error": str(e)})
            print(f"[FAIL] {name}: {e}")

    success = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    print(f"\n[done] {len(success)}/{total} success, {len(failed)} failed")
    for r in failed:
        print(f"  [FAIL] {r['name']}: {r['error']}")


if __name__ == "__main__":
    _DATASET_DIR = "/mnt/d/datasets/mnist"
    _SEED = 42
    _CONFIGS = [
        {"task": "multiclass", "model": "mlp", "epochs": 10, "lr": 0.01, "batch_size": 64},
        {"task": "multiclass", "model": "cnn", "epochs": 10, "lr": 0.001, "batch_size": 32},
        {"task": "binary", "model": "mlp", "epochs": 10, "lr": 0.01, "batch_size": 64},
        {"task": "binary", "model": "cnn", "epochs": 10, "lr": 0.001, "batch_size": 32},
        {"task": "regression", "model": "mlp", "epochs": 10, "lr": 0.01, "batch_size": 64},
        {"task": "regression", "model": "cnn", "epochs": 10, "lr": 0.001, "batch_size": 32},
    ]
    main(_CONFIGS, _DATASET_DIR, _SEED)
