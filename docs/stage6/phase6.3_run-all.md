---
tags: [docs, stage6, experiments, batch]
created: "2026-06-20"
updated: "2026-06-20"
---

# 실험 배치 스크립트

## 1. 개요

`experiments/run_all.py`는 `CONFIGS` 리스트에 정의된 task, model, hyperparameter 조합을 순차적으로 실행하는 배치 스크립트이다. `scripts/train.py`를 subprocess로 호출하여 각 조합의 학습을 실행하고, 결과를 `outputs/{exp_name}/`에 저장한다. 개별 스크립트를 조합별로 반복 실행하는 수고 없이 전체 실험을 한 번에 재현할 수 있다.

**목표**
- `CONFIGS` 리스트로 실험 조합을 선언적으로 정의하고 순차 실행한다.
- 각 조합은 `subprocess`로 `scripts/train.py`를 호출하여 독립된 프로세스로 실행한다.
- 전체 실험 결과가 `outputs/` 아래에 `exp_name` 기준으로 정리된다.

## 2. 개념

### 2.1. 배치 실험의 필요성

딥러닝 실험에서는 동일한 task를 서로 다른 learning rate, batch size, model 조합으로 반복 실행하여 최적 설정을 찾는 작업이 자주 필요하다. `run_all.py`는 이 반복 작업을 자동화하여 사람이 수동으로 명령을 입력하지 않아도 전체 실험 매트릭스를 실행한다.

`experiments/` 폴더의 스크립트 구성은 다음과 같다.

| 파일 | 역할 |
|---|---|
| `run_all.py` | 모든 조합을 순차 실행하는 최상위 배치 스크립트 |
| `run_train.py` | train 조합만 선택 실행 |
| `run_evaluate.py` | evaluate 조합만 선택 실행 |
| `run_predict.py` | predict 조합만 선택 실행 |
| `run_visualize.py` | visualize 조합만 선택 실행 |

### 2.2. subprocess 기반 실행

각 실험 조합을 동일 프로세스 내 함수 호출이 아닌 `subprocess`로 실행하는 이유는 두 가지이다. 첫째, 각 실험이 독립된 메모리 공간에서 시작하여 이전 실험의 전역 상태(난수 시드, NumPy 설정 등)가 다음 실험에 영향을 주지 않는다. 둘째, 한 실험이 예외로 종료되더라도 나머지 실험은 계속 진행할 수 있다.

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `experiments/run_all.py` | 배치 스크립트 | `CONFIGS` 리스트 (코드 내 정의) | 없음 | 전체 실험 조합 순차 실행 |
| `CONFIGS` | list of dict | - | - | task, model, hyperparameter 조합 정의 |

### 3.1. run_all.py 구현

```python
import subprocess
import sys

CONFIGS = [
    {"task": "multiclass", "model": "mlp", "epochs": 10, "lr": 0.01, "batch_size": 128, "seed": 42},
    {"task": "binary",     "model": "mlp", "epochs": 10, "lr": 0.01, "batch_size": 128, "seed": 42},
    {"task": "regression", "model": "mlp", "epochs": 10, "lr": 0.01, "batch_size": 128, "seed": 42},
]

def run(config):
    cmd = [
        sys.executable, "scripts/train.py",
        "--task",       config["task"],
        "--model",      config["model"],
        "--epochs",     str(config["epochs"]),
        "--lr",         str(config["lr"]),
        "--batch_size", str(config["batch_size"]),
        "--seed",       str(config["seed"]),
    ]
    print(f"[run] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[FAILED] task={config['task']} model={config['model']}")

if __name__ == "__main__":
    for config in CONFIGS:
        run(config)
```

`check=False`로 설정하여 한 조합이 실패하더라도 나머지 조합이 계속 실행된다. `sys.executable`을 사용하여 현재 Python 인터프리터를 그대로 사용하므로 conda 환경 내에서 `conda run`을 별도로 지정하지 않아도 된다.

### 3.2. run_train.py 구현

`run_all.py`에서 train 조합만 선택 실행하는 변형 스크립트이다. `run_evaluate.py`, `run_predict.py`, `run_visualize.py`도 같은 구조이며 호출 스크립트와 인자만 다르다.

```python
import subprocess
import sys

CONFIGS = [
    {"task": "multiclass", "model": "mlp", "epochs": 10, "lr": 0.01, "batch_size": 128, "seed": 42},
    {"task": "binary",     "model": "mlp", "epochs": 10, "lr": 0.01, "batch_size": 128, "seed": 42},
    {"task": "regression", "model": "mlp", "epochs": 10, "lr": 0.01, "batch_size": 128, "seed": 42},
]

if __name__ == "__main__":
    for cfg in CONFIGS:
        subprocess.run([
            sys.executable, "scripts/train.py",
            "--task", cfg["task"], "--model", cfg["model"],
            "--epochs", str(cfg["epochs"]), "--lr", str(cfg["lr"]),
            "--batch_size", str(cfg["batch_size"]), "--seed", str(cfg["seed"]),
        ], check=False)
```

## 4. 사용법

최소 사용 예제는 다음과 같다.

```bash
conda run -n numpy_py311 python experiments/run_all.py
```

예상 출력은 다음과 같다.

```text
[run] python scripts/train.py --task multiclass --model mlp --epochs 10 ...
[run] python scripts/train.py --task binary --model mlp --epochs 10 ...
[run] python scripts/train.py --task regression --model mlp --epochs 10 ...
```

프로젝트 통합 예제는 다음과 같다. `run_all.py` 실행 후 각 task 결과 폴더를 확인하는 흐름이다.

```bash
conda run -n numpy_py311 python experiments/run_all.py
ls outputs/
# multiclass_mlp_ep10_lr0.01_bs128/
# binary_mlp_ep10_lr0.01_bs128/
# regression_mlp_ep10_lr0.01_bs128/
```

## 5. 테스트

테스트 파일은 `tests/stage6/`에 별도로 두지 않는다. `run_all.py`는 `scripts/train.py`를 호출하는 orchestration 레이어이므로 `test_train.py`에서 개별 스크립트 동작을 검증하는 것으로 충분하다. 필요한 경우 `CONFIGS`에 소규모 조합(epochs=1, n_samples 최소)을 추가하여 smoke test를 작성한다.

## 6. 요약

`run_all.py`는 `CONFIGS` 리스트에 선언된 실험 조합을 순차적으로 `subprocess`로 실행하여 전체 실험 매트릭스를 자동화한다. `check=False` 설정으로 한 조합의 실패가 나머지 실험을 중단하지 않으며, 결과는 `outputs/{exp_name}/`에 조합별로 정리된다.

다음 Phase에서는 [[phase6.4_experiments-and-results]]을 다룬다.
