---
tags: [docs, stage6, experiments, batch]
created: "2026-06-20"
updated: "2026-06-21"
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

딥러닝 실험에서는 동일한 task를 서로 다른 model, learning rate, batch size 조합으로 반복 실행하여 결과를 비교하는 작업이 자주 필요하다. `run_all.py`는 이 반복 작업을 자동화하여 `CONFIGS` 리스트만 수정하면 전체 실험 매트릭스를 한 번에 재현할 수 있다.

`experiments/` 폴더의 스크립트 구성은 다음과 같다.

| 파일 | 역할 |
|---|---|
| `run_all.py` | train → evaluate → predict → visualize 전체 파이프라인을 순차 실행 |
| `train.py` | train 단계만 선택 실행 |
| `evaluate.py` | evaluate 단계만 선택 실행 |
| `predict.py` | predict 단계만 선택 실행 |
| `visualize.py` | visualize 단계만 선택 실행 |

### 2.2. run_all.py의 실행 구조

`run_all.py`는 `experiments/train.py`, `experiments/evaluate.py` 등 단계별 모듈의 `main` 함수를 직접 임포트하여 호출한다. subprocess를 사용하지 않고 동일 프로세스 내에서 단계를 순서대로 실행한다.

```text
from experiments.train    import main as train_main
from experiments.evaluate import main as evaluate_main
from experiments.predict  import main as predict_main
from experiments.visualize import main as visualize_main

train_main(CONFIGS, DATASET_DIR, SEED)
evaluate_main(CONFIGS, DATASET_DIR, SEED)
predict_main(CONFIGS, DATASET_DIR, SEED)
visualize_main(CONFIGS, DATASET_DIR, SEED)
```

각 단계별 모듈(`train.py`, `evaluate.py` 등)은 내부에서 `scripts/train.py` 등을 **subprocess**로 호출한다. 이 이중 구조에서 `run_all.py`는 단계 순서를 관리하고, 단계별 모듈이 실제 subprocess 실행을 담당한다.

### 2.3. subprocess 기반 실행

단계별 모듈(`experiments/train.py` 등)이 `scripts/train.py`를 subprocess로 실행하는 이유는 두 가지이다.

- **상태 격리**: 각 실험 조합이 독립된 프로세스에서 시작하여 이전 실험의 전역 상태(NumPy 난수 상태, 전역 변수 등)가 다음 실험에 영향을 주지 않는다.
- **내결함성**: `check=False`로 실행하여 한 조합이 실패해도 나머지 조합이 계속 진행된다. 실패한 조합은 `[FAIL]`로 표시하고 마지막에 요약한다.

```text
for cfg in CONFIGS:
    subprocess.run([sys.executable, "scripts/train.py", ...], check=False)
    → 성공: [OK] exp_name
    → 실패: [FAIL] exp_name: 오류 메시지
→ [done] 성공 수/전체 수 success, 실패 수 failed
```

`sys.executable`을 사용하므로 현재 활성화된 conda 환경의 Python을 그대로 사용한다.

핵심 용어는 다음과 같다.

| 용어 | 의미 | 이 프로젝트에서의 역할 |
|---|---|---|
| `CONFIGS` | 실험 조합을 정의한 dict 리스트 | task, model, epochs, lr, batch_size 조합 |
| `exp_name` | 실험 조합을 인코딩한 문자열 | `outputs/{exp_name}/` 저장 경로 결정 |
| subprocess | 독립 프로세스 실행 | 실험 간 상태 격리, 내결함성 확보 |
| `check=False` | 프로세스 실패 시 예외 미발생 | 한 조합 실패 후에도 나머지 계속 실행 |

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

### 3.2. train.py 구현 (단계별 모듈)

`experiments/train.py`는 CONFIGS 조합별로 `scripts/train.py`를 subprocess로 호출한다. `evaluate.py`, `predict.py`, `visualize.py`도 같은 구조이며 호출 대상 스크립트와 인자만 다르다.

```python
import subprocess
import sys

def main(configs, dataset_dir, seed):
    for i, cfg in enumerate(configs, 1):
        name = exp_name(cfg)
        checkpoint = os.path.join("outputs", name, "model.npz")
        subprocess.run(
            [sys.executable, "scripts/train.py",
             "--task", cfg["task"], "--model", cfg["model"],
             "--epochs", str(cfg["epochs"]), "--lr", str(cfg["lr"]),
             "--batch_size", str(cfg["batch_size"]), "--seed", str(seed),
             "--checkpoint", checkpoint],
            check=True,
        )
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
