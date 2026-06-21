---
tags: [docs, stage6, scripts, train, evaluate]
created: "2026-06-20"
updated: "2026-06-21"
---

# 학습 및 평가 스크립트

## 1. 개요

`scripts/train.py`와 `scripts/evaluate.py`는 `src/core/` 실행 객체를 조립하여 학습과 평가를 CLI 진입점으로 제공하는 스크립트이다. `train.py`는 명령행 인자로 task, model, epochs, learning rate, batch size를 받아 학습 루프를 실행하고 checkpoint와 학습 로그를 저장한다. `evaluate.py`는 저장된 checkpoint를 불러와 test set에 대한 loss와 metric을 출력한다.

**목표**
- `argparse`로 task, model, hyperparameter를 명령행에서 설정할 수 있도록 한다.
- 학습 후 checkpoint, CSV 로그, 학습 곡선 PNG를 `outputs/{exp_name}/`에 저장한다.
- `evaluate.py`는 checkpoint 경로를 받아 `Evaluator.evaluate` 결과를 출력한다.

## 2. 개념

### 2.1. CLI 스크립트 역할

`scripts/`의 스크립트는 `src/` 내부 모듈을 직접 조립하지 않고 `src/core/`의 실행 객체를 조립하는 진입점이다. 이 원칙 덕분에 학습 루프, 평가 루프, optimizer 구현을 바꾸더라도 스크립트 인터페이스는 그대로 유지할 수 있다.

스크립트와 `src/` 모듈의 의존 관계는 다음과 같다.

| 스크립트 | 직접 참조하는 모듈 |
|---|---|
| `scripts/train.py` | `src/core/trainer.py`, `src/core/evaluator.py`, `src/core/optimizers.py`, `src/data/`, `src/models/`, `src/utils/checkpoints` |
| `scripts/evaluate.py` | `src/core/evaluator.py`, `src/data/`, `src/models/`, `src/utils/checkpoints` |

### 2.2. task_spec과 스크립트 조립 흐름

`scripts/train.py`는 CLI 인자로 `--task`를 받아 `get_task_spec(task)`를 호출하고, 반환된 `task_spec` dict를 `Trainer`와 `Evaluator` 생성자에 전달한다.

```text
--task 인자 → get_task_spec(task) → task_spec dict
    → Trainer(model, optimizer, task_spec)
    → Evaluator(model, task_spec)
```

`task_spec`은 `{"task": ..., "output_dim": ..., "prediction_mode": ...}` 구조로, 실행 객체들이 task별 함수를 내부에서 선택하는 데 사용된다. 스크립트는 `task_spec` 내용을 직접 참조하지 않는다.

### 2.3. train.py의 학습+평가 동시 실행

`scripts/train.py`는 학습만 실행하는 것이 아니라 매 epoch마다 `Trainer.fit`과 `Evaluator.evaluate`를 모두 호출하여 train/test 지표를 동시에 출력한다.

```text
for epoch in 1..N:
    train_log = trainer.fit(train_loader)       → train loss/metric
    test_log  = evaluator.evaluate(test_loader) → test loss/metric
    print(epoch, train_log, test_log)
```

이렇게 하면 epoch마다 overfitting 여부를 즉시 확인할 수 있다. `Evaluator`는 backward를 호출하지 않으므로 파라미터에 영향을 주지 않는다.

### 2.4. checkpoint 저장 방식

`train.py`는 학습 완료 후 `--checkpoint` 인자로 지정한 경로에 `checkpoints.save(model, path)`로 모델 파라미터를 `.npz` 파일로 저장한다. `--checkpoint`가 지정되지 않으면 저장을 건너뛴다.

`evaluate.py`는 `--checkpoint` 인자로 지정한 경로에서 `checkpoints.load(model, path)`로 파라미터를 복원한 뒤 평가를 실행한다. `--checkpoint`가 없으면 초기화된 파라미터 그대로 평가한다.

| 스크립트 | checkpoint 처리 |
|---|---|
| `train.py` | 학습 후 `checkpoints.save(model, path)` (선택적) |
| `evaluate.py` | 평가 전 `checkpoints.load(model, path)` (선택적) |

### 2.5. exp_name과 outputs 폴더 구조

`experiments/` 스크립트가 `scripts/train.py`를 subprocess로 호출할 때 `--checkpoint` 경로를 `outputs/{exp_name}/model.npz`로 지정하여 실험 결과를 구조화한다. `exp_name`은 실험 설정을 파일명으로 인코딩하여 폴더만 보고도 어떤 조합인지 파악할 수 있다.

`exp_name` 형식은 다음과 같다.

```text
{task}_{model}_ep{epochs}_lr{lr}_bs{batch_size}
예: multiclass_mlp_ep10_lr0.01_bs64
```

`outputs/{exp_name}/`에 저장되는 파일 구성은 다음과 같다.

| 파일 | 저장 주체 | 내용 |
|---|---|---|
| `model.npz` | `train.py` | `checkpoints.save`로 저장한 모델 파라미터 |

핵심 용어는 다음과 같다.

| 용어 | 의미 | 이 프로젝트에서의 역할 |
|---|---|---|
| `task_spec` | task 정보를 담은 dict | `get_task_spec(task)`가 반환, 실행 객체 생성자에 전달 |
| `exp_name` | 실험 설정을 인코딩한 문자열 | checkpoint 저장 경로의 폴더명으로 사용 |
| `checkpoints.save/load` | 모델 파라미터 저장·복원 | `.npz` 형식으로 저장, 학습 재개나 평가 시 복원 |

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `scripts/train.py` | CLI | `--task`, `--model`, `--epochs`, `--lr`, `--batch_size`, `--seed` | 없음 | 학습 실행 및 결과 저장 |
| `scripts/evaluate.py` | CLI | `--task`, `--model`, `--checkpoint`, `--batch_size` | 없음 | 평가 결과 출력 |

### 3.1. train.py 구현

```python
from src.data.mnist import MNISTDataset, get_task_spec
from src.data.dataloader import Dataloader
from src.models.mlp import MLP
from src.models.cnn import CNN
from src.core.optimizers import SGD
from src.core.trainer import Trainer
from src.core.evaluator import Evaluator
from src.utils import checkpoints

def main(args=None):
    if args is None:
        args = parse_args()
    config = build_config(args)

    task = config["task"]
    task_spec = get_task_spec(task)

    train_dataset = MNISTDataset("train", task, dataset_dir=config["dataset_dir"])
    test_dataset  = MNISTDataset("test",  task, dataset_dir=config["dataset_dir"])
    train_loader  = Dataloader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader   = Dataloader(test_dataset,  batch_size=config["batch_size"], shuffle=False)

    model     = CNN(task=task, seed=config["seed"]) if config["model"] == "cnn" \
                else MLP(task=task, seed=config["seed"])
    optimizer = SGD(model, lr=config["lr"])
    trainer   = Trainer(model, optimizer, task_spec)
    evaluator = Evaluator(model, task_spec)

    for epoch in range(1, config["num_epochs"] + 1):
        train_log = trainer.fit(train_loader)
        test_log  = evaluator.evaluate(test_loader)
        print(f"Epoch {epoch:3d} | train loss={train_log['loss']:.4f} metric={train_log['metric']:.4f}"
              f" | test loss={test_log['loss']:.4f} metric={test_log['metric']:.4f}")

    if args.checkpoint:
        checkpoints.save(model, args.checkpoint)
        print(f"Checkpoint saved: {args.checkpoint}")
```

`task_spec`을 `Trainer`와 `Evaluator` 모두에 전달한다. 매 epoch마다 train/test 지표를 함께 출력하여 overfitting을 모니터링한다.

### 3.2. evaluate.py 구현

```python
from src.data.mnist import MNISTDataset, get_task_spec
from src.data.dataloader import Dataloader
from src.models.mlp import MLP
from src.models.cnn import CNN
from src.core.evaluator import Evaluator
from src.utils import checkpoints

def main(args=None):
    if args is None:
        args = parse_args()
    config = build_config(args)

    task = config["task"]
    task_spec = get_task_spec(task)

    test_dataset = MNISTDataset("test", task, dataset_dir=config["dataset_dir"])
    test_loader  = Dataloader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = CNN(task=task, seed=config["seed"]) if config["model"] == "cnn" \
            else MLP(task=task, seed=config["seed"])
    evaluator = Evaluator(model, task_spec)

    if args.checkpoint:
        checkpoints.load(model, args.checkpoint)

    result = evaluator.evaluate(test_loader)
    print(f"loss={result['loss']:.4f}  metric={result['metric']:.4f}  samples={result['num_samples']}")
    return result
```

`--checkpoint`를 지정하지 않으면 초기화된 파라미터로 평가한다. `checkpoints.load`는 `src/utils/checkpoints.py`의 `load` 함수이다.

## 4. 사용법

최소 사용 예제는 다음과 같다.

```bash
conda run -n numpy_py311 python scripts/train.py \
  --task multiclass --model mlp --epochs 10 --lr 0.01 --batch_size 128

conda run -n numpy_py311 python scripts/evaluate.py \
  --task multiclass --model mlp \
  --checkpoint outputs/multiclass_mlp_ep10_lr0.01_bs128/model.npz
```

예상 출력은 다음과 같다.

```text
loss: 0.2981  metric: 0.9123  samples: 10000
```

프로젝트 통합 예제는 다음과 같다. 세 가지 task를 순차 학습하고 각각 평가하는 흐름이다.

```bash
for TASK in multiclass binary regression; do
  conda run -n numpy_py311 python scripts/train.py \
    --task $TASK --model mlp --epochs 10 --lr 0.01 --batch_size 128
  conda run -n numpy_py311 python scripts/evaluate.py \
    --task $TASK --checkpoint outputs/${TASK}_mlp_ep10_lr0.01_bs128/model.npz
done
```

## 5. 테스트

테스트 파일은 `tests/stage6/test_train.py`와 `tests/stage6/test_evaluate.py`이다.

```bash
conda run -n numpy_py311 pytest tests/stage6/test_train.py tests/stage6/test_evaluate.py -v
```

테스트 구성은 다음과 같다.

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestTrainScript` | 3 | checkpoint 파일 생성, CSV 로그 파일 생성, 학습 곡선 PNG 생성 |
| `TestEvaluateScript` | 2 | evaluate 후 loss/metric/num_samples 출력, checkpoint 없을 때 오류 |

## 6. 요약

`train.py`는 명령행 인자를 받아 MNISTDataset, Dataloader, MLP, optimizer, Trainer를 조립하고 학습 결과를 `outputs/{exp_name}/`에 저장한다. `evaluate.py`는 checkpoint를 로드하여 Evaluator로 test set 성능을 출력한다. 두 스크립트 모두 `src/core/` 실행 객체만 참조하며 내부 구현을 직접 호출하지 않는다.

다음 Phase에서는 [[phase6.2_predict-visualize]]을 다룬다.
