---
tags: [docs, stage6, scripts, predict, visualize]
created: "2026-06-20"
updated: "2026-06-21"
---

# 예측 및 시각화 스크립트

## 1. 개요

`scripts/predict.py`와 `scripts/visualize.py`는 학습 완료된 모델 checkpoint를 불러와 예측과 시각화를 수행하는 CLI 스크립트이다. `predict.py`는 test set 일부에 대해 `Predictor.predict`를 호출하고 결과를 콘솔에 출력한다. `visualize.py`는 예측 결과와 입력 이미지를 grid 형태의 PNG로 저장한다. 두 스크립트 모두 `--checkpoint` 인자를 필수로 받으며, `train.py`로 학습한 결과를 즉시 확인하는 워크플로를 지원한다.

**목표**
- `predict.py`로 test set 일부의 raw/decoded 예측 결과를 출력한다.
- `visualize.py`로 예측 grid PNG를 `outputs/{exp_name}/predictions.png`에 저장한다.
- 두 스크립트 모두 checkpoint 경로와 task만 지정하면 실행 가능하다.

## 2. 개념

### 2.1. predict와 visualize의 역할 분리

`predict.py`는 예측값을 텍스트로 빠르게 확인하는 데 집중하고, `visualize.py`는 학습부터 시각화까지 전체 파이프라인을 한 번에 실행하는 통합 스크립트이다.

각 스크립트의 동작과 출력은 다음과 같다.

| 스크립트 | 동작 | 출력 형태 | 용도 |
|---|---|---|---|
| `predict.py` | 모델 생성 → checkpoint 로드 → predict | 콘솔 텍스트 (`prediction` 목록) | 빠른 예측값 확인 |
| `visualize.py` | 학습 → 평가 → predict → plot | PNG 파일 (학습 곡선 + 예측 grid) | 학습 결과 종합 시각화 |

`visualize.py`는 `train.py`와 달리 내부에서 직접 학습 루프를 실행한다. checkpoint 없이 학습부터 시각화까지 한 번에 수행하려는 경우에 사용한다.

### 2.2. predict.py의 샘플 구성

`predict.py`는 `--n` 인자로 test set 앞부분에서 $n$개 샘플을 선택한다. `MNISTDataset.__getitem__(i)`로 개별 샘플을 꺼내 `np.stack`으로 배치를 구성하여 `Predictor.predict`에 전달한다.

```text
dataset[0], dataset[1], ..., dataset[n-1]
    → np.stack(images)  → (n, 784) float32
    → predictor.predict(images)
    → result["predictions"]  → (n,) int32
```

`--checkpoint`를 지정하면 `checkpoints.load`로 파라미터를 복원하고, 지정하지 않으면 초기화된 모델로 예측한다.

### 2.3. visualize.py의 파이프라인

`visualize.py`는 학습 후 두 가지 시각화를 순서대로 실행한다.

```text
학습 루프 (Trainer.fit + Evaluator.evaluate, epochs 반복)
    → plot_training_log(logs, output_dir)    → 학습 곡선 PNG
    → predictor.predict(images)
    → viz.plot_predictions(images, labels, predictions) → 예측 grid PNG
```

`plot_training_log`는 `src/utils/training_plots.py`의 함수로, epoch별 train/test loss/metric을 PNG로 저장한다. `Visualizer.plot_predictions`는 이미지와 예측/정답을 grid로 배치하여 PNG를 저장한다.

**labels 변환**: `visualize.py`는 `MNISTDataset.__getitem__`이 반환하는 task별 target(`one-hot`, `binary`, `regression` 실수)을 사람이 읽을 수 있는 정수 레이블로 변환하는 `_decode_labels` 내부 함수를 포함한다. 이 변환은 `Visualizer`에 전달하는 `labels` 인자를 위한 전처리이다.

| task | target 형태 | decode 방법 |
|---|---|---|
| `multiclass` | `(10,)` one-hot | `argmax` |
| `binary` | `(1,)` float32 | int 변환 |
| `regression` | `(1,)` float32 `[0,1]` | `round(x * 9)` |

### 2.4. 두 스크립트의 인자 차이

| 인자 | `predict.py` | `visualize.py` |
|---|---|---|
| `--task` | O | O |
| `--model` | O | O |
| `--epochs` | X | O |
| `--lr` | X | O |
| `--batch_size` | X | O |
| `--checkpoint` | O (선택) | X |
| `--n` | O (샘플 수) | X |
| `--n_samples` | X | O (시각화 샘플 수) |
| `--output_dir` | X | O |

핵심 용어는 다음과 같다.

| 용어 | 의미 | 이 프로젝트에서의 역할 |
|---|---|---|
| `result["predictions"]` | `Predictor.predict` 반환 key | decoded 예측값 `(N,)` int32 |
| `plot_predictions` | `Visualizer` 메서드 | 이미지 + 정오답 grid PNG 저장 |
| `plot_training_log` | `training_plots.py` 함수 | epoch별 train/test 곡선 PNG 저장 |
| `_decode_labels` | `visualize.py` 내부 함수 | task별 target → 정수 레이블 변환 |

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `scripts/predict.py` | CLI | `--task`, `--model`, `--checkpoint`, `--n_samples` | 없음 | 예측 결과 콘솔 출력 |
| `scripts/visualize.py` | CLI | `--task`, `--model`, `--checkpoint`, `--n_samples`, `--n_cols`, `--out_dir` | 없음 | 예측 grid PNG 저장 |

### 3.1. predict.py 구현

```python
import numpy as np
from src.data.mnist import MNISTDataset, get_task_spec
from src.models.mlp import MLP
from src.models.cnn import CNN
from src.core.predictor import Predictor
from src.utils import checkpoints

def main(args=None):
    if args is None:
        args = parse_args()
    config = build_config(args)

    task = config["task"]
    task_spec = get_task_spec(task)

    model = CNN(task=task, seed=config["seed"]) if config["model"] == "cnn" \
            else MLP(task=task, seed=config["seed"])
    predictor = Predictor(model, task_spec)

    if args.checkpoint:
        checkpoints.load(model, args.checkpoint)

    dataset = MNISTDataset("test", task, dataset_dir=config["dataset_dir"])
    n = min(args.n, len(dataset))
    images = np.stack([dataset[i][0] for i in range(n)])

    result = predictor.predict(images)
    for i, pred in enumerate(result["predictions"]):
        print(f"[{i:2d}] prediction={pred}")
    return result
```

`Predictor`는 `task_spec`을 받아 `prediction_mode`를 결정한다. `result["predictions"]`는 task별 후처리가 완료된 `(n,)` int32 배열이다.

### 3.2. visualize.py 구현

```python
import numpy as np
from src.data.mnist import MNISTDataset, get_task_spec
from src.data.dataloader import Dataloader
from src.models.mlp import MLP
from src.models.cnn import CNN
from src.core.optimizers import SGD
from src.core.trainer import Trainer
from src.core.evaluator import Evaluator
from src.core.predictor import Predictor
from src.core.visualizer import Visualizer
from src.utils.training_plots import plot_training_log

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
    predictor = Predictor(model, task_spec)

    logs = []
    for epoch in range(1, config["num_epochs"] + 1):
        train_log = trainer.fit(train_loader)
        test_log  = evaluator.evaluate(test_loader)
        logs.append({"epoch": epoch, "train": train_log, "test": test_log})

    log_path = plot_training_log(logs, output_dir=args.output_dir)
    print(f"Training log saved: {log_path}")

    viz = Visualizer(output_dir=args.output_dir)
    n = min(args.n_samples, len(test_dataset))
    images     = np.stack([test_dataset[i][0] for i in range(n)])
    raw_labels = [test_dataset[i][1] for i in range(n)]
    labels     = _decode_labels(raw_labels, task)   # task별 target → 정수 레이블

    result = predictor.predict(images)
    pred_path = viz.plot_predictions(images, labels, result["predictions"])
    print(f"Predictions saved: {pred_path}")
```

`Visualizer(output_dir=args.output_dir)`로 저장 디렉터리를 지정한다. `plot_predictions`의 세 번째 인자 `labels`는 정수 레이블이며, `_decode_labels`로 task별 target을 변환한다.

## 4. 사용법

최소 사용 예제는 다음과 같다.

```bash
conda run -n numpy_py311 python scripts/predict.py \
  --task multiclass \
  --checkpoint outputs/multiclass_mlp_ep10_lr0.01_bs128/model.npz \
  --n_samples 16

conda run -n numpy_py311 python scripts/visualize.py \
  --task multiclass \
  --checkpoint outputs/multiclass_mlp_ep10_lr0.01_bs128/model.npz \
  --n_samples 32 --n_cols 8 \
  --out_dir outputs/multiclass_mlp_ep10_lr0.01_bs128
```

예상 출력은 다음과 같다.

```text
[  0] pred=7  true=7  O
[  1] pred=2  true=2  O
[  2] pred=1  true=1  O
...
saved: outputs/multiclass_mlp_ep10_lr0.01_bs128/predictions.png
```

프로젝트 통합 예제는 다음과 같다. 학습 후 즉시 예측 grid를 저장하는 전체 흐름이다.

```bash
EXP=multiclass_mlp_ep10_lr0.01_bs128
conda run -n numpy_py311 python scripts/train.py \
  --task multiclass --epochs 10 --lr 0.01 --batch_size 128
conda run -n numpy_py311 python scripts/visualize.py \
  --task multiclass \
  --checkpoint outputs/$EXP/model.npz \
  --n_samples 32 --out_dir outputs/$EXP
```

## 5. 테스트

테스트 파일은 `tests/stage6/test_predict.py`와 `tests/stage6/test_visualize.py`이다.

```bash
conda run -n numpy_py311 pytest tests/stage6/test_predict.py tests/stage6/test_visualize.py -v
```

테스트 구성은 다음과 같다.

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestPredictScript` | 2 | n_samples 만큼 예측 출력, 3 task 동작 확인 |
| `TestVisualizeScript` | 2 | predictions.png 파일 생성, out_dir 자동 생성 |

## 6. 요약

`predict.py`는 checkpoint를 로드하고 `Predictor.predict`로 decoded 예측값을 콘솔에 출력한다. `visualize.py`는 동일한 흐름으로 예측 결과를 `Visualizer.save_grid`로 PNG에 저장한다. 두 스크립트 모두 `src/core/` 실행 객체만 조립하며, `--checkpoint`와 `--task`만 지정하면 어떤 실험 결과에도 즉시 적용할 수 있다.

다음 Phase에서는 [[phase6.3_run-all]]을 다룬다.
