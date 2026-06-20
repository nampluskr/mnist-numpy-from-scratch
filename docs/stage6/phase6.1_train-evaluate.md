---
tags: [docs, stage6, scripts, train, evaluate]
created: "2026-06-20"
updated: "2026-06-20"
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

`scripts/`의 스크립트는 `src/` 내부 모듈을 직접 호출하지 않고 `src/core/`의 실행 객체만 조립한다. 이 원칙 덕분에 학습 루프, 평가 루프, optimizer 구현을 바꾸더라도 스크립트 인터페이스는 그대로 유지할 수 있다.

스크립트와 `src/` 모듈의 의존 관계는 다음과 같다.

| 스크립트 | 직접 참조하는 모듈 |
|---|---|
| `scripts/train.py` | `src/core/trainer.py`, `src/core/optimizers.py`, `src/data/`, `src/models/`, `src/utils/` |
| `scripts/evaluate.py` | `src/core/evaluator.py`, `src/utils/checkpoints.py`, `src/data/`, `src/models/` |

### 2.2. exp_name과 outputs 폴더 구조

학습 결과는 `outputs/{exp_name}/` 폴더에 저장한다. `exp_name`은 실험 설정을 파일명으로 인코딩하여 결과 폴더만 보고도 어떤 조합인지 파악할 수 있다.

`exp_name` 형식은 다음과 같다.

```text
{task}_{model}_ep{epochs}_lr{lr}_bs{batch_size}
예: multiclass_mlp_ep10_lr0.01_bs128
```

저장 파일 구성은 다음과 같다.

| 파일 | 내용 |
|---|---|
| `model.npz` | `save_checkpoint`로 저장한 모델 파라미터 |
| `train_log.csv` | epoch별 loss/metric 로그 |
| `training_curves.png` | 학습 곡선 그래프 |

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `scripts/train.py` | CLI | `--task`, `--model`, `--epochs`, `--lr`, `--batch_size`, `--seed` | 없음 | 학습 실행 및 결과 저장 |
| `scripts/evaluate.py` | CLI | `--task`, `--model`, `--checkpoint`, `--batch_size` | 없음 | 평가 결과 출력 |

### 3.1. train.py 구현

```python
import argparse
from src.data.mnist import MnistDataset
from src.data.dataloader import DataLoader
from src.models.mlp import MLP
from src.core.optimizers import SGD
from src.core.trainer import Trainer
from src.core.logger import Logger
from src.utils.checkpoints import save_checkpoint
from src.utils.training_plots import save_training_plots

def main(args):
    exp_name = f"{args.task}_{args.model}_ep{args.epochs}_lr{args.lr}_bs{args.batch_size}"
    out_dir = f"outputs/{exp_name}"

    train_ds = MnistDataset(split="train", task=args.task)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = MLP(task=args.task, seed=args.seed)
    optimizer = SGD(model, lr=args.lr)
    trainer = Trainer(model, optimizer, task=args.task)

    logs = trainer.fit(train_loader, epochs=args.epochs)

    logger = Logger()
    logger.load(logs)
    logger.to_csv(f"{out_dir}/train_log.csv")
    save_checkpoint(model, f"{out_dir}/model.npz")
    save_training_plots(logger.to_dict(), save_path=f"{out_dir}/training_curves.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="multiclass")
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
```

`exp_name`을 `out_dir` 경로와 CSV/PNG 파일명에 일관되게 사용한다. `save_checkpoint`와 `save_training_plots`는 `out_dir`이 없으면 자동으로 생성한다.

### 3.2. evaluate.py 구현

```python
import argparse
from src.data.mnist import MnistDataset
from src.data.dataloader import DataLoader
from src.models.mlp import MLP
from src.core.evaluator import Evaluator
from src.utils.checkpoints import load_checkpoint

def main(args):
    test_ds = MnistDataset(split="test", task=args.task)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = MLP(task=args.task)
    load_checkpoint(model, args.checkpoint)

    evaluator = Evaluator(model, task=args.task)
    result = evaluator.evaluate(test_loader)

    print(f"loss: {result['loss']:.4f}  metric: {result['metric']:.4f}  samples: {result['num_samples']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="multiclass")
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    main(parser.parse_args())
```

`--checkpoint`는 필수 인자로 지정하여 checkpoint 경로 누락 시 즉시 오류 메시지를 출력한다.

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

`train.py`는 명령행 인자를 받아 MnistDataset, DataLoader, MLP, optimizer, Trainer를 조립하고 학습 결과를 `outputs/{exp_name}/`에 저장한다. `evaluate.py`는 checkpoint를 로드하여 Evaluator로 test set 성능을 출력한다. 두 스크립트 모두 `src/core/` 실행 객체만 참조하며 내부 구현을 직접 호출하지 않는다.

다음 Phase에서는 [[phase6.2_predict-visualize]]을 다룬다.
