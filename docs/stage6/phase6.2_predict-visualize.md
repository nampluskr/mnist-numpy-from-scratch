---
tags: [docs, stage6, scripts, predict, visualize]
created: "2026-06-20"
updated: "2026-06-20"
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

`predict.py`는 예측값 자체를 텍스트로 확인하는 데 집중하고, `visualize.py`는 이미지와 예측값을 함께 시각적으로 확인하는 데 집중한다. 두 스크립트를 분리한 이유는 실험 중에는 빠른 수치 확인이 필요하고, 리포팅에는 이미지 grid가 유용하기 때문이다.

각 스크립트의 출력은 다음과 같다.

| 스크립트 | 출력 형태 | 용도 |
|---|---|---|
| `predict.py` | 콘솔 텍스트 (decoded, true label) | 빠른 수치 확인 |
| `visualize.py` | PNG 파일 (이미지 + 예측/정답 grid) | 오분류 육안 확인 및 리포팅 |

### 2.2. n_samples 인자

`predict.py`와 `visualize.py` 모두 `--n_samples` 인자로 test set에서 확인할 샘플 수를 지정한다. 기본값은 16이며, 전체 test set을 대상으로 할 때는 `--n_samples 10000`을 사용한다. `visualize.py`에서는 `--n_cols`로 grid 열 수를 조정할 수 있다.

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `scripts/predict.py` | CLI | `--task`, `--model`, `--checkpoint`, `--n_samples` | 없음 | 예측 결과 콘솔 출력 |
| `scripts/visualize.py` | CLI | `--task`, `--model`, `--checkpoint`, `--n_samples`, `--n_cols`, `--out_dir` | 없음 | 예측 grid PNG 저장 |

### 3.1. predict.py 구현

```python
import argparse
from src.data.mnist import MnistDataset
from src.models.mlp import MLP
from src.core.predictor import Predictor
from src.utils.checkpoints import load_checkpoint

def main(args):
    test_ds = MnistDataset(split="test", task=args.task)
    images = test_ds.images[:args.n_samples]
    labels_raw = test_ds.labels_raw[:args.n_samples]

    model = MLP(task=args.task)
    load_checkpoint(model, args.checkpoint)

    predictor = Predictor(model, task=args.task)
    result = predictor.predict(images)

    for i, (pred, true) in enumerate(zip(result["decoded"], labels_raw)):
        status = "O" if pred == int(true) else "X"
        print(f"[{i:3d}] pred={pred}  true={int(true)}  {status}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="multiclass")
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_samples", type=int, default=16)
    main(parser.parse_args())
```

`labels_raw`는 task 변환 이전의 원본 레이블(0~9 정수)이다. `MnistDataset`에서 `labels_raw` 속성으로 제공하며, `targets`(task별 변환 배열)와 구분하여 사용한다.

### 3.2. visualize.py 구현

```python
import argparse
import os
from src.data.mnist import MnistDataset
from src.models.mlp import MLP
from src.core.predictor import Predictor
from src.core.visualizer import Visualizer
from src.utils.checkpoints import load_checkpoint

def main(args):
    test_ds = MnistDataset(split="test", task=args.task)
    images = test_ds.images[:args.n_samples]
    labels_raw = test_ds.labels_raw[:args.n_samples]

    model = MLP(task=args.task)
    load_checkpoint(model, args.checkpoint)

    predictor = Predictor(model, task=args.task)
    result = predictor.predict(images)

    save_path = os.path.join(args.out_dir, "predictions.png")
    visualizer = Visualizer(task=args.task)
    visualizer.save_grid(images, result["decoded"], labels_raw, save_path, n_cols=args.n_cols)
    print(f"saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="multiclass")
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--n_cols", type=int, default=8)
    parser.add_argument("--out_dir", default="outputs/predictions")
    main(parser.parse_args())
```

`--out_dir`을 별도 인자로 분리하여 `train.py`가 생성한 `outputs/{exp_name}/` 폴더와 연동할 수 있다.

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
