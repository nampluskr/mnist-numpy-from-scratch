---
tags: [stage6, scripts, predict]
created: 2026-06-17
updated: 2026-06-20
---

# Phase 6.3 prediction CLI 구현

## 1. 역할

`scripts/predict.py`는 CLI 인자를 파싱하여 config를 구성하고, `Experiment`를 조립한 뒤 MNIST test 샘플에 대한 예측을 수행하는 진입점이다.
`--checkpoint` 인자가 제공된 경우 모델 파라미터를 로딩한 후 예측을 수행한다.

## 2. 구현

### 2.1. CLI 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--task` | `multiclass` | `multiclass`, `binary`, `regression` |
| `--seed` | `42` | 난수 시드 |
| `--dataset_dir` | `/mnt/d/datasets/mnist` | MNIST 데이터셋 경로 |
| `--checkpoint` | `None` | 파라미터 로딩 경로 |
| `--n` | `16` | 예측할 test 샘플 수 |

### 2.2. build_config(args)

`batch_size=args.n`으로 설정하여 예측 배치 크기를 샘플 수와 일치시킨다.
`num_epochs=0`으로 고정하여 학습을 수행하지 않는다.

### 2.3. main(args=None)

```text
config = build_config(args)
exp = Experiment(config)
if args.checkpoint:
    checkpoints.load(exp.model, args.checkpoint)
dataset = MnistDataset("test", args.task, dataset_dir=args.dataset_dir)
n = min(args.n, len(dataset))
images = np.stack([dataset[i][0] for i in range(n)])
result = exp.predictor.predict(images)
for i, pred in enumerate(result["predictions"]):
    print [i] prediction=pred
return result
```

`args.n`이 test set 크기를 초과하면 가용한 샘플 수만큼 자동으로 클리핑한다.

### 2.4. 인터페이스

```bash
python scripts/predict.py --task multiclass --checkpoint outputs/model --n 32
```

```python
import argparse
from scripts.predict import main

args = argparse.Namespace(task="multiclass", seed=42,
                          dataset_dir="...", checkpoint=None, n=16)
result = main(args)
# result: {"logits": ..., "predictions": ...}
```

## 3. 테스트

테스트 파일: `tests/stage6/test_predict.py`

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestBuildConfig` | 3 | 필수 키 존재, batch_size=n 매핑, num_epochs=0 |
| `TestPredictMain` | 15 (5항목 x 3 task) | 반환 dict, 필수 키, predictions 개수, dtype int32, n 초과 시 클리핑 |
| `TestPredictCheckpoint` | 1 | train 후 저장한 checkpoint 로딩 |

실행 명령:

```bash
conda run -n numpy_env pytest tests/stage6/test_predict.py -v
```

## 4. 설계 결정

- `Experiment`의 `predictor`를 재사용하고, 입력 images는 `MnistDataset`에서 직접 추출한다.
- `args.n`을 `batch_size`에 매핑하여 DataLoader 없이 단일 배치로 예측하는 방식을 채택한다.
- test set 샘플 수보다 많은 `--n`을 요청하면 가용 샘플 수로 자동 클리핑하여 오류 없이 동작한다.
