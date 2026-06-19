---
tags: [stage6, scripts, evaluate]
created: 2026-06-17
updated: 2026-06-20
---

# Phase 6.2 evaluation CLI 구현

## 1. 역할

`scripts/evaluate.py`는 CLI 인자를 파싱하여 config를 구성하고, `Experiment`를 조립한 뒤 test set 전체를 평가하는 진입점이다.
`--checkpoint` 인자가 제공된 경우 모델 파라미터를 파일에서 로딩한 후 평가를 수행한다.

## 2. 구현

### 2.1. CLI 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--task` | `multiclass` | `multiclass`, `binary`, `regression` |
| `--batch_size` | `64` | 배치 크기 |
| `--seed` | `42` | 난수 시드 |
| `--dataset_dir` | `/mnt/d/datasets/mnist` | MNIST 데이터셋 경로 |
| `--checkpoint` | `None` | 파라미터 로딩 경로 (생략 시 초기 가중치로 평가) |

### 2.2. build_config(args)

`num_epochs=0`을 고정하여 학습 없이 평가만 수행하는 config를 반환한다.

### 2.3. main(args=None)

```text
config = build_config(args)
exp = Experiment(config)
if args.checkpoint:
    checkpoints.load(exp.model, args.checkpoint)
result = exp.evaluator.evaluate(exp.test_loader)
print loss / metric / samples
return result
```

### 2.4. 인터페이스

```bash
python scripts/evaluate.py --task multiclass --checkpoint outputs/model
```

```python
import argparse
from scripts.evaluate import main

args = argparse.Namespace(task="multiclass", batch_size=64,
                          seed=42, dataset_dir="...", checkpoint=None)
result = main(args)
# result: {"loss": ..., "metric": ..., "num_samples": ...}
```

## 3. 테스트

테스트 파일: `tests/stage6/test_evaluate.py`

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestBuildConfig` | 3 | 필수 키 존재, num_epochs=0, task 매핑 |
| `TestEvaluateMain` | 15 (5항목 x 3 task) | 반환 dict, 필수 키 집합, loss/metric float, num_samples 양수 |
| `TestEvaluateCheckpoint` | 1 | train 후 저장한 checkpoint 로딩 |

실행 명령:

```bash
conda run -n numpy_env pytest tests/stage6/test_evaluate.py -v
```

## 4. 설계 결정

- `Experiment`는 train/test loader를 모두 생성하지만, evaluate는 `exp.test_loader`만 사용한다.
- `num_epochs=0`으로 config를 구성하면 `exp.run()`을 호출하지 않아도 모델이 초기화된 상태로 사용 가능하다.
- checkpoint 없이 호출하면 초기 가중치 기준 평가를 수행한다. 이는 학습 전 baseline 확인 용도로 활용할 수 있다.
