---
tags: [stage5, scripts, train]
created: 2026-06-17
updated: 2026-06-17
---

# Phase 5.1 training CLI 구현

## 1. 역할

`scripts/train.py`는 CLI 인자를 파싱하여 config를 구성하고, `Experiment`를 조립한 뒤 학습을 실행하는 최상위 진입점이다.
per-epoch loss/metric을 표준 출력으로 출력하고, `--checkpoint` 인자가 제공된 경우 학습 후 모델 파라미터를 저장한다.

## 2. 구현

### 2.1. CLI 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--task` | `multiclass` | `multiclass`, `binary`, `regression` |
| `--epochs` | `10` | 학습 epoch 수 |
| `--batch_size` | `64` | 배치 크기 |
| `--lr` | `0.01` | 학습률 |
| `--seed` | `42` | 난수 시드 |
| `--dataset_dir` | `/mnt/d/datasets/mnist` | MNIST 데이터셋 경로 |
| `--checkpoint` | `None` | 파라미터 저장 경로 (생략 시 저장 안 함) |

### 2.2. build_config(args)

`argparse.Namespace`를 `Experiment`가 수신하는 config dict로 변환한다.
`args.epochs`를 `config["num_epochs"]`로 매핑하는 것에 주의한다.

### 2.3. main(args=None)

```text
config = build_config(args)
exp = Experiment(config)
logs = exp.run()
for log in logs:
    print Epoch / train loss / train metric / test loss / test metric
if args.checkpoint:
    checkpoints.save(exp.model, args.checkpoint)
return logs
```

`args=None`이면 `parse_args()`로 CLI 인자를 파싱한다. 테스트에서는 `argparse.Namespace`를 직접 전달한다.

### 2.4. 인터페이스

```bash
python scripts/train.py --task multiclass --epochs 10 --checkpoint outputs/model
```

```python
# 테스트/스크립트 내 직접 호출
import argparse
from scripts.train import main

args = argparse.Namespace(task="multiclass", epochs=2, batch_size=64,
                          lr=0.01, seed=42, dataset_dir="...", checkpoint=None)
logs = main(args)
```

## 3. 테스트

테스트 파일: `tests/stage5/test_train.py`

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestBuildConfig` | 3 | 필수 키 존재, epochs/task 매핑 |
| `TestTrainMain` | 18 (6항목 x 3 task) | 반환 list, 길이, epoch 키, train/test 키, log 키 집합 |
| `TestTrainCheckpoint` | 2 | checkpoint 파일 생성, None 시 미생성 |

실행 명령:

```bash
conda run -n numpy_env pytest tests/stage5/test_train.py -v
```

## 4. 설계 결정

- `main(args=None)` 패턴으로 설계하여 CLI 실행과 함수 직접 호출 모두 지원한다.
- 학습 루프는 `Experiment.run()`에 위임하고 스크립트는 출력과 체크포인트 저장만 담당한다.
- `--checkpoint` 미지정 시 파라미터를 저장하지 않아 side effect를 최소화한다.
