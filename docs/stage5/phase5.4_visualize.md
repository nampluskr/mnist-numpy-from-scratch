---
tags: [stage5, scripts, visualize]
created: 2026-06-17
updated: 2026-06-19
---

# Phase 5.4 visualization CLI 구현

## 1. 역할

`scripts/visualize.py`는 CLI 인자를 파싱하여 config를 구성하고, `Experiment`를 조립하여 학습을 실행한 뒤 training log와 prediction 결과를 PNG 파일로 저장하는 진입점이다.
training log는 `src/utils/training_plots.py` helper가 저장하고, prediction 결과는 `Visualizer`가 저장한다.

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
| `--output_dir` | `outputs` | PNG 저장 경로 |
| `--n_samples` | `16` | 예측 시각화 샘플 수 |

### 2.2. _decode_labels(raw_labels, task)

`MnistDataset`이 반환하는 task별 target 배열을 시각화용 정수 레이블로 변환한다.

| task | 변환 방법 |
|---|---|
| `multiclass` | `np.argmax(one_hot)` |
| `binary` | `int32` 변환 |
| `regression` | `round(val * 9)` |

### 2.3. main(args=None)

```text
config = build_config(args)
exp = Experiment(config)
logs = exp.run()

log_path = plot_training_log(logs, output_dir=args.output_dir)

viz = Visualizer(output_dir=args.output_dir)
dataset = MnistDataset("test", task)
images, labels = 샘플 추출 + decode
result = exp.predictor.predict(images)
pred_path = viz.plot_predictions(images, labels, predictions)

return {"logs": logs, "log_path": log_path, "pred_path": pred_path}
```

### 2.4. 인터페이스

```bash
python scripts/visualize.py --task multiclass --epochs 10 --output_dir outputs
```

```python
import argparse
from scripts.visualize import main

args = argparse.Namespace(task="multiclass", epochs=2, batch_size=64,
                          lr=0.01, seed=42, dataset_dir="...",
                          output_dir="/tmp/out", n_samples=16)
result = main(args)
# result: {"logs": [...], "log_path": "...", "pred_path": "..."}
```

## 3. 테스트

테스트 파일: `tests/stage5/test_visualize.py`

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestBuildConfig` | 2 | 필수 키 존재, epochs 매핑 |
| `TestDecodeLabels` | 3 | multiclass argmax, binary int32, regression round*9 |
| `TestVisualizeMain` | 21 (7항목 x 3 task) | 반환 dict, 필수 키, PNG 파일 생성, logs 길이, 경로 str |

실행 명령:

```bash
conda run -n numpy_py311 pytest tests/stage5/test_visualize.py -v
```

## 4. 설계 결정

- `_decode_labels`를 모듈 내 헬퍼로 분리하여 테스트에서 단독으로 검증 가능하게 한다.
- training log 저장은 `src/utils/training_plots.py` helper를 호출하고, prediction 결과 저장은 `Visualizer`를 호출한다.
- `visualize.py`는 학습을 내부에서 실행하는 단독 실행형 스크립트이다. 학습된 모델을 받아 시각화만 하는 워크플로는 `evaluate.py` + `visualizer.py` 조합으로 구성한다.
- `n_samples`가 test set 크기를 초과하면 `min`으로 자동 클리핑한다.
