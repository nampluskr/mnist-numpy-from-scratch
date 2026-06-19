---
tags: [stage7, tutorial, regression, mlp]
created: 2026-06-18
updated: 2026-06-20
---

# Phase 7.4 Regression tutorial

## 1. 과제 개요

Regression 과제는 MNIST digit label을 연속값으로 보고 예측한다.
학습 target은 digit을 9로 나눈 값이며, 예측 결과는 다시 0부터 9까지의 digit으로 복원한다.

## 2. 데이터 규약

`MnistDataset`은 digit label을 float32 회귀 target으로 변환한다.
모델 출력은 하나의 연속 logit이며, predictor는 scale 복원, clip, round를 순서대로 적용한다.

| 항목 | 값 |
|---|---|
| 입력 shape | `(784,)` |
| target shape | `(1,)` |
| target 변환 | `label / 9.0` |
| output_dim | `1` |
| prediction_mode | `round_clip` |

## 3. 모델 구성

Regression MLP는 hidden 구조를 공유하고 마지막 출력 차원만 1로 둔다.

```text
Linear(784, 256) -> Sigmoid -> Linear(256, 128) -> Sigmoid -> Linear(128, 1)
```

## 4. 학습

다음 명령은 regression MLP를 학습하고 checkpoint를 저장한다.

```bash
PYTHONPATH=. conda run -n numpy_env python scripts/train.py --task regression --model mlp --checkpoint outputs/regression/mlp/model.npz
```

Phase 7.2에서 생성한 산출물은 다음 위치에 둔다.

| 파일 | 경로 |
|---|---|
| checkpoint | `outputs/regression/mlp/model.npz` |
| training log plot | `outputs/regression/mlp/training_log.png` |
| prediction plot | `outputs/regression/mlp/predictions.png` |

## 5. 평가

저장된 checkpoint를 사용해 regression MLP를 평가한다.

```bash
PYTHONPATH=. conda run -n numpy_env python scripts/evaluate.py --task regression --model mlp --checkpoint outputs/regression/mlp/model.npz
```

실행 결과는 다음과 같다.

| loss | metric | samples |
|---|---|---|
| `0.0408` | `0.5984` | `10000` |

Regression loss는 MSE이고 metric은 R2 score이다.
위 결과는 test split에서 R2 score 0.5984를 의미한다.

## 6. 예측

`predict.py`는 회귀 출력을 digit label로 복원한 prediction을 출력한다.

```bash
PYTHONPATH=. conda run -n numpy_env python scripts/predict.py --task regression --model mlp --checkpoint outputs/regression/mlp/model.npz --n 16
```

예측 후처리는 `round(clip(logit * 9, 0, 9))` 규약을 사용한다.

## 7. 시각화

`visualize.py`는 regression 학습 로그와 예측 결과 이미지를 저장한다.

```bash
PYTHONPATH=. MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib conda run -n numpy_env python scripts/visualize.py --task regression --model mlp --output_dir outputs/regression/mlp
```

`training_log.png`의 metric 곡선은 R2 score 변화를 나타낸다.
`predictions.png`는 회귀 출력이 후처리된 digit prediction을 표시한다.

## 8. 설계 결정

- 회귀 target은 `0..9` digit을 `0..1` 범위로 정규화하여 MSE 학습을 안정화한다.
- predictor에서 scale 복원과 clip을 수행하여 최종 예측값을 digit 범위로 제한한다.
- 같은 MLP 구조를 사용해 task 차이가 target, loss, metric, prediction mode에만 집중되도록 한다.
