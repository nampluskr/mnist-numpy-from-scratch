---
tags: [stage7, tutorial, regression, cnn]
created: 2026-06-18
updated: 2026-06-20
---

# Phase 7.4 Regression tutorial

## 1. 과제 개요

Regression 과제는 MNIST digit label을 연속값으로 보고 예측한다.
CNN 구현은 CuPy convolution 계층을 사용하며, target은 digit을 9로 나눈 값으로 정규화한다.

## 2. 데이터 규약

`MnistDataset`은 digit label을 float32 regression target으로 변환한다.
모델 출력은 하나의 연속 logit이며, predictor는 scale 복원, clip, round를 순서대로 적용한다.

| 항목 | 값 |
|---|---|
| 입력 shape | `(784,)` |
| CNN 내부 shape | `(N, 1, 28, 28)` |
| target shape | `(1,)` |
| target 변환 | `label / 9.0` |
| output_dim | `1` |
| prediction_mode | `round_clip` |

## 3. 모델 구성

Regression CNN은 classification CNN과 같은 convolution 구조를 사용한다.
마지막 `Linear`만 output dimension 1로 구성한다.

```text
Conv2d(1, 32, 3, padding=1) -> ReLU -> MaxPool2d(2, 2)
Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool2d(2, 2)
Flatten -> Linear(3136, 256) -> ReLU -> Linear(256, 1)
```

## 4. 학습

다음 명령은 regression CNN을 1 epoch 학습하고 checkpoint를 저장한다.

```bash
mkdir -p outputs/regression/cnn
PYTHONPATH=. conda run -n cupy_py311_cuda121 python scripts/train.py --task regression --model cnn --epochs 1 --checkpoint outputs/regression/cnn/model.npz
```

학습 실행 결과는 다음과 같다.

| train loss | train metric | test loss | test metric |
|---:|---:|---:|---:|
| `0.4332` | `-3.0822` | `0.0709` | `0.3040` |

## 5. 평가

저장된 checkpoint를 사용해 regression CNN을 평가한다.

```bash
PYTHONPATH=. conda run -n cupy_py311_cuda121 python scripts/evaluate.py --task regression --model cnn --checkpoint outputs/regression/cnn/model.npz
```

평가 결과는 다음과 같다.

| loss | metric | samples |
|---|---|---|
| `0.0703` | `0.3104` | `10000` |

Regression loss는 MSE이고 metric은 R2 score이다.
위 결과는 test split에서 R2 score `0.3104`를 의미한다.

## 6. 예측

`predict.py`는 regression 출력을 digit label로 복원한 prediction을 출력한다.

```bash
PYTHONPATH=. conda run -n cupy_py311_cuda121 python scripts/predict.py --task regression --model cnn --checkpoint outputs/regression/cnn/model.npz --n 16
```

예측 후처리는 `round(clip(logit * 9, 0, 9))` 규약을 사용한다.

## 7. 시각화

`visualize.py`는 regression CNN 학습 로그와 예측 결과 이미지를 저장한다.

```bash
PYTHONPATH=. MPLBACKEND=Agg conda run -n cupy_py311_cuda121 python scripts/visualize.py --task regression --model cnn --epochs 1 --output_dir outputs/regression/cnn
```

저장된 산출물은 다음과 같다.

| 파일 | 경로 |
|---|---|
| checkpoint | `outputs/regression/cnn/model.npz` |
| training log plot | `outputs/regression/cnn/training_log.png` |
| prediction plot | `outputs/regression/cnn/predictions.png` |

## 8. 설계 결정

- regression target은 `0..9` digit을 `0..1` 범위로 정규화하여 MSE 학습에 사용한다.
- predictor에서 scale 복원과 clip을 수행하여 최종 예측값을 digit 범위로 제한한다.
- Regression CNN은 1 epoch 기준으로 동작하지만, classification CNN보다 수치 안정성과 metric 개선 여지가 크다.
