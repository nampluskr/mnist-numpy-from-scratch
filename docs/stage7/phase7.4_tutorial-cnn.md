---
tags: [stage7, tutorial, binary, cnn]
created: 2026-06-18
updated: 2026-06-18
---

# Phase 7.4 Binary tutorial

## 1. 과제 개요

Binary 과제는 MNIST digit을 짝수와 홀수 두 범주로 분류한다.
CNN 구현은 CuPy convolution 계층을 사용하며, binary target 규약은 MLP와 동일하게 `label % 2`를 사용한다.

## 2. 데이터 규약

`MnistDataset`은 원본 digit label을 binary target으로 변환한다.
CNN은 하나의 logit을 출력하고, predictor는 sigmoid 값을 threshold로 변환한다.

| 항목 | 값 |
|---|---|
| 입력 shape | `(784,)` |
| CNN 내부 shape | `(N, 1, 28, 28)` |
| target shape | `(1,)` |
| target 변환 | `label % 2` |
| output_dim | `1` |
| prediction_mode | `threshold` |

## 3. 모델 구성

Binary CNN은 multiclass CNN과 같은 convolution 구조를 사용한다.
마지막 `Linear`만 output dimension 1로 구성한다.

```text
Conv2d(1, 32, 3, padding=1) -> ReLU -> MaxPool2d(2, 2)
Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool2d(2, 2)
Flatten -> Linear(3136, 256) -> ReLU -> Linear(256, 1)
```

## 4. 학습

다음 명령은 binary CNN을 1 epoch 학습하고 checkpoint를 저장한다.

```bash
mkdir -p outputs/binary/cnn
PYTHONPATH=. conda run -n cupy_py311_cuda121 python scripts/train.py --task binary --model cnn --epochs 1 --checkpoint outputs/binary/cnn/model.npz
```

생성된 checkpoint는 `outputs/binary/cnn/model.npz`에 저장한다.

## 5. 평가

저장된 checkpoint를 사용해 binary CNN을 평가한다.

```bash
PYTHONPATH=. conda run -n cupy_py311_cuda121 python scripts/evaluate.py --task binary --model cnn --checkpoint outputs/binary/cnn/model.npz
```

평가 결과는 다음과 같다.

| loss | metric | samples |
|---|---|---|
| `0.1780` | `0.9347` | `10000` |

Binary metric은 accuracy이므로, 위 결과는 test accuracy `0.9347`을 의미한다.

## 6. 예측

`predict.py`는 decoded binary prediction을 출력한다.

```bash
PYTHONPATH=. conda run -n cupy_py311_cuda121 python scripts/predict.py --task binary --model cnn --checkpoint outputs/binary/cnn/model.npz --n 16
```

예측 후처리는 sigmoid(logit) 값이 0.5 이상이면 1, 아니면 0으로 변환한다.

## 7. 시각화

`visualize.py`는 binary CNN 학습 로그와 예측 결과 이미지를 저장한다.

```bash
PYTHONPATH=. MPLBACKEND=Agg conda run -n cupy_py311_cuda121 python scripts/visualize.py --task binary --model cnn --epochs 1 --output_dir outputs/binary/cnn
```

저장된 산출물은 다음과 같다.

| 파일 | 경로 |
|---|---|
| checkpoint | `outputs/binary/cnn/model.npz` |
| training log plot | `outputs/binary/cnn/training_log.png` |
| prediction plot | `outputs/binary/cnn/predictions.png` |

## 8. 설계 결정

- binary CNN은 하나의 raw logit을 출력하고 binary cross entropy가 sigmoid를 내부 처리한다.
- binary target 변환을 dataset 계층에 두어 training, evaluation, prediction의 입력 규약을 통일한다.
- GPU 실행은 `cupy_py311_cuda121` environment를 기준으로 기록한다.
