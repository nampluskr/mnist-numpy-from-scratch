---
tags: [stage7, tutorial, multiclass, cnn]
created: 2026-06-18
updated: 2026-06-20
---

# Phase 7.2 Multiclass tutorial

## 1. 과제 개요

Multiclass 과제는 MNIST 이미지를 0부터 9까지 10개 class 중 하나로 분류한다.
CNN 구현은 CuPy를 사용해 convolution 계층을 GPU에서 실행하고, MLP와 같은 task, loss, metric, CLI 규약을 사용한다.

## 2. 데이터 규약

`MnistDataset`은 로컬 MNIST gz 파일을 읽고 이미지를 `(784,)` float32 배열로 정규화한다.
CNN은 forward 시작 시 입력을 `(N, 1, 28, 28)` 형태로 reshape하여 convolution 계층에 전달한다.

| 항목 | 값 |
|---|---|
| 입력 shape | `(784,)` |
| CNN 내부 shape | `(N, 1, 28, 28)` |
| target shape | `(10,)` |
| target 변환 | digit label -> one-hot |
| output_dim | `10` |
| prediction_mode | `argmax` |

## 3. 모델 구성

CNN은 `src/models/cnn.py`에서 구성한다.
Convolution 계층은 CuPy 배열로 계산하고, flatten 이후 fully connected 계층은 NumPy 배열로 계산한다.

```text
Conv2d(1, 32, 3, padding=1) -> ReLU -> MaxPool2d(2, 2)
Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool2d(2, 2)
Flatten -> Linear(3136, 256) -> ReLU -> Linear(256, 10)
```

## 4. 학습

다음 명령은 multiclass CNN을 1 epoch 학습하고 checkpoint를 저장한다.

```bash
mkdir -p outputs/multiclass/cnn
PYTHONPATH=. conda run -n cupy_py311_cuda121 python scripts/train.py --task multiclass --model cnn --epochs 1 --checkpoint outputs/multiclass/cnn/model.npz
```

학습 실행 결과는 다음과 같다.

| train loss | train metric | test loss | test metric |
|---:|---:|---:|---:|
| `0.4435` | `0.8602` | `0.2107` | `0.9365` |

## 5. 평가

저장된 checkpoint를 불러와 test split 10000개 샘플을 평가한다.

```bash
PYTHONPATH=. conda run -n cupy_py311_cuda121 python scripts/evaluate.py --task multiclass --model cnn --checkpoint outputs/multiclass/cnn/model.npz
```

평가 결과는 다음과 같다.

| loss | metric | samples |
|---|---|---|
| `0.2106` | `0.9345` | `10000` |

Multiclass metric은 accuracy이므로, 위 결과는 test accuracy `0.9345`를 의미한다.

## 6. 예측

`predict.py`는 test split 앞쪽 샘플을 읽고 checkpoint 모델의 decoded prediction을 출력한다.

```bash
PYTHONPATH=. conda run -n cupy_py311_cuda121 python scripts/predict.py --task multiclass --model cnn --checkpoint outputs/multiclass/cnn/model.npz --n 16
```

Multiclass CNN 예측은 raw logit의 `argmax` 결과이며, 출력값은 digit class index이다.

## 7. 시각화

`visualize.py`는 학습 로그와 예측 결과 이미지를 저장한다.

```bash
PYTHONPATH=. MPLBACKEND=Agg conda run -n cupy_py311_cuda121 python scripts/visualize.py --task multiclass --model cnn --epochs 1 --output_dir outputs/multiclass/cnn
```

저장된 산출물은 다음과 같다.

| 파일 | 경로 |
|---|---|
| checkpoint | `outputs/multiclass/cnn/model.npz` |
| training log plot | `outputs/multiclass/cnn/training_log.png` |
| prediction plot | `outputs/multiclass/cnn/predictions.png` |

## 8. 설계 결정

- CNN은 `cupy_py311_cuda121` environment에서 실행한다.
- 모델은 raw logit을 반환하고, cross entropy 계산과 `argmax` 후처리는 task/loss/predictor 계층에서 처리한다.
- Codex 실행 환경에서는 GPU device가 노출되지 않으므로 CNN GPU 실행 결과는 사용자 WSL terminal에서 수집한다.
