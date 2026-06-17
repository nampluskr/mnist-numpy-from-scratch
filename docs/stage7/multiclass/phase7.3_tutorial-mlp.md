---
tags: [stage7, tutorial, multiclass, mlp]
created: 2026-06-18
updated: 2026-06-18
---

# Phase 7.3 Multiclass 튜토리얼: MLP

## 1. 과제 개요

Multiclass 과제는 MNIST 이미지를 0부터 9까지 10개 class 중 하나로 분류한다.
MLP 기준 구현은 NumPy만 사용하며, 후속 PyTorch, TensorFlow, JAX 프로젝트에서 같은 task와 CLI 사용법을 재사용하기 위한 기준선이다.

## 2. 데이터 규약

`MnistDataset`은 로컬 MNIST gz 파일을 읽고 이미지를 `(784,)` float32 배열로 정규화한다.
Multiclass target은 one-hot float32 배열이며, 모델 출력 차원은 10이다.

| 항목 | 값 |
|---|---|
| 입력 shape | `(784,)` |
| target shape | `(10,)` |
| target 변환 | digit label -> one-hot |
| output_dim | `10` |
| prediction_mode | `argmax` |

## 3. 모델 구성

MLP는 `src/models/mlp.py`에서 구성한다.
모든 task에서 hidden 구조는 같고, 마지막 출력 차원만 task 규약에 따라 달라진다.

```text
Linear(784, 256) -> Sigmoid -> Linear(256, 128) -> Sigmoid -> Linear(128, 10)
```

## 4. 학습

다음 명령은 multiclass MLP를 10 epoch 학습하고 checkpoint를 저장한다.

```bash
PYTHONPATH=. conda run -n numpy_env python scripts/train.py --task multiclass --model mlp --checkpoint outputs/multiclass/mlp/model.npz
```

Phase 7.2에서 생성한 산출물은 다음 위치에 둔다.

| 파일 | 경로 |
|---|---|
| checkpoint | `outputs/multiclass/mlp/model.npz` |
| training log plot | `outputs/multiclass/mlp/training_log.png` |
| prediction plot | `outputs/multiclass/mlp/predictions.png` |

## 5. 평가

저장된 checkpoint를 불러와 test split 10000개 샘플을 평가한다.

```bash
PYTHONPATH=. conda run -n numpy_env python scripts/evaluate.py --task multiclass --model mlp --checkpoint outputs/multiclass/mlp/model.npz
```

실행 결과는 다음과 같다.

| loss | metric | samples |
|---|---|---|
| `0.4499` | `0.8839` | `10000` |

Multiclass metric은 accuracy이므로, 위 결과는 test accuracy 0.8839를 의미한다.

## 6. 예측

`predict.py`는 test split 앞쪽 샘플을 읽고 checkpoint 모델의 decoded prediction을 출력한다.

```bash
PYTHONPATH=. conda run -n numpy_env python scripts/predict.py --task multiclass --model mlp --checkpoint outputs/multiclass/mlp/model.npz --n 16
```

Multiclass 예측은 raw logit의 `argmax` 결과이며, 출력값은 digit class index이다.

## 7. 시각화

`visualize.py`는 학습 로그와 예측 결과 이미지를 저장한다.
학습과 시각화를 한 번에 실행하므로 같은 seed와 epoch 설정을 사용한다.

```bash
PYTHONPATH=. MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib conda run -n numpy_env python scripts/visualize.py --task multiclass --model mlp --output_dir outputs/multiclass/mlp
```

저장된 `training_log.png`는 loss와 accuracy 곡선을 나란히 보여준다.
`predictions.png`는 true label과 predicted label을 함께 표시한다.

## 8. 설계 결정

- 모델은 raw logit을 반환하고, cross entropy 계산과 `argmax` 후처리는 task/loss/predictor 계층에서 처리한다.
- `--model mlp`를 명시하여 Stage 7.1 이후 추가된 CNN 분기와 실행 의도를 분리한다.
- CLI 직접 실행 시 프로젝트 루트 import를 보장하기 위해 `PYTHONPATH=.`를 함께 사용한다.
