---
tags: [stage7, tutorial, binary, mlp]
created: 2026-06-18
updated: 2026-06-18
---

# Phase 7.4 Binary 튜토리얼: MLP

## 1. 과제 개요

Binary 과제는 MNIST digit을 짝수와 홀수 두 범주로 분류한다.
이 프로젝트의 binary 규약은 label을 `label % 2`로 변환하므로, 출력 0은 짝수 class, 출력 1은 홀수 class를 의미한다.

## 2. 데이터 규약

`MnistDataset`은 원본 digit label을 binary target으로 변환한다.
모델은 하나의 logit을 출력하고, predictor는 sigmoid 값을 threshold로 변환한다.

| 항목 | 값 |
|---|---|
| 입력 shape | `(784,)` |
| target shape | `(1,)` |
| target 변환 | `label % 2` |
| output_dim | `1` |
| prediction_mode | `threshold` |

## 3. 모델 구성

Binary MLP는 multiclass MLP와 같은 hidden 구조를 사용한다.
마지막 `Linear`만 output dimension 1로 구성한다.

```text
Linear(784, 256) -> Sigmoid -> Linear(256, 128) -> Sigmoid -> Linear(128, 1)
```

## 4. 학습

다음 명령은 binary MLP를 학습하고 checkpoint를 저장한다.

```bash
PYTHONPATH=. conda run -n numpy_env python scripts/train.py --task binary --model mlp --checkpoint outputs/binary/mlp/model.npz
```

Phase 7.2에서 생성한 산출물은 다음 위치에 둔다.

| 파일 | 경로 |
|---|---|
| checkpoint | `outputs/binary/mlp/model.npz` |
| training log plot | `outputs/binary/mlp/training_log.png` |
| prediction plot | `outputs/binary/mlp/predictions.png` |

## 5. 평가

저장된 checkpoint를 사용해 binary MLP를 평가한다.

```bash
PYTHONPATH=. conda run -n numpy_env python scripts/evaluate.py --task binary --model mlp --checkpoint outputs/binary/mlp/model.npz
```

실행 결과는 다음과 같다.

| loss | metric | samples |
|---|---|---|
| `0.2923` | `0.8814` | `10000` |

Binary metric은 accuracy이므로, 위 결과는 test accuracy 0.8814를 의미한다.

## 6. 예측

`predict.py`는 decoded binary prediction을 출력한다.

```bash
PYTHONPATH=. conda run -n numpy_env python scripts/predict.py --task binary --model mlp --checkpoint outputs/binary/mlp/model.npz --n 16
```

예측 후처리는 sigmoid(logit) 값이 0.5 이상이면 1, 아니면 0으로 변환한다.

## 7. 시각화

`visualize.py`는 binary 학습 로그와 예측 결과 이미지를 저장한다.

```bash
PYTHONPATH=. MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib conda run -n numpy_env python scripts/visualize.py --task binary --model mlp --output_dir outputs/binary/mlp
```

`predictions.png`의 true label과 predicted label은 binary target 기준의 0 또는 1이다.

## 8. 설계 결정

- 하나의 logit만 출력하고 binary cross entropy가 sigmoid를 내부 처리한다.
- binary target 변환을 dataset 계층에 두어 training, evaluation, prediction의 입력 규약을 통일한다.
- `--model mlp`를 명시하여 CNN 실행과 분리하고 CPU NumPy 기준 결과만 기록한다.
