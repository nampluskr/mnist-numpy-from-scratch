---
tags: [docs, stage3, nn, metrics]
created: "2026-06-20"
updated: "2026-06-21"
---

# 평가 지표

## 1. 개요

`src/nn/metrics.py`는 세 가지 평가 지표를 제공한다. `accuracy`, `binary_accuracy`, `r2_score`는 모두 `np.ndarray` logit과 target을 입력으로 받아 단일 스칼라를 반환한다. 손실 함수와 마찬가지로 logit 입력을 기준으로 하며, 내부에서 필요한 변환을 처리한다. Trainer와 Evaluator가 task에 따라 적절한 metric 함수를 선택하여 epoch별 성능을 집계한다.

**목표**
- `accuracy`로 multiclass 분류 정확도를 logit 입력으로 계산한다.
- `binary_accuracy`로 binary 분류 정확도를 logit 입력으로 계산한다.
- `r2_score`로 regression 설명력을 계산한다.

## 2. 개념

### 2.1. 평가 지표의 역할

손실 함수는 학습에 사용하지만 직관적인 해석이 어렵다. 평가 지표는 손실과 별개로 모델 성능을 사람이 이해하기 쉬운 단위로 표현한다. 학습 중에는 참고 지표로만 사용하고, 파라미터 업데이트에는 영향을 주지 않는다.

이 프로젝트의 task별 평가 지표는 다음과 같다.

| task | 지표 | 의미 | 범위 |
|---|---|---|---|
| `multiclass` | `accuracy` | 전체 중 정확히 예측한 비율 | `[0, 1]` |
| `binary` | `binary_accuracy` | 전체 중 임계값 기준 정확히 예측한 비율 | `[0, 1]` |
| `regression` | `r2_score` | 예측이 분산을 얼마나 설명하는지 | `(-inf, 1]` |

### 2.2. Accuracy

다중 클래스 분류에서 전체 샘플 중 정답을 맞힌 비율이다. logit에서 값이 가장 큰 인덱스를 예측 클래스로 선택하고, one-hot 정답의 클래스 인덱스와 비교한다.

$$
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\bigl[\arg\max(\hat{z}_i) = \arg\max(y_i)\bigr]
$$

$\hat{z}_i$는 샘플 $i$의 logit 벡터, $y_i$는 one-hot 정답 벡터이다. $\mathbf{1}[\cdot]$은 조건이 참이면 1, 거짓이면 0을 반환하는 지시 함수이다. `softmax`는 단조 증가 함수이므로 적용 전후에 `argmax` 결과가 바뀌지 않아 logit에 직접 적용해도 동일하다.

### 2.3. Binary Accuracy

이진 분류에서 전체 샘플 중 임계값 기준으로 정답을 맞힌 비율이다. `sigmoid(z) >= 0.5`는 `z >= 0`과 동치이므로 sigmoid를 계산하지 않고 logit의 부호만 확인한다.

$$
\text{Binary Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\bigl[\mathbf{1}[\hat{z}_i \geq 0] = y_i\bigr]
$$

$\hat{z}_i$는 scalar logit, $y_i \in \{0, 1\}$은 이진 정답이다. logit이 양수이면 클래스 1, 음수이면 클래스 0으로 예측한다.

### 2.4. R2 Score

회귀에서 모델이 데이터 분산을 얼마나 설명하는지 나타내는 지표이다. 결정 계수(coefficient of determination)라고도 한다.

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}, \quad SS_{res} = \sum_{i=1}^{N}(y_i - \hat{y}_i)^2, \quad SS_{tot} = \sum_{i=1}^{N}(y_i - \bar{y})^2
$$

$SS_{res}$는 잔차 제곱합(모델 예측과 실제값의 오차), $SS_{tot}$는 전체 분산(실제값과 평균의 편차)이다. $\bar{y}$는 정답의 평균이다. 완벽한 예측이면 $SS_{res} = 0$이므로 $R^2 = 1.0$, 모든 샘플을 평균으로 예측하면 $R^2 = 0.0$, 평균 예측보다 나쁘면 음수가 된다.

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `accuracy` | 함수 | `logits (N, C)`, `targets (N, C)` | scalar | multiclass 정확도 |
| `binary_accuracy` | 함수 | `logits (N, 1)`, `targets (N, 1)` | scalar | binary 분류 정확도 |
| `r2_score` | 함수 | `preds (N, 1)`, `targets (N, 1)` | scalar | 결정 계수 |

### 3.1. Accuracy

```python
def accuracy(logits, targets):
    return (logits.argmax(axis=1) == targets.argmax(axis=1)).mean()
```

`logits.argmax(axis=1)`는 각 행에서 가장 큰 값의 인덱스를 반환한다. `softmax`를 먼저 적용할 필요가 없다. `softmax`는 단조 증가 함수이므로 `argmax`의 결과를 바꾸지 않는다. `targets.argmax(axis=1)`는 one-hot target에서 정답 클래스 인덱스를 추출한다.

### 3.2. Binary Accuracy

```python
def binary_accuracy(logits, targets):
    return ((logits >= 0.0) == targets.astype(bool)).mean()
```

`sigmoid(x) >= 0.5`는 `x >= 0`과 동치이므로 `sigmoid`를 계산하지 않고 logit 부호만 확인한다. `targets.astype(bool)`은 target 배열을 boolean으로 변환하여 비교한다.

### 3.3. R2 Score

```python
def r2_score(preds, targets):
    ss_res = np.sum((preds - targets) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-8)
```

`ss_tot + 1e-8`은 모든 target이 동일한 경우(분산 0)의 division by zero를 방지한다.

## 4. 사용법

최소 사용 예제는 다음과 같다.

```python
import numpy as np
from src.nn.metrics import accuracy, binary_accuracy, r2_score

# multiclass
logits = np.array([[2.0, 0.5, 0.1], [0.1, 0.5, 2.0]], dtype=np.float32)
targets = np.array([[1., 0., 0.], [0., 0., 1.]], dtype=np.float32)
print(accuracy(logits, targets))

# binary
logits_b = np.array([[1.0], [-1.0]], dtype=np.float32)
targets_b = np.array([[1.], [0.]], dtype=np.float32)
print(binary_accuracy(logits_b, targets_b))

# regression
preds = np.array([[0.5], [1.0]], dtype=np.float32)
targets_r = np.array([[0.5], [1.0]], dtype=np.float32)
print(r2_score(preds, targets_r))
```

예상 출력은 다음과 같다.

```text
1.0
1.0
1.0
```

프로젝트 통합 예제는 다음과 같다. Evaluator에서 task에 따라 metric 함수를 선택하여 집계한다.

```python
from src.nn.metrics import accuracy

total_metric = 0.0
for images, targets in test_loader:
    logits = model.forward(images)
    total_metric += accuracy(logits, targets) * len(images)

print(total_metric / len(test_dataset))
```

## 5. 테스트

테스트 파일은 `tests/stage3/test_metrics.py`이다.

```bash
conda run -n numpy_py311 pytest tests/stage3/test_metrics.py -v
```

테스트 구성은 다음과 같다.

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestAccuracy` | 3 | 완벽한 예측 1.0, 0 예측 0.0, scalar 반환 |
| `TestBinaryAccuracy` | 3 | 완벽한 예측 1.0, logit 부호 기반 경계값 검증, scalar 반환 |
| `TestR2Score` | 4 | 완벽한 예측 1.0, 평균 예측 0.0 근사, 음수 반환 가능, scalar 반환 |

## 6. 요약

`metrics.py`는 `accuracy`, `binary_accuracy`, `r2_score` 세 가지 평가 지표를 제공한다. `accuracy`와 `binary_accuracy`는 logit 입력으로 activation 없이 동작하며, `r2_score`는 regression 출력을 분산 기준으로 평가한다. 학습 루프에서 loss와 함께 epoch별 성능 추적에 사용한다.

다음 Phase에서는 [[phase3.4_mlp-layers]]를 다룬다.
