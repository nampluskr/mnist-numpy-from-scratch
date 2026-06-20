---
tags: [docs, stage3, nn, losses]
created: "2026-06-20"
updated: "2026-06-21"
---

# 손실 함수와 gradient

## 1. 개요

`src/nn/losses.py`는 세 가지 손실 함수와 각 손실 함수의 gradient 함수를 제공한다. `cross_entropy`, `binary_cross_entropy`, `mse`는 스칼라 loss 값을 반환하고, 대응하는 `_grad` 함수는 backward에서 사용할 `d(loss)/d(logits)` 배열을 반환한다. `cross_entropy`와 `binary_cross_entropy`는 내부에서 activation을 적용하므로 호출부에서 별도로 `softmax`나 `sigmoid`를 적용하지 않는다. Trainer와 Evaluator가 task에 따라 적절한 함수를 선택하여 사용한다.

**목표**
- task별 손실 함수(`cross_entropy`, `binary_cross_entropy`, `mse`)를 logit 입력으로 제공한다.
- 각 손실 함수에 대응하는 gradient 함수를 제공하여 backward pass를 지원한다.
- 수치 안정성을 위해 log 계산 전에 epsilon clipping을 적용한다.

## 2. 개념

### 2.1. 손실 함수의 역할

손실 함수는 모델 출력과 정답 레이블 사이의 오차를 단일 스칼라로 요약한다. 학습 루프는 이 스칼라를 최소화하는 방향으로 파라미터를 업데이트한다. task마다 출력 형태와 정답 형태가 다르므로 손실 함수도 달라진다.

이 프로젝트의 task별 손실 함수는 다음과 같다.

| task | 손실 함수 | 내부 activation | 입력 형태 |
|---|---|---|---|
| `multiclass` | `cross_entropy` | `softmax` | logits `(N, C)`, targets one-hot `(N, C)` |
| `binary` | `binary_cross_entropy` | `sigmoid` | logits `(N, 1)`, targets `(N, 1)` |
| `regression` | `mse` | 없음 (identity) | preds `(N, 1)`, targets `(N, 1)` |

### 2.2. Cross Entropy

다중 클래스 분류(multiclass classification)에 사용한다. logit을 먼저 `softmax`로 확률로 변환한 뒤, 정답 클래스의 log 확률을 최대화하는 방향으로 학습한다.

**Softmax**

$$
\hat{y}_c = \text{softmax}(z)_c = \frac{e^{z_c}}{\sum_{k=1}^{C} e^{z_k}}
$$

$z_c$는 클래스 $c$의 logit이다. 출력 $\hat{y}_c$는 0과 1 사이의 확률이며 모든 클래스에 대해 합산하면 1이 된다.

**Negative Log-Likelihood (NLL)**

$$
L_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log(\hat{y}_{ic})
$$

$y_{ic}$는 one-hot 정답(정답 클래스만 1, 나머지 0)이므로, 합산은 정답 클래스 하나의 log 확률만 선택한다. $\hat{y}_{ic}$가 1에 가까울수록 loss가 0에 수렴한다.

**gradient (softmax + CE 결합)**

$$
\frac{\partial L_{\text{CE}}}{\partial z} = \frac{\hat{y} - y}{N}
$$

softmax와 cross entropy를 결합하면 gradient가 `(예측값 - 정답) / N`으로 단순해진다. 이것은 두 함수를 따로 미분하여 연쇄 법칙으로 결합한 결과와 동일하다.

### 2.3. Binary Cross Entropy

이진 분류(binary classification)에 사용한다. logit을 `sigmoid`로 변환한 뒤, 정답이 1인 경우는 $\log(\hat{y})$, 정답이 0인 경우는 $\log(1-\hat{y})$를 최대화한다.

**Sigmoid**

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

출력은 0과 1 사이의 단일 확률값이다. $\hat{y} > 0.5$이면 클래스 1, 그 이하이면 클래스 0으로 예측한다.

**Binary Cross Entropy**

$$
L_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \bigl[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \bigr]
$$

정답이 1인 샘플($y_i = 1$)은 첫 번째 항만 남고, 정답이 0인 샘플($y_i = 0$)은 두 번째 항만 남는다. 두 항 모두 예측이 맞을수록 0에 가까워진다.

**gradient (sigmoid + BCE 결합)**

$$
\frac{\partial L_{\text{BCE}}}{\partial z} = \frac{\hat{y} - y}{N}
$$

Cross Entropy와 동일하게, sigmoid와 BCE를 결합한 gradient도 `(예측값 - 정답) / N`으로 단순화된다.

### 2.4. Mean Squared Error

회귀(regression)에 사용한다. activation이 없으므로 logit이 곧 예측값이며, 예측값과 정답의 차이 제곱의 평균을 최소화한다.

**MSE**

$$
L_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
$$

$\hat{y}_i$는 모델이 출력한 실수 예측값, $y_i$는 정답 레이블이다. 이 프로젝트에서 `regression` task의 정답은 $y_i = \text{label} / 9.0 \in [0, 1]$이다.

**gradient**

$$
\frac{\partial L_{\text{MSE}}}{\partial \hat{y}} = \frac{2(\hat{y} - y)}{N}
$$

MSE는 activation이 없으므로 $\hat{y}$에 대한 gradient가 곧 logit에 대한 gradient이다. 계수 2는 관례적으로 유지하며, 학습률로 흡수할 수도 있다.

### 2.5. logit 입력 설계

손실 함수가 activation을 내부에서 처리하면 backward gradient 수식이 단순해진다. `softmax`와 `cross_entropy`를 결합한 gradient는 `softmax(logits) - targets`이며, `sigmoid`와 `binary_cross_entropy`를 결합한 gradient도 `sigmoid(logits) - targets`이다. 두 수식 모두 batch 크기로 나누어 정규화한다.

$$
\frac{\partial L}{\partial z} = \frac{\hat{y} - y}{N}
$$

여기서 $z$는 logit, $\hat{y}$는 activation 적용 후 예측값, $y$는 정답, $N$은 배치 크기이다.

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `cross_entropy` | 함수 | `logits (N, C)`, `targets (N, C)` | scalar | softmax + NLL loss |
| `cross_entropy_grad` | 함수 | `logits (N, C)`, `targets (N, C)` | `(N, C)` | `d(CE+softmax)/d(logits)` |
| `binary_cross_entropy` | 함수 | `logits (N, 1)`, `targets (N, 1)` | scalar | sigmoid + BCE loss |
| `binary_cross_entropy_grad` | 함수 | `logits (N, 1)`, `targets (N, 1)` | `(N, 1)` | `d(BCE+sigmoid)/d(logits)` |
| `mse` | 함수 | `preds (N, 1)`, `targets (N, 1)` | scalar | mean squared error |
| `mse_grad` | 함수 | `preds (N, 1)`, `targets (N, 1)` | `(N, 1)` | `d(MSE)/d(preds)` |

### 3.1. Cross Entropy

```python
def cross_entropy(logits, targets):
    preds = softmax(logits)
    return -np.mean(np.sum(targets * np.log(preds + 1e-8), axis=1))


def cross_entropy_grad(logits, targets):
    return (softmax(logits) - targets) / len(logits)
```

`log(preds + 1e-8)`는 `preds`가 0에 가까울 때 `log(0)` = `-inf` 발생을 방지한다. `targets * log(preds)`는 one-hot 정답이 1인 위치의 log 확률만 선택하며, 행 합산 후 평균을 취한다.

### 3.2. Binary Cross Entropy

```python
def binary_cross_entropy(logits, targets):
    preds = np.clip(sigmoid(logits), 1e-8, 1 - 1e-8)
    return -np.mean(targets * np.log(preds) + (1 - targets) * np.log(1 - preds))


def binary_cross_entropy_grad(logits, targets):
    return (sigmoid(logits) - targets) / len(logits)
```

`np.clip(sigmoid(logits), 1e-8, 1 - 1e-8)`는 `sigmoid` 출력을 `(0, 1)` 열린 구간으로 제한하여 `log(0)` 발생을 방지한다. gradient 계산에서는 clip 없이 `sigmoid`를 직접 사용한다.

### 3.3. Mean Squared Error

```python
def mse(preds, targets):
    return np.mean((preds - targets) ** 2)


def mse_grad(preds, targets):
    return 2.0 * (preds - targets) / len(preds)
```

`mse`는 activation이 없으므로 `preds`가 곧 raw logit이다. gradient는 `2 * (preds - targets) / N`이다.

## 4. 사용법

최소 사용 예제는 다음과 같다.

```python
import numpy as np
from src.nn.losses import cross_entropy, cross_entropy_grad

logits = np.random.randn(4, 10).astype(np.float32)
targets = np.eye(10, dtype=np.float32)[[0, 1, 2, 3]]

loss = cross_entropy(logits, targets)
grad = cross_entropy_grad(logits, targets)
print(loss)
print(grad.shape)
```

예상 출력은 다음과 같다.

```text
2.3025  # approx
(4, 10)
```

프로젝트 통합 예제는 다음과 같다. Trainer의 학습 루프에서 task에 따라 손실 함수와 gradient 함수를 선택한다.

```python
from src.nn.losses import cross_entropy, cross_entropy_grad

logits = model.forward(x_batch)
loss = cross_entropy(logits, y_batch)
grad = cross_entropy_grad(logits, y_batch)
model.backward(grad)
optimizer.step()
```

## 5. 테스트

테스트 파일은 `tests/stage3/test_losses.py`이다.

```bash
conda run -n numpy_py311 pytest tests/stage3/test_losses.py -v
```

테스트 구성은 다음과 같다.

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestCrossEntropy` | 4 | scalar 반환, 완벽한 예측시 loss 최소, grad shape, grad 합 0 |
| `TestBinaryCrossEntropy` | 4 | scalar 반환, 완벽한 예측시 loss 최소, grad shape, grad 합 0 근사 |
| `TestMse` | 4 | scalar 반환, 동일 입력시 0, grad shape, grad 방향 검증 |

## 6. 요약

`losses.py`는 `cross_entropy`, `binary_cross_entropy`, `mse` 세 가지 손실 함수와 각각의 `_grad` 함수를 제공한다. `cross_entropy`와 `binary_cross_entropy`는 activation을 내부에서 처리하므로 gradient 수식이 `(activation(logits) - targets) / N`으로 단순화된다. Trainer는 task에 따라 적절한 함수 쌍을 선택하여 forward와 backward를 수행한다.

다음 Phase에서는 [[phase3.3_metrics]]를 다룬다.
