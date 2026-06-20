---
tags: [docs, stage3, nn, activations]
created: "2026-06-20"
updated: "2026-06-21"
---

# 활성화 함수

## 1. 개요

`src/nn/activations.py`는 forward pass 전용 활성화 함수 4개를 제공한다. `sigmoid`, `softmax`, `relu`, `identity`는 모두 `np.ndarray`를 입력으로 받아 같은 shape의 `np.ndarray`를 반환한다. 이 함수들은 `src/nn/layers.py`의 레이어 모듈이 내부에서 호출하거나, `src/nn/losses.py`가 logit에 activation을 적용하는 데 사용한다. PyTorch의 `torch.nn.functional`에 대응하는 numpy-only 구현이다.

**목표**
- `sigmoid`를 수치적으로 안정적인 방식으로 구현한다.
- `softmax`의 overflow를 max subtraction으로 방지한다.
- `relu`와 `identity`를 간단하고 일관된 인터페이스로 제공한다.

## 2. 개념

### 2.1. 활성화 함수의 역할

활성화 함수는 레이어의 선형 변환 출력(logit)에 비선형성을 추가한다. 비선형 활성화가 없으면 여러 레이어를 쌓아도 하나의 선형 변환과 동일하므로 복잡한 패턴을 학습할 수 없다. task에 따라 출력 레이어의 활성화 함수가 달라진다.

이 프로젝트에서 task별 활성화 함수 선택은 다음과 같다.

| task | 출력 활성화 | 이유 |
|---|---|---|
| `multiclass` | `softmax` | 클래스 확률 합이 1이 되도록 정규화 |
| `binary` | `sigmoid` | 이진 확률을 `[0, 1]` 범위로 변환 |
| `regression` | `identity` | 원본 logit 값을 그대로 예측값으로 사용 |

### 2.2. Sigmoid

이진 분류 출력 레이어에 사용한다. 임의의 실수 logit을 `(0, 1)` 범위의 확률로 변환한다.

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

$x = 0$이면 $\sigma(0) = 0.5$이고, $x \to +\infty$이면 1에 수렴, $x \to -\infty$이면 0에 수렴한다. 출력 범위는 열린 구간 $(0, 1)$이며 정확히 0 또는 1이 되지 않는다.

**수치 안정성**: $x$가 매우 작은 음수이면 $e^{-x}$가 overflow한다. 음수 영역에서는 수식을 변형하여 회피한다.

$$
\sigma(x) = \begin{cases} \dfrac{1}{1 + e^{-x}} & x \geq 0 \\[6pt] \dfrac{e^{x}}{1 + e^{x}} & x < 0 \end{cases}
$$

두 수식은 수학적으로 동일하지만, 각 영역에서 지수 함수 인수의 부호가 항상 음수이거나 양수로 제한되어 overflow가 발생하지 않는다.

### 2.3. Softmax

다중 클래스 분류 출력 레이어에 사용한다. logit 벡터를 클래스 확률 분포로 변환하며, 출력의 합이 1이 된다.

$$
\text{softmax}(z)_c = \frac{e^{z_c}}{\displaystyle\sum_{k=1}^{C} e^{z_k}}
$$

$z_c$는 클래스 $c$의 logit이다. 출력은 각 클래스에 속할 확률이며 모든 클래스에 대해 합산하면 1이다. 값이 가장 큰 logit의 확률이 가장 높게 나온다.

**수치 안정성**: $z_c$가 크면 $e^{z_c}$가 overflow한다. 각 행에서 최대값을 빼는 max subtraction으로 회피한다.

$$
\text{softmax}(z)_c = \frac{e^{z_c - \max(z)}}{\displaystyle\sum_{k=1}^{C} e^{z_k - \max(z)}}
$$

$\max(z)$를 빼면 최대값 항이 $e^0 = 1$이 되어 overflow 없이 계산된다. 분자와 분모에 동일한 값을 빼므로 결과는 수학적으로 동일하다.

### 2.4. ReLU

hidden layer의 비선형 활성화로 사용한다. 음수 입력을 0으로 만들어 네트워크에 희소성(sparsity)을 부여한다.

$$
\text{ReLU}(x) = \max(0, x) = \begin{cases} x & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

양수 구간에서 gradient가 1로 일정하므로, sigmoid나 tanh에서 발생하는 vanishing gradient 문제를 완화한다. 음수 구간에서는 gradient가 0이 되어 해당 뉴런이 업데이트되지 않는다(dead neuron).

### 2.5. Identity

activation이 필요 없는 regression 출력 레이어에 사용한다. 입력을 변환 없이 그대로 반환한다.

$$
\text{identity}(x) = x
$$

선형 변환 레이어 직후에 아무 변환도 하지 않으므로 출력이 raw logit 그대로 예측값이 된다. 다른 activation과 동일한 인터페이스를 제공하여 task별 분기 코드 없이 통일된 방식으로 조합할 수 있다.

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `sigmoid` | 함수 | `(N,)` 또는 `(N, 1)` float32 | 같은 shape | `(0, 1)` 범위 이진 확률 |
| `softmax` | 함수 | `(N, C)` float32 | `(N, C)` float32 | 행 합이 1인 클래스 확률 |
| `relu` | 함수 | `(N, D)` float32 | `(N, D)` float32 | 음수를 0으로 클리핑 |
| `identity` | 함수 | `(N, D)` float32 | `(N, D)` float32 | 입력 그대로 반환 |

### 3.1. Sigmoid

```python
def sigmoid(x):
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    out[~pos] = np.exp(x[~pos]) / (1.0 + np.exp(x[~pos]))
    return out
```

양수 영역(`x >= 0`)과 음수 영역(`x < 0`)을 boolean mask로 분기 처리하여 수치 안정성을 확보한다. `x`가 매우 작을 때 `exp(-x)`가 overflow하는 문제를 음수 영역 수식으로 회피한다. `np.empty_like(x)`로 출력 배열을 미리 할당하고 마스크 인덱싱으로 채운다.

### 3.2. Softmax

```python
def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
```

`x.max(axis=-1, keepdims=True)`로 각 행의 최대값을 빼는 max subtraction을 적용한다. `exp(x - max(x))`에서 최대값 항은 `exp(0) = 1`이 되어 overflow 없이 계산된다. `keepdims=True`는 빼기와 나누기 연산에서 브로드캐스팅을 위해 차원을 유지한다.

### 3.3. ReLU

```python
def relu(x):
    return np.maximum(0.0, x)
```

`np.maximum(0.0, x)`는 원소별로 0과 비교하여 음수를 0으로 클리핑한다. hidden layer의 비선형 활성화로 사용하며, 양수 구간에서는 gradient가 1로 일정하여 vanishing gradient 문제를 완화한다.

### 3.4. Identity

```python
def identity(x):
    return x
```

입력을 그대로 반환한다. regression 출력 레이어에서 activation 없이 logit을 예측값으로 사용할 때 쓴다. 다른 activation과 동일한 인터페이스를 유지하므로 task별 분기 없이 통일된 방식으로 호출할 수 있다.

## 4. 사용법

최소 사용 예제는 다음과 같다.

```python
import numpy as np
from src.nn.activations import sigmoid, softmax, relu, identity

x = np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)

print(sigmoid(x))
print(softmax(x))
print(relu(x))
print(identity(x))
```

예상 출력은 다음과 같다.

```text
[[0.2689  0.5     0.7311  0.8808]]
[[0.0321  0.0871  0.2369  0.6439]]
[[0.      0.      1.      2.    ]]
[[-1.     0.      1.      2.    ]]
```

프로젝트 통합 예제는 다음과 같다. `losses.py`가 내부에서 activation을 적용하므로 호출부에서 별도로 적용하지 않는다.

```python
from src.nn.losses import cross_entropy, cross_entropy_grad

# logits: (N, 10) raw output from MLP
# targets: (N, 10) one-hot
loss = cross_entropy(logits, targets)      # softmax 내부 적용
grad = cross_entropy_grad(logits, targets) # d(softmax+CE)/d(logits)
```

## 5. 테스트

테스트 파일은 `tests/stage3/test_activations.py`이다.

```bash
conda run -n numpy_py311 pytest tests/stage3/test_activations.py -v
```

테스트 구성은 다음과 같다.

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestSigmoid` | 4 | 출력 범위 `(0, 1)`, 0 입력시 0.5, shape 보존, 수치 안정성 |
| `TestSoftmax` | 4 | 행 합 1.0, 출력 범위 `(0, 1)`, argmax 보존, shape 보존 |
| `TestRelu` | 3 | 음수 0 변환, 양수 보존, shape 보존 |
| `TestIdentity` | 2 | 입력과 동일한 배열 반환, shape 보존 |

## 6. 요약

`activations.py`는 `sigmoid`, `softmax`, `relu`, `identity` 4개의 forward 전용 활성화 함수를 제공한다. `sigmoid`와 `softmax`는 수치 안정성을 위한 처리를 포함한다. 이 함수들은 레이어 모듈과 loss 함수가 내부에서 호출하므로 호출부에서 별도로 적용하지 않는다.

다음 Phase에서는 [[phase3.2_losses]]를 다룬다.
