---
tags: [docs, stage3, nn, activations]
created: "2026-06-20"
updated: "2026-06-20"
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

### 2.2. 수치 안정성

`sigmoid`와 `softmax`는 지수 함수를 포함하므로 큰 입력값에서 overflow가 발생한다.

`sigmoid(x)`에서 `x`가 매우 크면 `exp(-x)`가 0에 가까워 안전하지만, `x`가 매우 작으면 `exp(-x)`가 overflow한다. 이를 방지하기 위해 양수 입력과 음수 입력을 분기 처리한다.

- `x >= 0`: `1 / (1 + exp(-x))`
- `x < 0`: `exp(x) / (1 + exp(x))`

`softmax(x)`에서는 각 행에서 최대값을 빼는 max subtraction 기법을 적용한다. `exp(x - max(x))`는 최대값이 0이 되므로 overflow 없이 계산된다.

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `sigmoid` | 함수 | `(N,)` 또는 `(N, 1)` float32 | 같은 shape | `(0, 1)` 범위 이진 확률 |
| `softmax` | 함수 | `(N, C)` float32 | `(N, C)` float32 | 행 합이 1인 클래스 확률 |
| `relu` | 함수 | `(N, D)` float32 | `(N, D)` float32 | 음수를 0으로 클리핑 |
| `identity` | 함수 | `(N, D)` float32 | `(N, D)` float32 | 입력 그대로 반환 |

### 3.1. sigmoid

```python
def sigmoid(x):
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    out[~pos] = np.exp(x[~pos]) / (1.0 + np.exp(x[~pos]))
    return out
```

양수 영역과 음수 영역을 boolean mask로 분기 처리하여 수치 안정성을 확보한다. `np.empty_like(x)`로 출력 배열을 미리 할당하고 마스크 인덱싱으로 채운다.

### 3.2. softmax

```python
def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
```

`x.max(axis=-1, keepdims=True)`는 각 행의 최대값을 구하고, 이를 뺀 뒤 지수 함수를 적용하여 overflow를 방지한다. `keepdims=True`는 브로드캐스팅을 위해 차원을 유지한다.

### 3.3. relu와 identity

```python
def relu(x):
    return np.maximum(0.0, x)

def identity(x):
    return x
```

`relu`는 `np.maximum(0.0, x)`로 음수를 0으로 클리핑한다. `identity`는 입력을 그대로 반환하며, regression 출력 레이어에서 activation 없이 logit을 통과시킬 때 사용한다.

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
