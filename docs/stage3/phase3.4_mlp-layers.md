---
tags: [docs, stage3, nn, layers]
created: "2026-06-20"
updated: "2026-06-21"
---

# MLP 레이어 모듈

## 1. 개요

`src/nn/layers.py`는 MLP를 구성하는 레이어 모듈을 제공한다. `Module` 기반 클래스 계층을 통해 `Linear`, `Sigmoid`, `ReLU`, `Sequential` 레이어를 일관된 인터페이스로 구현한다. 모든 레이어는 `forward`와 `backward` 메서드를 가지며, 파라미터가 있는 레이어는 `params`와 `grads` 리스트를 통해 optimizer와 연동한다. PyTorch의 `torch.nn` 모듈에 대응하는 numpy-only 구현이다.

**목표**
- `Module` 기반 클래스로 forward/backward/train/eval 인터페이스를 통일한다.
- `Linear`의 He 초기화와 backward gradient 누적을 구현한다.
- `Sequential`로 레이어를 체인으로 연결하고 params/grads를 자동으로 집계한다.

## 2. 개념

### 2.1. Module 인터페이스

딥러닝 프레임워크에서 모든 레이어는 같은 인터페이스를 따르면 조합이 자유롭다. 이 프로젝트의 `Module`은 최소한의 인터페이스를 정의한다.

| 메서드/속성 | 역할 |
|---|---|
| `forward(x)` | 순전파 계산. 입력 `x`를 받아 출력을 반환한다. |
| `backward(dout)` | 역전파 계산. 상위 gradient `dout`을 받아 하위 gradient를 반환한다. |
| `params` | 파라미터 배열 리스트. optimizer가 업데이트 대상으로 참조한다. |
| `grads` | gradient 배열 리스트. `params`와 같은 순서로 대응한다. |
| `train()` / `eval()` | 학습/평가 모드 전환. `Dropout` 등 모드에 따라 동작이 달라지는 레이어에 사용한다. |
| `__call__(x)` | `forward(x)`를 호출. `layer(x)` 형식으로 사용 가능하다. |

### 2.2. Linear

완전 연결(fully-connected) 레이어이다. 입력 $x$에 가중치 행렬 $W$를 곱하고 편향 $b$를 더하는 아핀 변환을 수행한다.

**Forward**

$$
y = xW + b, \quad x \in \mathbb{R}^{N \times D_{in}},\ W \in \mathbb{R}^{D_{in} \times D_{out}},\ b \in \mathbb{R}^{D_{out}}
$$

입력 $x$는 배치 크기 $N$, 입력 차원 $D_{in}$의 행렬이다. 출력 $y$의 shape는 $(N, D_{out})$이다.

**Backward**

상위 레이어에서 전달된 gradient $\frac{\partial L}{\partial y}$를 `dout`이라 하면, chain rule로 각 파라미터와 입력에 대한 gradient는 다음과 같다.

$$
\frac{\partial L}{\partial W} = x^T \cdot \text{dout}, \quad
\frac{\partial L}{\partial b} = \sum_{i} \text{dout}_i, \quad
\frac{\partial L}{\partial x} = \text{dout} \cdot W^T
$$

$\frac{\partial L}{\partial W}$와 $\frac{\partial L}{\partial b}$는 optimizer가 참조하는 `grads` 리스트에 in-place로 저장하고, $\frac{\partial L}{\partial x}$는 하위 레이어로 반환한다.

**He 초기화**

$$
W \sim \mathcal{N}\!\left(0,\ \sqrt{\frac{2}{D_{in}}}\right)
$$

ReLU 활성화 이후 신호 분산이 레이어를 거칠수록 감소하는 문제를 방지하는 초기화 방식이다. sigmoid 계열에는 Xavier 초기화가 적합하지만, 이 프로젝트에서는 hidden layer의 activation 종류에 관계없이 He 초기화를 통일하여 사용한다.

### 2.3. Sigmoid

활성화 레이어로서 `src/nn/activations.py`의 `sigmoid` 함수를 감싸고, backward에 필요한 forward 출력을 저장한다.

**Forward**

$$
\hat{y} = \sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Backward**

sigmoid의 미분은 자기 자신으로 표현된다.

$$
\frac{\partial L}{\partial x} = \text{dout} \cdot \sigma(x) \cdot (1 - \sigma(x))
$$

$\sigma(x)$는 `forward`에서 `self._out`으로 저장한 값을 재사용하므로 backward에서 재계산하지 않는다.

### 2.4. ReLU

활성화 레이어로서 양수 구간은 그대로 통과시키고 음수 구간은 0으로 만든다.

**Forward**

$$
\hat{y} = \max(0, x) = \begin{cases} x & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

**Backward**

ReLU의 미분은 forward에서 양수였던 위치는 1, 음수였던 위치는 0이다.

$$
\frac{\partial L}{\partial x} = \text{dout} \cdot \mathbf{1}[x > 0]
$$

$\mathbf{1}[x > 0]$은 `forward`에서 `self._mask = x > 0`으로 저장한 boolean 배열이다. 재계산 없이 마스크를 그대로 곱한다.

### 2.5. Sequential

여러 레이어를 순서대로 연결하는 컨테이너이다. forward는 순방향, backward는 역방향으로 레이어를 순회한다.

**Forward**

$$
y = f_n(\cdots f_2(f_1(x)) \cdots)
$$

$f_1, f_2, \ldots, f_n$은 등록된 레이어이다. 각 레이어의 출력이 다음 레이어의 입력이 된다.

**Backward**

$$
\frac{\partial L}{\partial x} = f_1'\!\left(f_2'\!\left(\cdots f_n'(\text{dout}) \cdots\right)\right)
$$

역순으로 각 레이어의 `backward`를 호출하며 gradient를 전파한다. `params`와 `grads`는 생성자에서 하위 레이어의 리스트를 `extend`로 수집하므로, optimizer는 `Sequential.params` 하나로 모든 파라미터를 접근한다.

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `Module` | 클래스 | - | - | 기반 클래스. forward/backward/train/eval 정의 |
| `Linear` | 클래스 | `in_features`, `out_features` | layer instance | 완전 연결 레이어 |
| `Sigmoid` | 클래스 | - | layer instance | sigmoid 활성화 레이어 |
| `ReLU` | 클래스 | - | layer instance | ReLU 활성화 레이어 |
| `Sequential` | 클래스 | `*layers` | layer instance | 레이어 체인 |

### 3.1. Linear

```python
class Linear(Module):
    def __init__(self, in_features, out_features, seed=None):
        super().__init__()
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / in_features)  # He init
        self.w = (rng.standard_normal((in_features, out_features)) * scale).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)
        self.params = [self.w, self.b]
        self.grads = [self.grad_w, self.grad_b]

    def forward(self, x):
        self._x = x
        return x @ self.w + self.b

    def backward(self, dout):
        self.grad_w[...] = self._x.T @ dout
        self.grad_b[...] = dout.sum(axis=0)
        return dout @ self.w.T
```

He 초기화(`scale = sqrt(2 / in_features)`)는 ReLU 활성화 이후 gradient 소실을 완화하는 초기화 방식이다. `np.random.default_rng(seed)`는 재현 가능한 초기화를 위해 seed를 지정할 수 있게 한다.

### 3.2. Sigmoid

```python
class Sigmoid(Module):
    def forward(self, x):
        self._out = sigmoid(x)
        return self._out

    def backward(self, dout):
        return dout * self._out * (1.0 - self._out)
```

`forward`에서 `self._out`에 sigmoid 출력을 저장한다. `backward`에서 `sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))`를 계산할 때 이 값을 재사용하므로 sigmoid를 다시 계산하지 않는다.

### 3.3. ReLU

```python
class ReLU(Module):
    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout):
        return dout * self._mask
```

`forward`에서 `self._mask = x > 0`으로 boolean 마스크를 저장한다. `backward`에서 마스크를 그대로 곱하여 음수였던 위치의 gradient를 0으로 만든다.

### 3.4. Sequential

```python
class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        for layer in self.layers:
            self.params.extend(layer.params)
            self.grads.extend(layer.grads)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
```

`Sequential`은 생성자에서 하위 레이어의 `params`와 `grads`를 자신의 리스트에 추가한다. optimizer는 `Sequential.params`를 통해 모든 파라미터에 접근할 수 있다. `backward`는 레이어를 역순으로 순회한다.

## 4. 사용법

최소 사용 예제는 다음과 같다.

```python
import numpy as np
from src.nn.layers import Linear, Sigmoid, Sequential

model = Sequential(
    Linear(784, 256, seed=42),
    Sigmoid(),
    Linear(256, 10, seed=42),
)

x = np.random.randn(32, 784).astype(np.float32)
logits = model(x)
print(logits.shape)
print(len(model.params))
```

예상 출력은 다음과 같다.

```text
(32, 10)
4
```

프로젝트 통합 예제는 다음과 같다. `MLP` 모델은 `Sequential`로 레이어를 조립하고 optimizer에 `model.params`와 `model.grads`를 전달한다.

```python
from src.nn.layers import Linear, Sigmoid, Sequential
from src.nn.losses import cross_entropy_grad

model = Sequential(Linear(784, 256), Sigmoid(), Linear(256, 10))
optimizer = SGD(model, lr=0.01)

logits = model.forward(x_batch)
grad = cross_entropy_grad(logits, y_batch)
model.backward(grad)
optimizer.step()
```

## 5. 테스트

테스트 파일은 `tests/stage3/test_layers.py`이다.

```bash
conda run -n numpy_py311 pytest tests/stage3/test_layers.py -v
```

테스트 구성은 다음과 같다.

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestLinear` | 6 | forward shape, bias 동작, backward dx shape, grad_w/grad_b shape, in-place 대입 유지 |
| `TestSigmoid` | 3 | forward 범위, backward shape, 0 입력시 gradient 0.25 |
| `TestReLU` | 3 | 음수 0 변환, backward mask 적용, shape 보존 |
| `TestSequential` | 5 | params/grads 수집, forward shape, backward shape, train/eval 전파 |

## 6. 요약

`layers.py`는 `Module` 기반의 `Linear`, `Sigmoid`, `ReLU`, `Sequential` 레이어를 제공한다. `Linear.backward`는 chain rule로 `grad_w`, `grad_b`, `dx`를 계산하며, in-place 대입으로 `grads` 리스트와 배열 참조 일관성을 유지한다. `Sequential`은 하위 레이어의 `params`와 `grads`를 자동으로 집계하여 optimizer가 단일 진입점으로 모든 파라미터를 업데이트할 수 있게 한다.

다음 Phase에서는 [[phase3.5_cnn-layers]]를 다룬다.
