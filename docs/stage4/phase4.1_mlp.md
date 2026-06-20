---
tags: [docs, stage4, models, mlp]
created: "2026-06-20"
updated: "2026-06-21"
---

# MLP 모델

## 1. 개요

`src/models/mlp.py`의 `MLP`는 `src/nn/` 레이어 모듈을 조립하여 구성한 3층 완전 연결 네트워크이다. `forward`는 raw logit을 반환하고, activation과 gradient 계산은 `src/nn/losses.py`가 처리한다. task에 따라 출력 차원이 달라지며, `MLP.params`와 `MLP.grads`를 통해 optimizer와 연동한다. CPU 기반 numpy 구현이며, 후속 PyTorch, TensorFlow, JAX 프로젝트와 동일한 public interface를 유지한다.

**목표**
- `src/nn/Sequential`로 `Linear` + `Sigmoid` 레이어를 조립하여 3층 MLP를 구성한다.
- task에 따라 출력 차원을 자동으로 설정한다.
- `params`와 `grads` property로 optimizer와 연동하는 인터페이스를 제공한다.

## 2. 개념

### 2.1. MLP란 무엇인가

MLP(Multi-Layer Perceptron)는 완전 연결(fully-connected) 레이어를 여러 층 쌓은 피드포워드 신경망이다. 각 레이어는 이전 레이어의 출력 전체를 입력으로 받아 다음 레이어로 전달한다. "완전 연결"이라는 이름은 한 레이어의 모든 뉴런이 다음 레이어의 모든 뉴런과 연결된다는 의미이다.

단순한 선형 변환만으로는 데이터의 비선형 패턴을 학습할 수 없다. 선형 변환을 아무리 많이 쌓아도 결국 하나의 선형 변환과 동일하기 때문이다. MLP는 각 레이어 사이에 비선형 활성화 함수를 삽입하여 이 한계를 극복한다. 이론적으로 충분한 뉴런을 가진 단일 hidden layer MLP는 임의의 연속 함수를 근사할 수 있다(Universal Approximation Theorem).

MNIST 손글씨 분류를 예로 들면, 입력 이미지는 28x28 = 784개의 픽셀값으로 이루어진 벡터이다. MLP는 이 784차원 벡터를 숨겨진 표현(hidden representation)을 거쳐 출력 차원으로 압축한다. hidden layer는 원본 픽셀 사이의 복잡한 상관관계를 추상적인 특징(feature)으로 변환하는 역할을 한다.

### 2.2. 이 프로젝트의 MLP 구조

이 프로젝트의 MLP는 `784 -> 256 -> 128 -> output_dim` 구조를 사용한다. 입력 784차원을 두 개의 hidden layer(256, 128)를 거쳐 task별 출력 차원으로 줄인다. 각 hidden layer는 `Linear` 변환 뒤 `Sigmoid` 활성화로 구성된다. 출력 레이어는 `Linear`만 있으며 activation이 없다.

```text
입력: (N, 784)
  |
  v
Linear(784, 256)  -- W: (784, 256), b: (256,)
  |
Sigmoid           -- 비선형 변환
  |
  v
Linear(256, 128)  -- W: (256, 128), b: (128,)
  |
Sigmoid           -- 비선형 변환
  |
  v
Linear(128, output_dim)  -- W: (128, output_dim), b: (output_dim,)
  |
  v
출력: (N, output_dim)  -- raw logit
```

출력 레이어 이후 activation은 `src/nn/losses.py`가 담당한다. `cross_entropy`는 내부에서 `softmax`를 적용하고, `binary_cross_entropy`는 `sigmoid`를 적용하며, `mse`는 activation 없이 logit을 그대로 예측값으로 사용한다.

task별 출력 차원과 해석은 다음과 같다.

| task | output_dim | 출력 activation | 해석 |
|---|---|---|---|
| `multiclass` | 10 | `softmax` (losses.py 내부) | 10개 클래스별 raw score |
| `binary` | 1 | `sigmoid` (losses.py 내부) | 이진 클래스의 raw score |
| `regression` | 1 | 없음 (identity) | 예측값 그대로 사용 |

### 2.3. 파라미터 수와 메모리

이 MLP의 학습 가능 파라미터는 W(가중치 행렬)와 b(편향 벡터)이다. 각 `Linear` 레이어의 파라미터 수는 다음과 같다.

`multiclass` task를 예로 들면 파라미터 수는 다음과 같다.

| 레이어 | W shape | b shape | 파라미터 수 |
|---|---|---|---|
| `Linear(784, 256)` | `(784, 256)` | `(256,)` | 200,960 |
| `Linear(256, 128)` | `(256, 128)` | `(128,)` | 32,896 |
| `Linear(128, 10)` | `(128, 10)` | `(10,)` | 1,290 |
| 합계 | - | - | 235,146 |

`MLP.params`는 이 6개(W, b 각 3쌍) 배열의 리스트이다. `float32` 기준으로 약 0.9 MB를 차지한다.

### 2.4. Forward Pass

Forward pass는 입력 $x$가 레이어를 순서대로 통과하며 출력 logit을 생성하는 과정이다.

**1층: Linear(784, 256)**

$$
z^{(1)} = x \cdot W^{(1)} + b^{(1)}, \quad x \in \mathbb{R}^{N \times 784},\ W^{(1)} \in \mathbb{R}^{784 \times 256}
$$

출력 $z^{(1)}$의 shape는 $(N, 256)$이다. $N$은 배치 크기(batch size)이며, 한 번의 forward에서 여러 샘플을 동시에 처리한다.

**1층 Sigmoid 활성화**

$$
a^{(1)} = \sigma(z^{(1)}) = \frac{1}{1 + e^{-z^{(1)}}}
$$

선형 변환 결과를 $(0, 1)$ 범위의 비선형 값으로 변환한다. 이 값이 다음 레이어의 입력이 된다.

**2층: Linear(256, 128)**

$$
z^{(2)} = a^{(1)} \cdot W^{(2)} + b^{(2)}, \quad W^{(2)} \in \mathbb{R}^{256 \times 128}
$$

**2층 Sigmoid 활성화**

$$
a^{(2)} = \sigma(z^{(2)})
$$

**3층 출력: Linear(128, output_dim)**

$$
\hat{y} = a^{(2)} \cdot W^{(3)} + b^{(3)}, \quad W^{(3)} \in \mathbb{R}^{128 \times \text{output\_dim}}
$$

출력 $\hat{y}$는 activation이 적용되지 않은 raw logit이다. losses.py의 손실 함수가 이후 단계에서 activation을 처리한다.

전체 forward를 수식으로 정리하면 다음과 같다.

$$
\hat{y} = \sigma\!\left(z^{(2)}\right) \cdot W^{(3)} + b^{(3)},
\quad z^{(2)} = \sigma\!\left(z^{(1)}\right) \cdot W^{(2)} + b^{(2)},
\quad z^{(1)} = x \cdot W^{(1)} + b^{(1)}
$$

### 2.5. Backward Pass와 역전파

Backward pass는 손실 함수의 gradient를 출력 레이어부터 입력 방향으로 역방향 전파하여 각 파라미터의 gradient를 계산하는 과정이다. Chain Rule(연쇄 법칙)을 재귀적으로 적용한다.

$$
\frac{\partial L}{\partial W^{(k)}} = \frac{\partial L}{\partial z^{(k)}} \cdot \frac{\partial z^{(k)}}{\partial W^{(k)}}
$$

**출발점: losses.py가 제공하는 gradient**

backward의 시작점은 `losses.py`의 `*_grad` 함수가 반환하는 `d(loss)/d(logits)`이다. 예를 들어 `cross_entropy_grad`는 다음을 반환한다.

$$
\frac{\partial L}{\partial \hat{y}} = \frac{\text{softmax}(\hat{y}) - y}{N}
$$

이 배열이 `model.backward(grad_out)`의 `grad_out` 인수로 전달된다.

**3층 Linear backward**

`grad_out`을 $\delta^{(3)} = \frac{\partial L}{\partial \hat{y}}$라 하면, `Linear` 레이어의 backward는 다음을 계산한다.

$$
\frac{\partial L}{\partial W^{(3)}} = (a^{(2)})^T \cdot \delta^{(3)}, \quad
\frac{\partial L}{\partial b^{(3)}} = \sum_i \delta^{(3)}_i, \quad
\frac{\partial L}{\partial a^{(2)}} = \delta^{(3)} \cdot (W^{(3)})^T
$$

계산된 $\frac{\partial L}{\partial W^{(3)}}$와 $\frac{\partial L}{\partial b^{(3)}}$는 `grad_w`, `grad_b`에 in-place로 저장한다. $\frac{\partial L}{\partial a^{(2)}}$는 2층 Sigmoid backward로 전달된다.

**2층 Sigmoid backward**

$$
\frac{\partial L}{\partial z^{(2)}} = \frac{\partial L}{\partial a^{(2)}} \cdot \sigma(z^{(2)}) \cdot (1 - \sigma(z^{(2)}))
$$

sigmoid의 미분 $\sigma'(z) = \sigma(z)(1 - \sigma(z))$를 이용한다. $\sigma(z^{(2)})$는 forward에서 `Sigmoid._out`으로 이미 저장되어 있으므로 재계산하지 않는다.

**2층 Linear backward**

$$
\frac{\partial L}{\partial W^{(2)}} = (a^{(1)})^T \cdot \frac{\partial L}{\partial z^{(2)}}, \quad
\frac{\partial L}{\partial a^{(1)}} = \frac{\partial L}{\partial z^{(2)}} \cdot (W^{(2)})^T
$$

**1층 Sigmoid backward**

$$
\frac{\partial L}{\partial z^{(1)}} = \frac{\partial L}{\partial a^{(1)}} \cdot \sigma(z^{(1)}) \cdot (1 - \sigma(z^{(1)}))
$$

**1층 Linear backward**

$$
\frac{\partial L}{\partial W^{(1)}} = x^T \cdot \frac{\partial L}{\partial z^{(1)}}, \quad
\frac{\partial L}{\partial b^{(1)}} = \sum_i \frac{\partial L}{\partial z^{(1)}_i}
$$

역전파가 완료되면 각 `Linear` 레이어의 `grad_w`와 `grad_b`에 gradient가 저장된다. optimizer는 이 값을 참조하여 W와 b를 업데이트한다.

### 2.6. params와 grads 설계

`MLP.params`와 `MLP.grads`는 내부 `Sequential.params`, `Sequential.grads`를 그대로 노출하는 property이다. `Sequential` 생성자에서 하위 `Linear` 레이어들의 `params`와 `grads`를 `extend`로 수집한다.

```text
MLP.params = Sequential.params
           = [W1, b1, W2, b2, W3, b3]
               ^              ^
               Linear(784,256) Linear(128,output_dim)
```

`params[0]`은 첫 번째 `Linear`의 `w`, `params[1]`은 첫 번째 `Linear`의 `b`이다. 이 순서로 6개 배열이 나열된다. `grads`도 같은 순서로 대응한다.

중요한 설계 포인트는 `grads` 리스트의 배열이 `params` 배열과 동일한 객체를 참조한다는 점이다. `Linear.backward`에서 `grad_w[...] = ...` 형식의 in-place 대입을 사용하기 때문에, optimizer가 `grads[i]`를 참조하면 항상 최신 gradient를 읽을 수 있다.

```text
params[0] = W1 (array object A)     grads[0] = grad_w1 (array object B)
                                         |
                                    backward에서 B[...] = 새로운 gradient 값
                                    (B 자체는 교체되지 않으므로 참조 유지)
```

optimizer는 다음과 같이 동작한다.

```text
for param, grad in zip(model.params, model.grads):
    param -= lr * grad   # in-place 업데이트
```

### 2.7. 파라미터 초기화와 seed 파생

각 `Linear` 레이어는 He 초기화로 가중치를 설정한다.

$$
W \sim \mathcal{N}\!\left(0,\ \sqrt{\frac{2}{D_{in}}}\right)
$$

`MLP`는 최상위 seed 하나를 받아 `np.random.default_rng(seed).integers(0, 2**31, size=3)`으로 3개의 레이어 seed를 파생한다. 각 `Linear`에 독립적인 seed를 전달하므로, 최상위 seed 하나만 고정하면 MLP 전체의 초기화 재현성이 보장된다.

seed를 지정하지 않으면(`seed=None`) numpy 기본 난수 생성기를 사용하여 실행마다 다른 초기값을 갖는다.

### 2.8. logit 입력 설계의 의미

MLP는 activation 없이 raw logit을 반환한다. 이 설계가 의미하는 바는 다음과 같다.

**수치 안정성**: `softmax(logit)` 후 `log(prob)`를 계산하면 중간에 매우 작은 값이 생겨 `log(0)` 문제가 발생할 수 있다. `cross_entropy`가 logit을 직접 받아 `softmax`와 `log`를 결합하여 처리하면 내부에서 epsilon clipping 등의 안정화 처리가 가능하다.

**gradient 단순화**: `softmax` + `cross_entropy`를 결합하면 gradient가 `softmax(logit) - target`으로 단순화된다. `sigmoid` + `binary_cross_entropy`도 마찬가지로 `sigmoid(logit) - target`이 된다. 두 함수를 분리하면 중간 미분값이 복잡해지지만, 결합하면 `(예측 - 정답) / N` 형태로 직관적으로 이해할 수 있다.

**인터페이스 통일**: PyTorch의 `nn.CrossEntropyLoss`도 logit 입력 방식을 사용한다. 이 프로젝트의 설계는 후속 PyTorch 프로젝트와 동일한 인터페이스를 유지하기 위한 설계이다.

### 2.9. 학습 한 스텝의 전체 흐름

MLP를 이용한 학습 한 스텝의 전체 흐름은 다음과 같다.

```text
[1] images, targets = next(train_loader)
      images: (N, 784) float32
      targets: (N, 10) one-hot (multiclass 예시)

[2] logits = model.forward(images)
      Sequential 내부에서 레이어를 순서대로 통과
      출력: (N, 10) raw logit

[3] loss = cross_entropy(logits, targets)
      내부에서 softmax 적용
      출력: scalar float

[4] grad = cross_entropy_grad(logits, targets)
      내부에서 softmax 적용
      출력: (N, 10) d(loss)/d(logits)

[5] model.backward(grad)
      Sequential 내부에서 레이어를 역순으로 통과
      각 Linear의 grad_w, grad_b에 gradient 저장

[6] optimizer.step()
      model.params와 model.grads를 순회하며 in-place 업데이트
      W -= lr * grad_w, b -= lr * grad_b
```

이 6단계가 반복되며 파라미터가 손실을 줄이는 방향으로 수렴한다.

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `MLP` | 클래스 | `task: str`, `seed: int` | model instance | 3층 MLP |
| `forward` | 메서드 | `x (N, 784)` float32 | `(N, output_dim)` float32 | raw logit 반환 |
| `backward` | 메서드 | `grad_out (N, output_dim)` | 없음 | backpropagation 수행 |
| `params` | property | - | list of ndarray | optimizer가 업데이트할 파라미터 리스트 |
| `grads` | property | - | list of ndarray | params와 대응하는 gradient 리스트 |

### 3.1. MLP 생성자

```python
class MLP:
    def __init__(self, task="multiclass", seed=None):
        spec = get_task_spec(task)
        self.task = task
        self.output_dim = spec["output_dim"]

        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 2**31, size=3)

        self.net = Sequential(
            Linear(784, 256, seed=int(seeds[0])),
            Sigmoid(),
            Linear(256, 128, seed=int(seeds[1])),
            Sigmoid(),
            Linear(128, self.output_dim, seed=int(seeds[2])),
        )
```

`np.random.default_rng(seed).integers(..., size=3)`으로 3개의 레이어 seed를 파생하여 각 `Linear`에 전달한다. 최상위 seed 하나로 모든 레이어 초기화의 재현성을 보장한다.

### 3.2. forward와 backward

```python
def forward(self, x):
    return self.net(x)

def backward(self, grad_out):
    return self.net.backward(grad_out)
```

`forward`는 `Sequential.__call__`을 통해 레이어를 순서대로 통과한다. `backward`는 `src/nn/losses.py`의 `_grad` 함수가 반환한 `d(loss)/d(logits)` 배열을 받아 역방향으로 전파한다.

### 3.3. params와 grads property

```python
@property
def params(self):
    return self.net.params

@property
def grads(self):
    return self.net.grads
```

`net.params`는 `Sequential` 생성 시 모든 `Linear` 레이어의 `w`, `b`가 등록된 리스트이다. 순서는 `[W1, b1, W2, b2, W3, b3]`이며 총 6개이다. `grads`도 동일한 순서로 대응한다.

## 4. 사용법

최소 사용 예제는 다음과 같다.

```python
import numpy as np
from src.models.mlp import MLP

model = MLP(task="multiclass", seed=42)
x = np.random.randn(32, 784).astype(np.float32)
logits = model.forward(x)

print(logits.shape)
print(len(model.params))
```

예상 출력은 다음과 같다.

```text
(32, 10)
6
```

프로젝트 통합 예제는 다음과 같다. Trainer의 학습 루프에서 forward, backward, optimizer.step 흐름을 따른다.

```python
from src.models.mlp import MLP
from src.nn.losses import cross_entropy, cross_entropy_grad
from src.core.optimizers import SGD

model = MLP(task="multiclass", seed=42)
optimizer = SGD(model, lr=0.01)

for images, targets in train_loader:
    logits = model.forward(images)
    loss = cross_entropy(logits, targets)
    grad = cross_entropy_grad(logits, targets)
    model.backward(grad)
    optimizer.step()
```

## 5. 테스트

테스트 파일은 `tests/stage4/test_mlp.py`이다.

```bash
conda run -n numpy_py311 pytest tests/stage4/test_mlp.py -v
```

테스트 구성은 다음과 같다.

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestMLPForward` | 4 | task별 logit shape, dtype float32 |
| `TestMLPParams` | 3 | params 수 6개, grads 수 6개, shape 일치 |
| `TestMLPBackward` | 3 | backward 후 grads 비 0 확인, params 변경 없음 확인 |
| `TestMLPSeed` | 2 | 동일 seed 동일 초기값, 다른 seed 다른 초기값 |

## 6. 요약

`MLP`는 `Sequential(Linear, Sigmoid, Linear, Sigmoid, Linear)` 구조로 조립된 3층 완전 연결 네트워크이다. forward는 raw logit을 반환하며, activation과 gradient 계산은 `losses.py`가 담당하는 logit 입력 설계를 유지한다. backward는 losses.py가 제공하는 `d(loss)/d(logits)`를 출발점으로 각 레이어를 역순으로 순회하며 파라미터 gradient를 계산한다. `params`/`grads` property로 optimizer가 단일 진입점으로 모든 파라미터를 업데이트할 수 있다.

다음 Phase에서는 [[phase4.2_cnn]]을 다룬다.
