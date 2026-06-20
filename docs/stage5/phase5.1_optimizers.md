---
tags: [docs, stage5, optimizers]
created: "2026-06-20"
updated: "2026-06-20"
---

# Optimizer 구현

## 1. 개요

`src/core/optimizers.py`는 `SGD`와 `Adam` 두 가지 optimizer를 제공한다. optimizer는 `model.params`와 `model.grads` 리스트를 받아 in-place로 파라미터를 업데이트한다. `Trainer.fit` 내부에서 매 batch마다 `optimizer.step()`을 호출하는 방식으로 동작하며, 모델과 optimizer가 분리된 구조를 유지한다.

**목표**
- `SGD`와 `Adam` optimizer를 `model.params`/`model.grads` 인터페이스에 맞춰 구현한다.
- 파라미터 in-place 업데이트를 수행하고 반환값은 없다.
- `Adam`은 momentum과 adaptive learning rate를 결합하여 수렴 안정성을 높인다.

## 2. 개념

### 2.1. SGD

SGD(Stochastic Gradient Descent)는 가장 기본적인 optimizer이다. 현재 gradient 방향으로 learning rate만큼 파라미터를 이동한다.

$$
\theta \leftarrow \theta - \eta \cdot \nabla_\theta L
$$

$\theta$는 업데이트할 파라미터, $\eta$는 learning rate, $\nabla_\theta L$는 loss에 대한 gradient이다. gradient가 크면 파라미터 이동도 크고, gradient가 작으면 이동이 작다. 단순하지만 learning rate 설정에 민감하여 너무 크면 발산하고 너무 작으면 수렴이 느리다.

핵심 용어는 다음과 같다.

| 용어 | 의미 | 이 프로젝트에서의 역할 |
|---|---|---|
| learning rate | 파라미터 업데이트 보폭 | `SGD(model, lr=0.01)` 형태로 설정 |
| gradient | loss를 파라미터로 미분한 값 | `model.grads`에서 참조 |
| in-place update | 기존 배열을 직접 수정 | `param -= lr * grad` 형태로 적용 |

### 2.2. Adam

Adam(Adaptive Moment Estimation)은 gradient의 1차 moment(평균)와 2차 moment(분산)를 추적하여 파라미터별로 learning rate를 자동으로 조절한다.

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$
$$
\theta \leftarrow \theta - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

$m_t$는 gradient의 지수 이동 평균(1차 moment), $v_t$는 gradient 제곱의 지수 이동 평균(2차 moment)이다. $\hat{m}_t$와 $\hat{v}_t$는 초기 단계에서 0으로 편향된 값을 보정하는 bias correction이다. $\epsilon$은 분모가 0이 되는 것을 방지하는 수치 안정성 상수이다.

Adam은 gradient가 큰 파라미터에는 작은 step을 적용하고, gradient가 작은 파라미터에는 큰 step을 적용하여 파라미터별 학습 속도를 자동으로 조절한다.

| 용어 | 의미 | 이 프로젝트에서의 역할 |
|---|---|---|
| `beta1` | 1차 moment 감쇠 계수 | 기본값 0.9 |
| `beta2` | 2차 moment 감쇠 계수 | 기본값 0.999 |
| `epsilon` | 수치 안정성 상수 | 기본값 1e-8 |
| bias correction | 초기 편향 보정 | `t` 스텝 수를 추적하여 적용 |

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `SGD` | 클래스 | `model`, `lr: float` | optimizer instance | 단순 gradient descent |
| `Adam` | 클래스 | `model`, `lr: float`, `beta1: float`, `beta2: float`, `eps: float` | optimizer instance | adaptive learning rate optimizer |
| `step` | 메서드 | 없음 | 없음 | `model.params`를 in-place 업데이트 |

### 3.1. SGD 구현

```python
class SGD:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def step(self):
        for param, grad in zip(self.model.params, self.model.grads):
            param -= self.lr * grad
```

`param -= self.lr * grad`는 NumPy 배열의 in-place 연산이다. `param = param - self.lr * grad`와 달리 새 배열을 생성하지 않으므로, `model.params` 리스트가 참조하는 배열 객체가 유지된다.

### 3.2. Adam 구현

```python
class Adam:
    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in model.params]
        self.v = [np.zeros_like(p) for p in model.params]

    def step(self):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        for i, (param, grad) in enumerate(zip(self.model.params, self.model.grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            param -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)
```

`self.m`과 `self.v`는 `model.params`와 같은 shape의 0 배열로 초기화한다. `self.t`를 스텝마다 증가시켜 bias correction을 적용한다. bias correction을 `lr_t`에 통합하여 루프 내 연산을 줄인다.

## 4. 사용법

최소 사용 예제는 다음과 같다.

```python
import numpy as np
from src.models.mlp import MLP
from src.core.optimizers import SGD, Adam

model = MLP(task="multiclass", seed=42)
optimizer = SGD(model, lr=0.01)

x = np.random.randn(32, 784).astype(np.float32)
logits = model.forward(x)
grad_out = np.ones_like(logits) / 32
model.backward(grad_out)
optimizer.step()

print(len(model.params))
```

예상 출력은 다음과 같다.

```text
6
```

프로젝트 통합 예제는 다음과 같다. `Trainer.fit` 내부에서 batch마다 호출되는 흐름이다.

```python
from src.models.mlp import MLP
from src.core.optimizers import Adam
from src.nn.losses import cross_entropy, cross_entropy_grad

model = MLP(task="multiclass", seed=42)
optimizer = Adam(model, lr=0.001)

for images, targets in train_loader:
    logits = model.forward(images)
    loss = cross_entropy(logits, targets)
    grad = cross_entropy_grad(logits, targets)
    model.backward(grad)
    optimizer.step()
```

## 5. 테스트

테스트 파일은 `tests/stage5/test_optimizers.py`이다.

```bash
conda run -n numpy_py311 pytest tests/stage5/test_optimizers.py -v
```

테스트 구성은 다음과 같다.

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestSGD` | 3 | step 후 params 변경, lr=0 시 params 불변, in-place 업데이트 확인 |
| `TestAdam` | 4 | step 후 params 변경, t 증가 확인, m/v 초기화 shape 일치, lr=0 수렴 후 step 안정성 |

## 6. 요약

`SGD`는 gradient에 learning rate를 곱해 파라미터를 직접 이동하고, `Adam`은 1차/2차 moment와 bias correction을 적용하여 파라미터별 adaptive learning rate를 계산한다. 두 optimizer 모두 `model.params`/`model.grads` 인터페이스를 통해 in-place 업데이트를 수행하며 반환값이 없다.

다음 Phase에서는 [[phase5.2_trainer-evaluator]]을 다룬다.
