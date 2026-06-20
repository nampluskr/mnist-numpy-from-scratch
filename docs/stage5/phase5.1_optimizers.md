---
tags: [docs, stage5, optimizers]
created: "2026-06-20"
updated: "2026-06-21"
---

# Optimizer 구현

## 1. 개요

`src/core/optimizers.py`는 `SGD`와 `Adam` 두 가지 optimizer를 제공한다. optimizer는 `model.params`와 `model.grads` 리스트를 받아 in-place로 파라미터를 업데이트한다. `Trainer.fit` 내부에서 매 batch마다 `optimizer.step()`을 호출하는 방식으로 동작하며, 모델과 optimizer가 분리된 구조를 유지한다.

**목표**
- `SGD`와 `Adam` optimizer를 `model.params`/`model.grads` 인터페이스에 맞춰 구현한다.
- 파라미터 in-place 업데이트를 수행하고 반환값은 없다.
- `Adam`은 momentum과 adaptive learning rate를 결합하여 수렴 안정성을 높인다.

## 2. 개념

### 2.1. Gradient Descent 원리

신경망 학습은 loss $L$을 최소화하는 파라미터 $\theta$를 찾는 최적화 문제이다. $L$은 파라미터의 함수이므로, $\theta$ 공간에서 $L$이 감소하는 방향으로 $\theta$를 반복적으로 이동하면 최솟값에 가까워진다.

$$
\theta \leftarrow \theta - \eta \cdot \nabla_\theta L
$$

$\nabla_\theta L = \frac{\partial L}{\partial \theta}$는 $\theta$에 대한 loss의 편미분이다. gradient는 loss가 가장 빠르게 증가하는 방향을 가리키므로, 그 반대 방향으로 $\eta$만큼 이동하면 loss가 감소한다.

**Full batch vs Mini-batch**: 전체 데이터셋 gradient를 정확히 계산하는 것은 비용이 크다. mini-batch gradient는 전체 gradient의 불편 추정량(unbiased estimator)이므로, 작은 배치로 계산해도 기대값 기준으로 같은 방향을 가리킨다. 이 프로젝트의 `Trainer`는 mini-batch 단위로 gradient를 계산하고 즉시 `optimizer.step()`을 호출한다.

### 2.2. SGD

SGD(Stochastic Gradient Descent)는 가장 기본적인 optimizer이다. 현재 gradient 방향으로 learning rate만큼 파라미터를 이동한다.

$$
\theta_t \leftarrow \theta_{t-1} - \eta \cdot g_t
$$

$\theta_t$는 스텝 $t$ 이후의 파라미터, $\eta$는 learning rate, $g_t = \nabla_{\theta} L$는 스텝 $t$에서의 gradient이다.

**Learning rate의 영향**: learning rate $\eta$는 파라미터 이동 보폭을 결정한다.

| $\eta$ 설정 | 결과 |
|---|---|
| 너무 크다 | gradient 방향으로 과도하게 이동 → loss가 발산할 수 있다 |
| 적절하다 | loss가 안정적으로 감소한다 |
| 너무 작다 | 이동 보폭이 미미하여 수렴이 매우 느리다 |

**In-place update**: `param -= lr * grad`는 배열의 내용을 직접 수정한다. `param = param - lr * grad`는 새 배열을 생성하여 `model.params` 리스트의 참조가 끊어지므로 반드시 in-place 연산을 사용해야 한다.

핵심 용어는 다음과 같다.

| 용어 | 의미 | 이 프로젝트에서의 역할 |
|---|---|---|
| learning rate $\eta$ | 파라미터 업데이트 보폭 | `SGD(model, lr=0.01)` 형태로 설정 |
| gradient $g_t$ | loss를 파라미터로 미분한 값 | `model.grads`에서 참조 |
| in-place update | 기존 배열을 직접 수정 | `param -= lr * grad` 형태로 적용 |

### 2.3. Adam

SGD는 모든 파라미터에 동일한 learning rate를 적용한다. 그러나 파라미터마다 gradient 크기가 다르면 단일 learning rate가 모든 파라미터에 적합하지 않다. Adam(Adaptive Moment Estimation)은 gradient의 1차 moment(이동 평균)와 2차 moment(제곱 이동 평균)를 유지하여 파라미터별로 effective learning rate를 자동으로 조절한다.

**1차 moment — gradient 이동 평균**

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$m_t$는 과거 gradient의 지수 이동 평균(Exponential Moving Average, EMA)이다. $\beta_1$이 클수록 과거 gradient의 영향이 오래 유지된다. $\beta_1 = 0.9$이면 약 10 스텝의 gradient를 평균 낸 효과가 있다. 이는 gradient의 방향이 일관되면 가속하고, 진동하면 상쇄되는 momentum 효과를 낸다.

**2차 moment — gradient 제곱 이동 평균**

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$v_t$는 gradient 제곱의 EMA로, gradient의 분산 척도이다. gradient가 자주 크게 나타나는 파라미터는 $v_t$가 크고, 자주 작게 나타나는 파라미터는 $v_t$가 작다. $\sqrt{v_t}$로 나누어 effective learning rate를 조절하면, gradient가 큰 파라미터는 step이 작아지고 gradient가 작은 파라미터는 step이 커진다.

**Bias correction**

$m_0 = 0$, $v_0 = 0$으로 초기화하면 초기 스텝에서 $m_t$와 $v_t$가 0 쪽으로 편향된다. $t = 1$일 때 $m_1 = (1 - \beta_1) g_1$으로, 실제 gradient보다 $(1 - \beta_1)$ 배 작다. Bias correction은 이 편향을 제거한다.

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$t$가 커질수록 $\beta_1^t \to 0$이므로 보정 계수 $\frac{1}{1 - \beta_1^t} \to 1$이 되어 편향이 사라진다. $t = 1$에서 $\frac{1}{1 - 0.9} = 10$이고, $t = 100$에서는 거의 1에 가깝다.

**파라미터 업데이트**

$$
\theta_t \leftarrow \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

$\epsilon$은 $\sqrt{\hat{v}_t}$가 0에 가까울 때 분모가 0이 되는 것을 방지하는 수치 안정성 상수이다(기본값 $10^{-8}$). 전체 업데이트를 풀어 쓰면 다음과 같다.

$$
\theta_t \leftarrow \theta_{t-1} - \eta \cdot \frac{m_t / (1 - \beta_1^t)}{\sqrt{v_t / (1 - \beta_2^t)} + \epsilon}
$$

**SGD와의 비교**

| 항목 | SGD | Adam |
|---|---|---|
| 업데이트 방향 | 현재 gradient $g_t$ | 1차 moment $\hat{m}_t$ (과거 gradient 포함) |
| 업데이트 크기 | 모든 파라미터 동일 $\eta$ | 파라미터별 $\eta / \sqrt{\hat{v}_t}$ |
| 추가 메모리 | 없음 | 파라미터당 $m$, $v$ 두 배열 |
| 하이퍼파라미터 | $\eta$ | $\eta$, $\beta_1$, $\beta_2$, $\epsilon$ |
| 수렴 안정성 | learning rate에 민감 | 넓은 learning rate 범위에서 안정적 |

핵심 용어는 다음과 같다.

| 용어 | 의미 | 이 프로젝트에서의 역할 |
|---|---|---|
| $\beta_1$ | 1차 moment 감쇠 계수 | 기본값 0.9. 클수록 과거 gradient를 오래 기억 |
| $\beta_2$ | 2차 moment 감쇠 계수 | 기본값 0.999. 클수록 분산 추정이 느리게 변화 |
| $\epsilon$ | 수치 안정성 상수 | 기본값 1e-8. 분모 0 방지 |
| bias correction | 초기 0 편향 보정 | iter(스텝 수)를 추적하여 분모에 적용 |
| in-place update | 기존 배열 직접 수정 | `m[...] =`, `param[...] -=` 형태로 적용 |

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
