---
tags: [stage5, core, optimizers]
created: 2026-06-17
updated: 2026-06-20
---

# Phase 5.1 optimizer 구현

## 1. 역할

`src/core/optimizers.py`는 SGD와 Adam 두 가지 옵티마이저를 구현한다.
`model.params`와 `model.grads`를 참조하여 파라미터를 in-place로 업데이트한다.

## 2. 구현

### 2.1. SGD(model, lr)

가장 단순한 경사하강법이다. 매 `step()` 호출마다 아래 규칙으로 파라미터를 갱신한다.

```text
param ← param - lr × grad
```

| 인자 | 설명 |
|---|---|
| `model` | `params`, `grads` 속성을 가진 모델 인스턴스 |
| `lr` | 학습률 |

### 2.2. Adam(model, lr, beta1=0.9, beta2=0.999, eps=1e-8)

적응형 모멘텀 기반 옵티마이저이다. 1차·2차 모멘트를 추적하고 bias correction을 적용하여 파라미터를 갱신한다.

```text
m ← β1 × m + (1 - β1) × grad
v ← β2 × v + (1 - β2) × grad²
m̂ = m / (1 - β1^t)
v̂ = v / (1 - β2^t)
param ← param - lr × m̂ / (√v̂ + ε)
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `model` | | `params`, `grads` 속성을 가진 모델 인스턴스 |
| `lr` | | 학습률 |
| `beta1` | `0.9` | 1차 모멘트 감쇠율 |
| `beta2` | `0.999` | 2차 모멘트 감쇠율 |
| `eps` | `1e-8` | 분모 안정화 상수 |

생성 시 `ms`, `vs`를 `params`와 동일한 shape의 0 배열로 초기화한다.
`iter`는 `step()` 호출마다 1씩 증가하며 bias correction 지수로 사용한다.

### 2.3. 인터페이스

```python
from src.core.optimizers import SGD, Adam
from src.models.mlp import MLP

model = MLP(task="multiclass", seed=0)

# SGD
opt = SGD(model, lr=0.01)
opt.step()

# Adam
opt = Adam(model, lr=0.001)
opt.step()
```

## 3. 테스트

테스트 파일: `tests/stage5/test_optimizers.py`

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestSGD` | 5 | 정확한 수치 업데이트, in-place 갱신, step() 반환값 None, lr=0 불변, 누적 2회 step |
| `TestAdam` | 7 | 파라미터 변경 확인, in-place 갱신, step() 반환값 None, iter 증가, m/v shape, 음의 방향 이동, 기본 인자값 |

실행 명령:

```bash
conda run -n numpy_env pytest tests/stage5/test_optimizers.py -v
```

## 4. 설계 결정

- `param -= self.lr * grad` (SGD)와 `param[...] -= ...` (Adam) 방식을 구분하여 사용한다. 두 방식 모두 numpy 배열을 in-place로 갱신하지만, Adam에서는 `m[...] =` 형태로 모멘트 버퍼를 갱신해야 하므로 일관성 확보를 위해 `param[...] -=`을 사용한다.
- 옵티마이저는 생성 시점에 `model.params`와 `model.grads`를 참조로 저장한다. `MLP` → `Sequential` → `Linear`의 참조 체인이 그대로 유지되므로 `backward` 이후 `grads`가 자동으로 최신 gradient를 반영한다.
- `ms`, `vs`는 `params`의 복사본이 아닌 동일 shape의 독립 배열로 초기화한다. 모멘트 버퍼는 파라미터와 별개로 관리되어야 한다.
