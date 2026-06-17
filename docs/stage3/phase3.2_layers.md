---
tags: [stage3, nn, layers]
created: 2026-06-17
updated: 2026-06-17
---

# Phase 3.2 layer module 구현

## 1. 역할

`src/nn/layers.py`는 NumPy 기반 레이어 모듈을 구현한다.
PyTorch의 `torch.nn`에 대응하는 객체 인터페이스로, `MLP` 등 모델 클래스가 조립하여 사용한다.

## 2. 구현

### 2.1. Module (베이스 클래스)

모든 레이어의 공통 인터페이스를 정의한다.

| 속성/메서드 | 역할 |
|---|---|
| `params: list` | 학습 가능한 파라미터 배열 목록 |
| `grads: list` | `params`에 대응하는 gradient 배열 목록 |
| `__call__(x)` | `forward(x)` 위임 |
| `forward(x)` | 서브클래스에서 구현 |
| `backward(dout)` | 서브클래스에서 구현 |

### 2.2. Linear(in_features, out_features, seed=None)

완전연결 레이어를 구현한다. 가중치 초기화, forward, backward 모두 포함한다.

가중치 초기화 규약은 다음과 같다.

- `w`: He 초기화 - `std = sqrt(2 / in_features)`, shape `(in_features, out_features)`, `float32`
- `b`: 0으로 초기화, shape `(out_features,)`, `float32`
- `seed` 인자로 재현성을 보장한다.

`backward(dout)` 동작은 다음과 같다.

- `grad_w[...] = x.T @ dout` (in-place 갱신)
- `grad_b[...] = dout.sum(axis=0)` (in-place 갱신)
- `dout @ w.T`를 상위 레이어로 전달할 gradient로 반환

`params = [w, b]`, `grads = [grad_w, grad_b]`이며 `Sequential`이 이를 참조로 수집한다.

### 2.3. Sigmoid

순전파 출력을 `_out`에 캐싱하여 역전파에서 재사용한다.

- `forward(x)` : `sigmoid(x)` 반환, 결과를 `_out`에 저장
- `backward(dout)` : `dout * _out * (1 - _out)` 반환
- `params = []`, `grads = []` (학습 파라미터 없음)

### 2.4. ReLU

순전파에서 양수 위치 마스크를 `_mask`에 저장하여 역전파에 사용한다.

- `forward(x)` : `x * (x > 0)` 반환, 마스크를 `_mask`에 저장
- `backward(dout)` : `dout * _mask` 반환

### 2.5. Sequential(*layers)

여러 레이어를 순서대로 조합하는 컨테이너 모듈이다.

- `forward(x)` : 레이어 순서대로 `x`를 통과시킨다.
- `backward(dout)` : 레이어 역순으로 `dout`을 전파한다.
- `params` / `grads` : 생성 시 각 레이어의 `params`, `grads` 원소를 순서대로 `extend`하여 수집한다. 각 원소는 복사본이 아닌 레이어 내부 배열의 참조이다.

### 2.6. 인터페이스

```python
from src.nn.layers import Linear, Sigmoid, ReLU, Sequential

net = Sequential(
    Linear(784, 256, seed=0),
    Sigmoid(),
    Linear(256, 10, seed=1),
)

x = ...               # (N, 784) float32
logits = net(x)       # (N, 10) float32

grad_out = ...        # (N, 10) float32 - losses.*_grad 에서 계산
net.backward(grad_out)

# 파라미터 업데이트 (SGD 예시)
for param, grad in zip(net.params, net.grads):
    param -= lr * grad
```

## 3. 테스트

테스트 파일: `tests/stage3/test_layers.py`

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestLinear` | 7 | forward shape, params/grads 개수, weight shape, backward grad shape, grads in-place 갱신, seed 재현성, He 초기화 표준편차 |
| `TestSigmoid` | 5 | forward shape/범위, backward shape/수치값(x=0에서 0.25), params 없음 |
| `TestReLU` | 2 | 음수 → 0, backward 마스크 적용 |
| `TestSequential` | 4 | forward shape, params 집계 수, backward grad shape, params가 레이어 내부 참조 |

실행 명령:

```bash
conda run -n numpy_env pytest tests/stage3/test_layers.py -v
```

## 4. 설계 결정

- `backward`에서 `grad_w[...] =` 형태로 in-place 갱신한다. `grad_w = ...`로 재할당하면 `Sequential.grads`가 보유한 참조가 무효화된다.
- `Sequential` 생성 시 `params`를 참조로 수집하므로, 각 레이어의 `backward` 이후 `Sequential.grads` 원소가 자동으로 최신 gradient를 반영한다. 옵티마이저는 `Sequential.grads`만 참조하면 된다.
- `Sigmoid`와 `ReLU`는 `Module.__init__`을 호출하지 않아도 `params = []`, `grads = []`가 `Module.__init__`에서 설정된다.
