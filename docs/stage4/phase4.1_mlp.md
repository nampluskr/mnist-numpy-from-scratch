---
tags: [docs, stage4, models, mlp]
created: "2026-06-20"
updated: "2026-06-20"
---

# MLP 모델

## 1. 개요

`src/models/mlp.py`의 `MLP`는 `src/nn/` 레이어 모듈을 조립하여 구성한 3층 완전 연결 네트워크이다. `forward`는 raw logit을 반환하고, activation과 gradient 계산은 `src/nn/losses.py`가 처리한다. task에 따라 출력 차원이 달라지며, `MLP.params`와 `MLP.grads`를 통해 optimizer와 연동한다. CPU 기반 numpy 구현이며, 후속 PyTorch, TensorFlow, JAX 프로젝트와 동일한 public interface를 유지한다.

**목표**
- `src/nn/Sequential`로 `Linear` + `Sigmoid` 레이어를 조립하여 3층 MLP를 구성한다.
- task에 따라 출력 차원을 자동으로 설정한다.
- `params`와 `grads` property로 optimizer와 연동하는 인터페이스를 제공한다.

## 2. 개념

### 2.1. MLP 구조

이 프로젝트의 MLP는 `784 -> 256 -> 128 -> output_dim` 구조를 사용한다. 각 hidden layer는 `Linear` + `Sigmoid` 활성화로 구성되며, 출력 레이어는 activation 없이 raw logit을 반환한다.

```text
Linear(784, 256) -> Sigmoid -> Linear(256, 128) -> Sigmoid -> Linear(128, output_dim)
```

task별 출력 차원은 다음과 같다.

| task | output_dim | 출력 해석 |
|---|---|---|
| `multiclass` | 10 | 클래스별 raw score, `cross_entropy`가 softmax 적용 |
| `binary` | 1 | 이진 raw score, `binary_cross_entropy`가 sigmoid 적용 |
| `regression` | 1 | 예측값 그대로 사용, `mse`와 연동 |

### 2.2. params와 grads

`MLP.params`와 `MLP.grads`는 내부 `Sequential.params`, `Sequential.grads`를 그대로 노출하는 property이다. `Sequential` 생성 시 모든 `Linear` 레이어의 `w`, `b` 배열이 `params` 리스트에 등록된다. optimizer는 이 리스트를 순회하며 in-place 업데이트를 수행한다.

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

`np.random.default_rng(seed).integers(..., size=3)`로 3개의 레이어 seed를 파생하여 각 `Linear`에 전달한다. 최상위 seed 하나로 모든 레이어 초기화의 재현성을 보장한다.

### 3.2. forward와 backward

```python
def forward(self, x):
    return self.net(x)

def backward(self, grad_out):
    return self.net.backward(grad_out)
```

`forward`는 `Sequential.__call__`을 통해 레이어를 순서대로 통과한다. `backward`는 `src/nn/losses.py`의 `_grad` 함수가 반환한 `d(loss)/d(logits)` 배열을 받아 역방향으로 전파한다.

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

`MLP`는 `Sequential(Linear, Sigmoid, Linear, Sigmoid, Linear)` 구조로 조립된 3층 완전 연결 네트워크이다. task에 따라 출력 차원이 자동으로 설정되고, `params`/`grads` property로 optimizer와 연동한다. `forward`는 raw logit을 반환하며, activation과 gradient 계산은 `losses.py`가 담당하는 logit 입력 설계를 유지한다.

다음 Phase에서는 [[phase4.2_cnn]]을 다룬다.
