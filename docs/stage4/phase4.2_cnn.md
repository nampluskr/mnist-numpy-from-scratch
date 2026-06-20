---
tags: [docs, stage4, models, cnn, cupy]
created: "2026-06-20"
updated: "2026-06-20"
---

# CNN 모델

## 1. 개요

`src/models/cnn.py`의 `CNN`은 CuPy 기반 2층 합성곱 네트워크이다. `MLP`와 동일한 public interface(`forward`, `backward`, `params`, `grads`)를 유지하여 Trainer, Evaluator, Predictor가 모델 종류에 관계없이 같은 방식으로 동작한다. Conv 경로는 CuPy 배열로 GPU에서 실행하고, `Flatten` 이후 FC 경로는 numpy 배열로 CPU에서 실행한다. CuPy가 없는 환경에서는 자동으로 numpy fallback이 적용된다.

**목표**
- `Conv2d`, `MaxPool2d`, `Flatten`, `Linear`, `ReLU`, `Dropout`을 조립하여 CNN을 구성한다.
- `forward`에서 `(N, 784)` numpy 입력을 받아 `(N, output_dim)` numpy logit을 반환한다.
- CuPy/numpy 경계를 `forward`/`backward` 내부에서 명시적으로 처리한다.

## 2. 개념

### 2.1. CNN 구조

이 프로젝트의 CNN은 두 개의 Conv-ReLU-Pool 블록과 FC 블록으로 구성된다.

```text
Conv2d(1, 32, 3, pad=1) -> ReLU -> MaxPool2d(2, 2)    [28x28 -> 14x14]
Conv2d(32, 64, 3, pad=1) -> ReLU -> MaxPool2d(2, 2)   [14x14 -> 7x7]
Flatten                                                 [(64, 7, 7) -> 3136]
Linear(3136, 256) -> ReLU -> Dropout(0.5)
Linear(256, output_dim)
```

MNIST 이미지는 `(N, 784)` 형태로 입력되므로 `forward` 시작 시 `(N, 1, 28, 28)`로 reshape한다.

### 2.2. CuPy/numpy 경계

Conv 경로와 FC 경로의 배열 모듈이 다르므로 `Flatten` 이후 명시적으로 변환한다.

| 경로 | 배열 모듈 | 변환 시점 |
|---|---|---|
| Conv 경로 (`conv_net`, `flatten`) | CuPy (또는 numpy fallback) | - |
| numpy 변환 | - | `Flatten` 출력 후 `.get()` 호출 |
| FC 경로 (`dropout`, `fc_net`) | numpy | - |

backward에서는 역방향으로 `fc_net` -> `dropout` -> numpy->CuPy 변환 -> `flatten` -> `conv_net` 순으로 전파한다.

### 2.3. CuPy fallback

파일 상단에서 `import cupy`를 시도하고, 실패하면 `numpy`를 `_xp`로 사용한다.

```python
try:
    import cupy as _xp
except ImportError:
    import numpy as _xp
```

이 패턴으로 GPU가 없는 환경에서도 CPU로 동일한 코드를 실행할 수 있다. 성능은 낮지만 기능 검증이 가능하다.

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `CNN` | 클래스 | `task: str`, `seed: int` | model instance | CuPy 기반 CNN |
| `forward` | 메서드 | `x (N, 784)` numpy float32 | `(N, output_dim)` numpy float32 | raw logit 반환 |
| `backward` | 메서드 | `grad_out (N, output_dim)` numpy | 없음 | backpropagation 수행 |
| `params` | property | - | list of ndarray | Conv + FC 파라미터 리스트 |
| `grads` | property | - | list of ndarray | params와 대응하는 gradient 리스트 |
| `train()` / `eval()` | 메서드 | - | 없음 | Dropout 등 학습/평가 모드 전환 |

### 3.1. forward

```python
def forward(self, x):
    x_xp = self._xp.asarray(x).reshape(-1, 1, 28, 28)
    x_xp = self.conv_net(x_xp)
    x_xp = self.flatten(x_xp)
    x_np = x_xp.get() if hasattr(x_xp, "get") else np.asarray(x_xp)
    x_np = self.dropout(x_np)
    return self.fc_net(x_np)
```

`self._xp.asarray(x)`는 numpy 배열을 CuPy 배열로 변환한다. `.get()`은 CuPy 배열을 numpy로 복사하는 CuPy 전용 메서드이며, numpy fallback 환경에서는 `np.asarray`로 대체한다. `hasattr(x_xp, "get")`으로 두 경우를 분기한다.

### 3.2. backward

```python
def backward(self, grad_out):
    grad = self.fc_net.backward(grad_out)
    grad = self.dropout.backward(grad)
    grad_xp = self._xp.asarray(grad)
    grad_xp = self.flatten.backward(grad_xp)
    self.conv_net.backward(grad_xp)
```

FC 경로의 gradient는 numpy이므로 Conv 경로로 전달 전 `self._xp.asarray(grad)`로 CuPy 배열로 변환한다. `conv_net.backward`는 반환값이 없다(최하위 레이어이므로 `dx`를 사용하지 않는다).

### 3.3. params와 grads

```python
@property
def params(self):
    return self.conv_net.params + self.fc_net.params

@property
def grads(self):
    return self.conv_net.grads + self.fc_net.grads
```

Conv 경로 파라미터(CuPy 배열)와 FC 경로 파라미터(numpy 배열)를 하나의 리스트로 합친다. optimizer는 이 리스트를 순회하며 각 배열 타입에 맞게 업데이트한다.

## 4. 사용법

최소 사용 예제는 다음과 같다.

```python
import numpy as np
from src.models.cnn import CNN

model = CNN(task="multiclass", seed=42)
x = np.random.randn(4, 784).astype(np.float32)
logits = model.forward(x)

print(logits.shape)
print(len(model.params))
```

예상 출력은 다음과 같다.

```text
(4, 10)
8
```

프로젝트 통합 예제는 다음과 같다. `CNN`은 `MLP`와 동일한 인터페이스를 사용한다.

```python
from src.models.cnn import CNN
from src.nn.losses import cross_entropy, cross_entropy_grad
from src.core.optimizers import Adam

model = CNN(task="multiclass", seed=42)
optimizer = Adam(model, lr=0.001)

model.train()
for images, targets in train_loader:
    logits = model.forward(images)
    loss = cross_entropy(logits, targets)
    grad = cross_entropy_grad(logits, targets)
    model.backward(grad)
    optimizer.step()
```

## 5. 테스트

테스트 파일은 `tests/stage4/test_cnn.py`이다.

```bash
conda run -n cupy_py311_cuda118 pytest tests/stage4/test_cnn.py -v
```

CuPy 없는 환경에서는 numpy fallback으로 실행한다.

```bash
conda run -n numpy_py311 pytest tests/stage4/test_cnn.py -v
```

테스트 구성은 다음과 같다.

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestCNNForward` | 4 | task별 logit shape, dtype float32(numpy), 소규모 배치 |
| `TestCNNParams` | 3 | params 수 8개, grads 수 8개, shape 일치 |
| `TestCNNBackward` | 3 | backward 후 grads 비 0 확인, params 변경 없음 확인 |
| `TestCNNTrainEval` | 2 | train/eval 모드 전환, Dropout 동작 차이 |

## 6. 요약

`CNN`은 두 개의 Conv-ReLU-Pool 블록과 FC 블록으로 구성된 CuPy 기반 합성곱 네트워크이다. `MLP`와 동일한 `forward`/`backward`/`params`/`grads` 인터페이스를 유지하여 Trainer, Evaluator가 모델 종류를 구분하지 않고 동일하게 동작한다. CuPy/numpy 경계는 `Flatten` 전후에서 명시적으로 처리하며, CuPy 미설치 환경에서는 자동으로 numpy fallback이 적용된다.

다음 Stage에서는 [[phase5.1_optimizers]]를 다룬다.
