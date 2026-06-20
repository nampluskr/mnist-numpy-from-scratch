---
tags: [docs, stage4, models, cnn, cupy]
created: "2026-06-20"
updated: "2026-06-21"
---

# CNN 모델

## 1. 개요

`src/models/cnn.py`의 `CNN`은 CuPy 기반 2층 합성곱 네트워크이다. `MLP`와 동일한 public interface(`forward`, `backward`, `params`, `grads`)를 유지하여 Trainer, Evaluator, Predictor가 모델 종류에 관계없이 같은 방식으로 동작한다. Conv 경로는 CuPy 배열로 GPU에서 실행하고, `Flatten` 이후 FC 경로는 numpy 배열로 CPU에서 실행한다. CuPy가 없는 환경에서는 자동으로 numpy fallback이 적용된다.

**목표**
- `Conv2d`, `MaxPool2d`, `Flatten`, `Linear`, `ReLU`, `Dropout`을 조립하여 CNN을 구성한다.
- `forward`에서 `(N, 784)` numpy 입력을 받아 `(N, output_dim)` numpy logit을 반환한다.
- CuPy/numpy 경계를 `forward`/`backward` 내부에서 명시적으로 처리한다.

## 2. 개념

### 2.1. CNN이란 무엇인가

CNN(Convolutional Neural Network)은 합성곱 연산을 이용하여 이미지의 지역적 패턴을 학습하는 신경망이다. MLP가 입력 전체를 하나의 벡터로 펼쳐 모든 뉴런을 연결하는 것과 달리, CNN은 작은 커널(filter)을 이미지 위에서 슬라이딩하며 지역 패턴을 감지한다.

MLP로 MNIST 이미지를 처리하면 784개 픽셀이 모두 동등한 위치로 취급된다. 픽셀 `(0, 0)`과 `(1, 0)`이 같은 행에 인접한 픽셀이라는 공간 정보가 완전히 사라진다. CNN은 공간 구조를 유지하면서 인접 픽셀 간의 관계(선, 모서리, 곡선 등)를 명시적으로 학습한다.

CNN이 이미지 처리에 MLP보다 효과적인 이유는 세 가지이다.

**지역 연결성(Local Connectivity)**: 커널은 이미지의 작은 영역만 봄으로써 지역 패턴에 집중한다. 3x3 커널은 9개 픽셀만 연결되어 있으므로 완전 연결보다 파라미터가 훨씬 적다.

**가중치 공유(Weight Sharing)**: 동일한 커널이 이미지 전체를 슬라이딩하므로, 이미지 어느 위치에 같은 패턴이 나타나더라도 동일한 커널로 감지한다. 이는 파라미터 수를 크게 줄이고 위치 불변성(translation invariance)을 제공한다.

**계층적 특징 학습(Hierarchical Feature Learning)**: 앞쪽 레이어는 선, 모서리 같은 저수준 특징을, 뒤쪽 레이어는 더 복잡한 고수준 특징을 학습한다. 이 계층 구조로 손글씨 숫자의 복잡한 형태를 단계적으로 표현한다.

### 2.2. 이 프로젝트의 CNN 구조

이 프로젝트의 CNN은 두 개의 Conv-ReLU-Pool 블록과 FC(Fully Connected) 블록으로 구성된다. MNIST 이미지는 `(N, 784)` 형태로 입력되므로 `forward` 시작 시 `(N, 1, 28, 28)`로 reshape한다.

```text
입력: (N, 784) float32  --(reshape)-->  (N, 1, 28, 28)
  |
  v  [Conv 블록 1]  -- CuPy 배열
Conv2d(1, 32, 3, pad=1)    (N, 1, 28, 28) -> (N, 32, 28, 28)
ReLU
MaxPool2d(2, 2)            (N, 32, 28, 28) -> (N, 32, 14, 14)
  |
  v  [Conv 블록 2]  -- CuPy 배열
Conv2d(32, 64, 3, pad=1)   (N, 32, 14, 14) -> (N, 64, 14, 14)
ReLU
MaxPool2d(2, 2)            (N, 64, 14, 14) -> (N, 64, 7, 7)
  |
  v
Flatten                    (N, 64, 7, 7) -> (N, 3136)
  |
  v  [CuPy -> numpy 변환 경계]
Dropout(0.5)               (N, 3136) -> (N, 3136)  -- numpy 배열
  |
  v  [FC 블록]  -- numpy 배열
Linear(3136, 256)          (N, 3136) -> (N, 256)
ReLU
Linear(256, output_dim)    (N, 256) -> (N, output_dim)
  |
  v
출력: (N, output_dim) float32 numpy  -- raw logit
```

각 블록의 역할은 다음과 같다.

| 블록 | 구성 | 역할 |
|---|---|---|
| Conv 블록 1 | `Conv2d(1, 32, 3, pad=1) -> ReLU -> MaxPool2d(2, 2)` | 저수준 특징(선, 모서리) 추출 |
| Conv 블록 2 | `Conv2d(32, 64, 3, pad=1) -> ReLU -> MaxPool2d(2, 2)` | 고수준 특징(곡선, 부분 형태) 추출 |
| FC 블록 | `Dropout -> Linear(3136, 256) -> ReLU -> Linear(256, output_dim)` | 특징 조합 및 분류 |

task별 출력 차원은 MLP와 동일하다.

| task | output_dim | 출력 activation | 해석 |
|---|---|---|---|
| `multiclass` | 10 | `softmax` (losses.py 내부) | 10개 클래스별 raw score |
| `binary` | 1 | `sigmoid` (losses.py 내부) | 이진 클래스의 raw score |
| `regression` | 1 | 없음 (identity) | 예측값 그대로 사용 |

### 2.3. 합성곱과 feature map

`Conv2d(1, 32, 3, pad=1)`에서 숫자의 의미는 다음과 같다.

- `in_channels=1`: 입력 채널 수. MNIST는 흑백이므로 1이다.
- `out_channels=32`: 출력 채널 수, 즉 학습할 커널(필터)의 수이다. 32개의 서로 다른 패턴 감지기를 학습한다.
- `kernel_size=3`: 커널 크기. 3x3 픽셀 영역을 한 번에 본다.
- `padding=1`: 입력 가장자리에 0을 1칸씩 추가한다. 28x28 입력이 Conv 후에도 28x28을 유지한다.

커널 하나는 `(1, 3, 3)` 형태의 학습 가능한 가중치 배열이다. 이 커널을 이미지 위에서 슬라이딩하면 28x28 feature map 1개가 생긴다. 32개의 커널이 있으므로 출력은 `(32, 28, 28)` feature map이 된다.

padding이 없으면 3x3 커널 적용 후 28x28이 26x26으로 줄어든다. `padding=1`을 추가하면 입력이 30x30으로 확장되어 Conv 후에도 28x28을 유지한다. 이를 same padding이라 한다.

**feature map이란**: 커널이 이미지의 각 위치를 슬라이딩하며 "이 위치에 이 패턴이 얼마나 있는가"를 나타내는 2D 행렬이다. 32채널의 feature map은 32가지 서로 다른 패턴의 공간 분포를 동시에 표현한다.

### 2.4. 공간 크기 변화 추적

입력부터 Flatten까지 feature map의 공간 크기 변화를 추적하면 다음과 같다.

| 레이어 | 입력 shape | 출력 shape | 크기 변화 이유 |
|---|---|---|---|
| reshape | `(N, 784)` | `(N, 1, 28, 28)` | 1D -> 2D 복원 |
| `Conv2d(1, 32, 3, pad=1)` | `(N, 1, 28, 28)` | `(N, 32, 28, 28)` | pad=1로 크기 유지 |
| `ReLU` | `(N, 32, 28, 28)` | `(N, 32, 28, 28)` | 변화 없음 |
| `MaxPool2d(2, 2)` | `(N, 32, 28, 28)` | `(N, 32, 14, 14)` | 2x2 윈도우로 절반 축소 |
| `Conv2d(32, 64, 3, pad=1)` | `(N, 32, 14, 14)` | `(N, 64, 14, 14)` | pad=1로 크기 유지 |
| `ReLU` | `(N, 64, 14, 14)` | `(N, 64, 14, 14)` | 변화 없음 |
| `MaxPool2d(2, 2)` | `(N, 64, 14, 14)` | `(N, 64, 7, 7)` | 2x2 윈도우로 절반 축소 |
| `Flatten` | `(N, 64, 7, 7)` | `(N, 3136)` | 64 x 7 x 7 = 3136 |

Flatten 이후 3136은 `64 * 7 * 7`에서 나온다. 이 값이 첫 번째 FC 레이어의 입력 차원이 된다.

출력 크기 수식으로도 확인할 수 있다. Conv 블록 1의 경우 다음과 같다.

$$
out\_h = \left\lfloor \frac{28 + 2 \times 1 - 3}{1} \right\rfloor + 1 = 28
$$

MaxPool2d(2, 2)의 경우 다음과 같다.

$$
out\_h = \left\lfloor \frac{28 + 0 - 2}{2} \right\rfloor + 1 = 14
$$

### 2.5. 파라미터 수와 MLP 비교

CNN의 학습 가능 파라미터는 다음과 같다.

| 레이어 | W shape | b shape | 파라미터 수 |
|---|---|---|---|
| `Conv2d(1, 32, 3)` | `(32, 1, 3, 3)` | `(32,)` | 288 + 32 = 320 |
| `Conv2d(32, 64, 3)` | `(64, 32, 3, 3)` | `(64,)` | 18,432 + 64 = 18,496 |
| `Linear(3136, 256)` | `(3136, 256)` | `(256,)` | 802,816 + 256 = 803,072 |
| `Linear(256, output_dim)` | `(256, 10)` | `(10,)` | 2,560 + 10 = 2,570 |
| 합계 | - | - | 824,458 |

`CNN.params`는 총 8개 배열의 리스트이다(`Conv2d` 2개 x 2 + `Linear` 2개 x 2). `float32` 기준으로 약 3.1 MB를 차지한다.

MLP(235,146 파라미터)와 비교하면 CNN이 약 3.5배 더 많다. 그러나 Conv 레이어 자체는 커널 크기 덕분에 매우 적은 파라미터(320 + 18,496)로 이미지 특징을 효율적으로 추출한다. 파라미터 대부분은 FC 레이어에 집중된다.

### 2.6. Forward Pass

Forward pass는 numpy 입력을 CuPy로 변환하여 Conv 경로를 통과시키고, Flatten 후 다시 numpy로 변환하여 FC 경로를 통과하는 두 단계로 나뉜다.

**Step 1: numpy -> CuPy 변환과 reshape**

```text
x: (N, 784) numpy
  -> _xp.asarray(x): (N, 784) CuPy
  -> reshape(-1, 1, 28, 28): (N, 1, 28, 28) CuPy
```

`_xp.asarray(x)`는 numpy 배열을 GPU 메모리로 복사한다. CuPy fallback 환경에서는 numpy 배열 그대로 유지된다.

**Step 2: Conv 경로 (CuPy)**

```text
(N, 1, 28, 28) CuPy
  -> Conv2d(1, 32, 3, pad=1) -> ReLU -> MaxPool2d(2, 2): (N, 32, 14, 14) CuPy
  -> Conv2d(32, 64, 3, pad=1) -> ReLU -> MaxPool2d(2, 2): (N, 64, 7, 7) CuPy
  -> Flatten: (N, 3136) CuPy
```

모든 연산이 GPU에서 실행된다. `im2col`, 행렬 곱, `argmax` 등 연산을 CuPy가 GPU에서 처리한다.

**Step 3: CuPy -> numpy 변환**

```text
(N, 3136) CuPy
  -> .get(): (N, 3136) numpy  (GPU -> CPU 메모리 복사)
```

`.get()`은 CuPy 전용 메서드로 GPU 메모리의 배열을 CPU 메모리(numpy)로 복사한다. numpy fallback 환경에서는 `np.asarray()`로 대체한다. 이 지점이 CuPy/numpy 경계이다.

**Step 4: FC 경로 (numpy)**

```text
(N, 3136) numpy
  -> Dropout(0.5): (N, 3136) numpy  (학습 모드에서만 마스크 적용)
  -> Linear(3136, 256) -> ReLU: (N, 256) numpy
  -> Linear(256, output_dim): (N, output_dim) numpy  -- raw logit
```

FC 경로는 CPU에서 실행된다. 출력은 numpy 배열이므로 `cross_entropy`, `cross_entropy_grad` 등 losses.py 함수와 직접 연동된다.

### 2.7. Backward Pass

Backward pass는 forward의 역순으로 진행된다. FC 경로(numpy)에서 시작하여 numpy -> CuPy 변환 후 Conv 경로(CuPy)로 이어진다.

**Step 1: FC 경로 역전파 (numpy)**

```text
grad_out: (N, output_dim) numpy  -- from losses.*_grad
  -> fc_net.backward(grad_out): (N, 3136) numpy
     (Linear(256, output_dim).backward -> ReLU.backward -> Linear(3136, 256).backward)
  -> dropout.backward(grad): (N, 3136) numpy
     (학습 모드에서 forward의 마스크를 그대로 곱함)
```

**Step 2: numpy -> CuPy 변환**

```text
(N, 3136) numpy
  -> _xp.asarray(grad): (N, 3136) CuPy  (CPU -> GPU 메모리 복사)
```

**Step 3: Flatten 역전파와 Conv 경로 역전파 (CuPy)**

```text
(N, 3136) CuPy
  -> flatten.backward: (N, 64, 7, 7) CuPy  (단순 reshape)
  -> conv_net.backward:
     MaxPool2d(2,2).backward -> Conv2d(32,64,3).backward
     -> MaxPool2d(2,2).backward -> Conv2d(1,32,3).backward
```

`conv_net.backward`의 반환값(입력에 대한 gradient)은 사용하지 않는다. CNN의 최하위 레이어이므로 더 전달할 곳이 없다. 각 `Conv2d`의 `grad_w`와 `grad_b`에 gradient가 in-place로 저장된다.

### 2.8. CuPy/numpy 경계 설계

Conv 경로를 CuPy로, FC 경로를 numpy로 나눈 이유는 다음과 같다.

**Conv 연산의 GPU 가속**: `im2col` 후 행렬 곱은 대규모 병렬 연산이다. 28x28 이미지 배치를 Conv 레이어로 처리할 때 GPU의 병렬 처리 능력이 효과적으로 활용된다. CuPy는 NumPy와 동일한 API를 제공하므로 `xp` 파라미터만 변경하면 GPU 실행이 가능하다.

**FC 경로의 numpy 유지**: FC 레이어는 단순한 행렬 곱이며 `Linear`와 `Dropout`은 `src/nn/layers.py`의 numpy 기반 구현을 그대로 사용한다. 별도의 CuPy 변환 없이 numpy로 처리한다.

**경계 지점**: `Flatten` 직후가 경계이다. forward에서는 CuPy `Flatten` 출력을 `.get()`으로 numpy로 변환한다. backward에서는 numpy FC gradient를 `_xp.asarray()`로 CuPy로 변환한 뒤 `Flatten.backward`에 전달한다.

```text
forward:  ... -> Flatten (CuPy) --.get()--> Dropout (numpy) -> ...
backward: ... <- Flatten (CuPy) <-_xp.asarray()-- Dropout (numpy) <- ...
```

### 2.9. Dropout의 역할

`Dropout(0.5)`은 FC 첫 번째 레이어 직전에 배치된다. 학습 중에는 3136개 뉴런 중 절반을 무작위로 비활성화한다. 이는 FC 레이어가 특정 특징에 과도하게 의존하지 않도록 강제하여 과적합을 방지한다.

inverted dropout 방식으로 비활성화되지 않은 뉴런의 출력을 $\frac{1}{1-p} = 2.0$으로 스케일링한다. 이렇게 하면 학습과 평가 사이의 출력 스케일이 일치하여, 평가 모드에서 별도의 스케일 조정 없이 입력을 그대로 통과시킬 수 있다.

train/eval 모드 전환은 `model.train()` / `model.eval()` 호출로 이루어진다. `CNN.train()`은 `conv_net`, `dropout`, `fc_net` 각각에 `train()` 신호를 전파한다.

```text
model.train() 호출 시:
  conv_net.train()  -> 내부 ReLU 레이어들의 training=True
  dropout.train()   -> training=True (마스크 적용)
  fc_net.train()    -> 내부 ReLU 레이어들의 training=True
```

학습 루프에서는 `model.train()`을, 평가 루프에서는 `model.eval()`을 명시적으로 호출해야 Dropout이 올바르게 동작한다.

### 2.10. CuPy fallback 설계

파일 상단에서 CuPy import를 시도하고 실패하면 numpy를 `_xp`로 사용한다.

```python
try:
    import cupy as _xp
except ImportError:
    import numpy as _xp
```

이 패턴은 다음을 가능하게 한다.

- CUDA GPU가 있는 환경: CuPy로 GPU 가속 실행
- GPU 없는 환경(개발·테스트): numpy로 CPU 실행, 동일한 코드로 기능 검증

`Conv2d`와 `MaxPool2d`는 `xp` 파라미터로 배열 모듈을 받아 `xp.zeros`, `xp.argmax` 등의 연산을 처리한다. 모듈 자체는 CuPy/numpy를 구분하지 않으므로 fallback이 자동으로 동작한다.

단, CuPy fallback 시 im2col의 루프 기반 연산이 CPU에서 실행되므로 대규모 배치에서 MLP보다 느릴 수 있다. fallback은 성능 검증이 아닌 기능 검증 용도이다.

### 2.11. params와 grads 설계

`CNN.params`와 `CNN.grads`는 `conv_net.params + fc_net.params`로 구성된다. 순서는 다음과 같다.

```text
CNN.params = [conv1_W, conv1_b, conv2_W, conv2_b,   <- conv_net (CuPy 배열)
              fc1_W,   fc1_b,   fc2_W,   fc2_b]      <- fc_net   (numpy 배열)
```

총 8개 배열이며, 앞 4개는 CuPy 배열, 뒤 4개는 numpy 배열이다. optimizer는 이 리스트를 순회하며 각 배열 타입에 맞게 in-place 업데이트한다. `Adam`과 `SGD`는 배열 연산만 수행하므로 CuPy와 numpy 배열 모두 동일하게 처리한다.

`Dropout`과 `ReLU`는 파라미터가 없으므로 `params`와 `grads` 리스트에 포함되지 않는다.

### 2.12. MLP와 CNN 인터페이스 비교

Trainer, Evaluator, Predictor는 모델 종류를 알 필요 없이 동일한 인터페이스로 동작한다.

| 항목 | MLP | CNN |
|---|---|---|
| 입력 | `(N, 784)` numpy | `(N, 784)` numpy |
| 출력 | `(N, output_dim)` numpy | `(N, output_dim)` numpy |
| `forward(x)` | `Sequential.__call__` | reshape -> CuPy Conv -> numpy FC |
| `backward(grad)` | `Sequential.backward` | numpy FC -> CuPy Conv |
| `params` | 6개 numpy 배열 | 8개 (CuPy 4 + numpy 4) |
| `train()` / `eval()` | Sequential에 위임 | conv_net, dropout, fc_net 각각 |
| GPU 실행 | 없음 | Conv 경로 (CuPy) |

입출력이 모두 `(N, 784)` -> `(N, output_dim)` numpy이므로 Trainer는 다음과 같이 모델을 교체하여 동일하게 사용한다.

```python
model = MLP(task="multiclass", seed=42)  # 또는 CNN(task="multiclass", seed=42)
optimizer = SGD(model, lr=0.01)          # 동일한 optimizer 코드

model.train()
for images, targets in train_loader:
    logits = model.forward(images)       # 내부 처리는 다르지만 동일한 호출
    loss = cross_entropy(logits, targets)
    grad = cross_entropy_grad(logits, targets)
    model.backward(grad)
    optimizer.step()
```

### 2.13. 학습 한 스텝의 전체 흐름

CNN을 이용한 학습 한 스텝의 전체 흐름은 다음과 같다.

```text
[1] images, targets = next(train_loader)
      images: (N, 784) numpy float32
      targets: (N, 10) one-hot numpy (multiclass 예시)

[2] logits = model.forward(images)
      (N, 784) --(CuPy asarray + reshape)--> (N, 1, 28, 28) CuPy
      --(conv_net)--> (N, 64, 7, 7) CuPy
      --(flatten)--> (N, 3136) CuPy
      --(.get())--> (N, 3136) numpy
      --(dropout + fc_net)--> (N, 10) numpy

[3] loss = cross_entropy(logits, targets)
      내부에서 softmax 적용, scalar float 반환

[4] grad = cross_entropy_grad(logits, targets)
      (N, 10) numpy d(loss)/d(logits)

[5] model.backward(grad)
      fc_net.backward + dropout.backward: numpy
      --(CuPy asarray)--> CuPy gradient
      flatten.backward + conv_net.backward: CuPy
      각 Conv2d의 grad_w, grad_b에 gradient 저장

[6] optimizer.step()
      model.params와 model.grads를 순회하며 in-place 업데이트
      (CuPy 배열은 CuPy 연산, numpy 배열은 numpy 연산)
```

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

FC 경로의 gradient는 numpy이므로 Conv 경로로 전달 전 `self._xp.asarray(grad)`로 CuPy 배열로 변환한다. `conv_net.backward`는 반환값이 없다. 최하위 레이어이므로 `dx`를 사용하지 않는다.

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

`CNN`은 두 개의 Conv-ReLU-Pool 블록과 FC 블록으로 구성된 CuPy 기반 합성곱 네트워크이다. `MLP`와 동일한 `forward`/`backward`/`params`/`grads` 인터페이스를 유지하여 Trainer, Evaluator가 모델 종류를 구분하지 않고 동일하게 동작한다. CuPy/numpy 경계는 `Flatten` 전후에서 명시적으로 처리하며, CuPy 미설치 환경에서는 자동으로 numpy fallback이 적용된다. Conv 블록은 지역 패턴과 공간 구조를 학습하고, FC 블록은 추출된 특징을 조합하여 task별 logit을 생성한다.

다음 Stage에서는 [[phase5.1_optimizers]]를 다룬다.
