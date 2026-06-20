---
tags: [docs, stage3, nn, conv, cnn]
created: "2026-06-20"
updated: "2026-06-20"
---

# CNN 레이어 모듈

## 1. 개요

`src/nn/conv.py`는 CNN을 구성하는 레이어 모듈과 핵심 변환 함수를 제공한다. `im2col`/`col2im` 기반으로 `Conv2d`, `MaxPool2d`, `Flatten`, `Dropout`을 구현한다. 모든 레이어는 `Module` 기반이며 numpy와 cupy를 모두 지원한다. `xp` 파라미터로 배열 모듈을 선택하여 CPU(numpy)와 GPU(cupy) 전환이 가능하다.

**목표**
- `im2col`/`col2im`로 2D 합성곱을 행렬 곱으로 변환하여 배치 처리를 효율화한다.
- `Conv2d`와 `MaxPool2d`를 `Module` 인터페이스로 구현한다.
- `xp` 파라미터로 numpy/cupy 전환을 지원한다.

## 2. 개념

### 2.1. im2col 변환

직접 루프로 합성곱을 구현하면 배치와 공간 차원에 걸쳐 4중 이상의 중첩 루프가 필요하여 매우 느리다. `im2col`은 입력 텐서를 kernel이 슬라이딩하는 모든 위치의 패치를 행으로 펼쳐 하나의 2D 행렬로 변환한다. 이 행렬에 커널 가중치 행렬을 곱하면 합성곱 결과를 한 번의 행렬 곱으로 계산할 수 있다.

입출력 shape 변환은 다음과 같다.

| 단계 | shape | 설명 |
|---|---|---|
| 입력 `x` | `(B, C, H, W)` | 배치 이미지 |
| `im2col` 출력 `col` | `(B*out_h*out_w, C*K*K)` | 각 행이 하나의 패치 |
| 커널 `w` reshape | `(out_c, C*K*K)` | 각 행이 하나의 필터 |
| 행렬 곱 결과 | `(B*out_h*out_w, out_c)` | 각 행이 하나의 출력 위치 |
| 최종 출력 | `(B, out_c, out_h, out_w)` | reshape + transpose |

출력 크기는 다음 수식으로 계산한다.

$$
out\_h = \frac{H + 2 \times padding - K}{stride} + 1
$$

### 2.2. col2im 역변환

`col2im`은 `im2col`의 역연산으로, backward에서 `dx`를 복원하는 데 사용한다. 동일한 입력 위치가 여러 패치에 등장할 수 있으므로 단순 역배열이 아닌 누적 합산(`+=`)으로 복원한다.

### 2.3. Dropout

`Dropout`은 학습 중 무작위로 뉴런을 비활성화하여 과적합을 방지하는 정규화 기법이다. 학습 모드에서는 확률 `p`로 뉴런을 0으로 만들고, 나머지 뉴런의 값을 `1 / (1 - p)`로 스케일링하여 기댓값을 유지한다(inverted dropout). 평가 모드에서는 입력을 그대로 통과시킨다.

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `im2col` | 함수 | `x (B, C, H, W)`, `kernel_size`, `stride`, `padding`, `xp` | `col, out_h, out_w` | 이미지 패치 행렬 변환 |
| `col2im` | 함수 | `col`, `x_shape`, `kernel_size`, `stride`, `padding`, `xp` | `dx (B, C, H, W)` | 패치 행렬 역변환 |
| `Conv2d` | 클래스 | `in_c`, `out_c`, `kernel_size`, `stride`, `padding`, `xp` | layer instance | 2D 합성곱 레이어 |
| `MaxPool2d` | 클래스 | `kernel_size`, `stride`, `padding`, `xp` | layer instance | 2D max pooling 레이어 |
| `Flatten` | 클래스 | - | layer instance | `(B, C, H, W)` -> `(B, C*H*W)` |
| `Dropout` | 클래스 | `p=0.5` | layer instance | inverted dropout |

### 3.1. Conv2d

```python
class Conv2d(Module):
    def forward(self, x):
        B, C, H, W = x.shape
        self._x = x
        col_x, out_h, out_w = im2col(x, self.kernel_size, self.stride, self.padding, xp=self.xp)
        self._col_cache = (col_x, out_h, out_w)
        col_w = self.w.reshape(self.out_channels, -1)
        out = col_x @ col_w.T + self.b
        return out.reshape(B, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)

    def backward(self, dout):
        col_x, out_h, out_w = self._col_cache
        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        self.grad_b[...] = dout_flat.sum(axis=0)
        self.grad_w[...] = (dout_flat.T @ col_x).reshape(self.grad_w.shape)
        col_w = self.w.reshape(self.out_channels, -1)
        dcol_x = dout_flat @ col_w
        return col2im(dcol_x, self._x.shape, self.kernel_size, self.stride, self.padding, xp=self.xp)
```

커널 가중치 초기화는 He 초기화를 numpy로 생성한 뒤 `xp.asarray`로 변환한다. numpy로 먼저 생성하는 이유는 `np.random.default_rng`가 numpy 배열을 반환하기 때문이며, cupy로 직접 생성하면 seed 재현성이 다를 수 있다.

### 3.2. MaxPool2d

`MaxPool2d`는 `im2col`로 각 풀링 윈도우를 행으로 펼친 뒤 행별 최대값을 취한다. backward에서는 최대값의 위치(`_max_indices`)를 기록하여 해당 위치에만 gradient를 전달한다.

### 3.3. Dropout

```python
class Dropout(Module):
    def forward(self, x):
        if self.training:
            self._mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype) / (1.0 - self.p)
            return x * self._mask
        return x

    def backward(self, dout):
        if self.training:
            return dout * self._mask
        return dout
```

`Module.training` 플래그로 학습/평가 모드를 구분한다. `Sequential.train()` / `Sequential.eval()`이 하위 레이어에 전파하므로 별도로 각 레이어를 전환하지 않아도 된다.

## 4. 사용법

최소 사용 예제는 다음과 같다.

```python
import numpy as np
from src.nn.conv import Conv2d, MaxPool2d, Flatten

x = np.random.randn(4, 1, 28, 28).astype(np.float32)

conv = Conv2d(1, 8, kernel_size=3, padding=1)
pool = MaxPool2d(kernel_size=2)
flat = Flatten()

out = flat(pool(conv(x)))
print(out.shape)
```

예상 출력은 다음과 같다.

```text
(4, 1568)
```

프로젝트 통합 예제는 다음과 같다. CuPy 환경에서 GPU 실행 시 `xp=cp`를 전달한다.

```python
try:
    import cupy as cp
    xp = cp
except ImportError:
    xp = np

from src.nn.conv import Conv2d, MaxPool2d, Flatten
from src.nn.layers import Linear, ReLU, Sequential

model = Sequential(
    Conv2d(1, 8, kernel_size=3, padding=1, xp=xp),
    ReLU(),
    MaxPool2d(kernel_size=2, xp=xp),
    Flatten(),
    Linear(8 * 14 * 14, 10),
)
```

## 5. 테스트

테스트 파일은 `tests/stage3/test_conv.py`이다.

```bash
conda run -n numpy_py311 pytest tests/stage3/test_conv.py -v
```

테스트 구성은 다음과 같다.

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestIm2col` | 4 | output shape, stride/padding 적용, col 값 검증 |
| `TestCol2im` | 2 | 역변환 shape, im2col과 col2im 왕복 shape 일치 |
| `TestConv2d` | 5 | forward shape, padding/stride, backward dx/grad_w shape, in-place 대입 |
| `TestMaxPool2d` | 4 | forward shape, backward shape, max 위치 gradient 전달 |
| `TestFlatten` | 3 | forward shape, backward shape 복원 |
| `TestDropout` | 4 | training 모드 마스크 적용, eval 모드 통과, inverted scaling |

## 6. 요약

`conv.py`는 `im2col`/`col2im` 기반으로 `Conv2d`, `MaxPool2d`, `Flatten`, `Dropout`을 제공한다. `im2col`은 합성곱을 행렬 곱으로 변환하여 배치 처리를 효율화하고, `xp` 파라미터로 numpy/cupy를 선택하여 CPU와 GPU 실행을 모두 지원한다. 모든 레이어는 `Module` 인터페이스를 따르므로 `Sequential`로 MLP 레이어와 자유롭게 조합할 수 있다.

다음 Stage에서는 [[phase4.1_mlp]]를 다룬다.
