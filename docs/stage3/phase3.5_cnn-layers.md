---
tags: [docs, stage3, nn, conv, cnn]
created: "2026-06-20"
updated: "2026-06-21"
---

# CNN 레이어 모듈

## 1. 개요

`src/nn/conv.py`는 CNN을 구성하는 레이어 모듈과 핵심 변환 함수를 제공한다. `im2col`/`col2im` 기반으로 `Conv2d`, `MaxPool2d`, `Flatten`, `Dropout`을 구현한다. 모든 레이어는 `Module` 기반이며 numpy와 cupy를 모두 지원한다. `xp` 파라미터로 배열 모듈을 선택하여 CPU(numpy)와 GPU(cupy) 전환이 가능하다.

**목표**
- `im2col`/`col2im`로 2D 합성곱을 행렬 곱으로 변환하여 배치 처리를 효율화한다.
- `Conv2d`와 `MaxPool2d`를 `Module` 인터페이스로 구현한다.
- `xp` 파라미터로 numpy/cupy 전환을 지원한다.

## 2. 개념

### 2.1. im2col

직접 루프로 합성곱을 구현하면 배치·공간 차원에 걸쳐 4중 이상의 중첩 루프가 필요하여 매우 느리다. `im2col`은 커널이 슬라이딩하는 모든 위치의 입력 패치를 행으로 펼쳐 하나의 2D 행렬로 변환한다. 이 행렬에 커널 가중치 행렬을 곱하면 합성곱 전체를 단일 행렬 곱으로 계산할 수 있다.

**출력 크기 수식**

$$
out\_h = \left\lfloor \frac{H + 2 \times padding - K}{stride} \right\rfloor + 1, \qquad
out\_w = \left\lfloor \frac{W + 2 \times padding - K}{stride} \right\rfloor + 1
$$

$H, W$는 입력 공간 크기, $K$는 커널 크기, $stride$는 슬라이딩 보폭, $padding$은 입력 가장자리에 추가하는 0의 폭이다.

**Shape 변환**

| 단계 | shape | 설명 |
|---|---|---|
| 입력 `x` | `(B, C, H, W)` | 배치 이미지 |
| `im2col` 출력 `col` | `(B·out_h·out_w, C·K·K)` | 각 행이 하나의 커널 위치 패치 |
| 커널 `w` reshape | `(out_c, C·K·K)` | 각 행이 하나의 출력 채널 필터 |
| 행렬 곱 `col @ w.T` | `(B·out_h·out_w, out_c)` | 각 행이 하나의 출력 공간 위치 |
| reshape + transpose | `(B, out_c, out_h, out_w)` | 최종 feature map |

### 2.2. col2im

`col2im`은 `im2col`의 역연산으로, Conv2d backward에서 입력에 대한 gradient `dx`를 복원하는 데 사용한다.

동일한 입력 픽셀이 커널이 겹치는 여러 위치의 패치에 반복 등장할 수 있다. 따라서 단순 역배열이 아닌 각 패치 위치의 gradient를 원래 입력 좌표에 누적 합산(`+=`)하여 복원한다.

$$
dx[b, c, h, w] = \sum_{\text{patches covering } (h, w)} d\_col[\text{patch}, \text{position}]
$$

패딩이 적용된 경우 패딩 영역의 gradient는 버리고 원래 입력 크기에 해당하는 부분만 반환한다.

### 2.3. Conv2d

2D 합성곱 레이어이다. `im2col`로 입력을 행렬로 펼친 뒤 커널 가중치와 행렬 곱을 수행한다. `xp` 파라미터로 numpy(CPU)와 cupy(GPU)를 선택할 수 있다.

**Forward**

$$
\text{out}[b, f, i, j] = \sum_{c=1}^{C} \sum_{k_h=0}^{K-1} \sum_{k_w=0}^{K-1} x[b, c,\, i \cdot s + k_h,\, j \cdot s + k_w] \cdot W[f, c, k_h, k_w] + b[f]
$$

$b$는 배치 인덱스, $f$는 출력 채널(필터) 인덱스, $s$는 stride, $W$는 커널 가중치이다. `im2col` 후 행렬 곱으로 구현하면 이 루프를 단일 연산으로 처리한다.

**Backward**

`dout`을 `(B·out_h·out_w, out_c)` 형태로 flatten한 후, chain rule로 각 gradient를 계산한다.

$$
\frac{\partial L}{\partial W} = \text{dout\_flat}^T \cdot col\_x, \qquad
\frac{\partial L}{\partial b} = \sum \text{dout\_flat}, \qquad
\frac{\partial L}{\partial col\_x} = \text{dout\_flat} \cdot W_{\text{flat}}
$$

$\frac{\partial L}{\partial col\_x}$는 `col2im`으로 원래 입력 shape인 `(B, C, H, W)`로 복원한다.

**He 초기화**

$$
W \sim \mathcal{N}\!\left(0,\ \sqrt{\frac{2}{C \cdot K \cdot K}}\right)
$$

입력 채널 수와 커널 크기를 반영하여 분산을 설정한다. numpy로 먼저 생성한 뒤 `xp.asarray`로 변환하여 cupy에서도 재현 가능한 초기화를 보장한다.

### 2.4. MaxPool2d

2D max pooling 레이어이다. 커널 크기의 윈도우 내에서 최대값만 통과시키고 나머지는 버린다. 공간 크기를 줄여 파라미터 수와 계산량을 감소시킨다.

**Forward**

$$
\text{out}[b, c, i, j] = \max_{k_h, k_w} x[b, c,\, i \cdot s + k_h,\, j \cdot s + k_w]
$$

`im2col`로 각 풀링 윈도우를 행으로 펼친 뒤 행별 `argmax`로 최대값 위치를 기록하고, `max`로 최대값을 선택한다.

**Backward**

최대값이 있던 위치에만 gradient를 전달하고 나머지 위치는 0이다.

$$
\frac{\partial L}{\partial x[b, c, h, w]} = \begin{cases} \text{dout}[b, c, i, j] & (h, w) = \arg\max\text{인 경우} \\ 0 & \text{그 외} \end{cases}
$$

forward에서 `_max_indices`로 저장한 argmax 위치에 `dout`을 배치한 뒤 `col2im`으로 복원한다.

### 2.5. Flatten

4D 텐서를 2D로 펼치는 레이어이다. CNN에서 공간 feature map을 fully-connected 레이어에 연결하기 위해 사용한다.

**Forward**

$$
\text{out} = x.\text{reshape}(B, -1), \quad (B, C, H, W) \to (B,\, C \cdot H \cdot W)
$$

**Backward**

forward에서 저장한 입력 shape으로 단순 reshape하여 gradient를 복원한다.

$$
\frac{\partial L}{\partial x} = \text{dout}.\text{reshape}(B, C, H, W)
$$

### 2.6. Dropout

학습 중 무작위로 뉴런을 비활성화하여 과적합을 방지하는 정규화 레이어이다. inverted dropout 방식을 사용한다.

**Forward (학습 모드)**

$$
\text{mask}[i] = \begin{cases} 0 & \text{확률 } p \\ \dfrac{1}{1-p} & \text{확률 } 1-p \end{cases}, \qquad \text{out} = x \cdot \text{mask}
$$

마스크가 살아남은 뉴런의 출력을 $\frac{1}{1-p}$로 스케일링하여, 드롭아웃 여부와 관계없이 출력의 기댓값을 유지한다(inverted dropout). 평가 모드에서는 마스크를 적용하지 않고 입력을 그대로 통과시킨다.

**Backward**

forward에서 저장한 마스크를 그대로 곱한다. 비활성화된 뉴런의 gradient는 0이 된다.

$$
\frac{\partial L}{\partial x} = \text{dout} \cdot \text{mask}
$$

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

### 3.1. im2col

```python
def im2col(x, kernel_size, stride=1, padding=0, xp=np):
    B, C, H, W = x.shape
    K = kernel_size
    out_h = (H + 2 * padding - K) // stride + 1
    out_w = (W + 2 * padding - K) // stride + 1

    if padding > 0:
        x = xp.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")

    col = xp.zeros((B, C, K, K, out_h, out_w), dtype=x.dtype)
    for kh in range(K):
        for kw in range(K):
            col[:, :, kh, kw, :, :] = x[:, :, kh:kh + stride * out_h:stride,
                                                kw:kw + stride * out_w:stride]

    return col.transpose(0, 4, 5, 1, 2, 3).reshape(B * out_h * out_w, -1), out_h, out_w
```

중간 버퍼 `col`을 `(B, C, K, K, out_h, out_w)` shape으로 먼저 구성한 뒤, `transpose`와 `reshape`으로 `(B·out_h·out_w, C·K·K)` 행렬로 변환한다. 슬라이싱 `kh:kh+stride*out_h:stride`는 커널 위치 `kh`에서 시작하여 stride 간격으로 모든 출력 위치의 값을 한 번에 추출한다.

### 3.2. col2im

```python
def col2im(col, x_shape, kernel_size, stride=1, padding=0, xp=np):
    B, C, H, W = x_shape
    K = kernel_size
    out_h = (H + 2 * padding - K) // stride + 1
    out_w = (W + 2 * padding - K) // stride + 1

    col = col.reshape(B, out_h, out_w, C, K, K).transpose(0, 3, 4, 5, 1, 2)

    H_pad, W_pad = H + 2 * padding, W + 2 * padding
    dx_padded = xp.zeros((B, C, H_pad, W_pad), dtype=col.dtype)
    for kh in range(K):
        for kw in range(K):
            dx_padded[:, :, kh:kh + stride * out_h:stride,
                           kw:kw + stride * out_w:stride] += col[:, :, kh, kw, :, :]

    return dx_padded[:, :, padding:padding + H, padding:padding + W] if padding > 0 \
           else dx_padded[:, :, :H, :W]
```

`+=` 누적 합산으로 각 패치 위치의 gradient를 원래 입력 좌표에 더한다. 동일한 입력 픽셀이 여러 패치에 등장하는 경우 모든 기여를 합산해야 정확한 gradient가 된다. 패딩 영역은 슬라이싱으로 제거한다.

### 3.3. Conv2d

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

`forward`에서 `col_x`와 `out_h`, `out_w`를 `_col_cache`에 저장하여 backward가 재계산 없이 재사용한다. `backward`에서 `dout`을 `(B·out_h·out_w, out_c)` 형태로 flatten한 뒤 chain rule로 `grad_w`, `grad_b`, `dcol_x`를 계산하고, `col2im`으로 `dx`를 복원한다. 커널 가중치는 numpy로 초기화한 뒤 `xp.asarray`로 변환하여 cupy에서도 seed 재현성을 보장한다.

### 3.4. MaxPool2d

```python
class MaxPool2d(Module):
    def forward(self, x):
        B, C, H, W = x.shape
        self._input_shape = x.shape
        col_x, out_h, out_w = im2col(x, self.kernel_size, self.stride, self.padding, xp=self.xp)
        self._out_h, self._out_w = out_h, out_w
        col_x = col_x.reshape(-1, self.kernel_size * self.kernel_size)
        self._max_indices = self.xp.argmax(col_x, axis=1)
        self._cache = col_x
        return self.xp.max(col_x, axis=1).reshape(B, out_h, out_w, C).transpose(0, 3, 1, 2)

    def backward(self, dout):
        dout_flat = dout.transpose(0, 2, 3, 1).flatten()
        dcol = self.xp.zeros_like(self._cache)
        dcol[self.xp.arange(self._max_indices.size), self._max_indices] = dout_flat
        return col2im(dcol, self._input_shape, self.kernel_size, self.stride, self.padding, xp=self.xp)
```

`im2col` 후 `(B·out_h·out_w·C, K·K)` 형태로 reshape하여 행별 최대값과 위치를 구한다. `_max_indices`는 `argmax` 결과로, backward에서 해당 위치에만 gradient를 배치하고 나머지는 0으로 유지한다. `stride` 기본값은 `kernel_size`와 같아 겹침 없이 풀링된다.

### 3.5. Flatten

```python
class Flatten(Module):
    def forward(self, x):
        self._input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self._input_shape)
```

`forward`에서 입력 shape을 `_input_shape`에 저장하고 `(B, -1)`로 reshape한다. `backward`는 저장된 shape으로 그대로 복원한다. 파라미터가 없으므로 `params`와 `grads`는 빈 리스트이다.

### 3.6. Dropout

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

`Module.training` 플래그로 학습/평가 모드를 구분한다. 학습 모드에서는 마스크를 생성하여 적용하고, 평가 모드에서는 입력을 그대로 반환한다. `Sequential.train()` / `Sequential.eval()`이 하위 레이어에 플래그를 전파하므로 별도로 각 레이어를 전환하지 않아도 된다.

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
