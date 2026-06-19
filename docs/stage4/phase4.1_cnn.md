---
tags: [docs, stage4]
created: 2026-06-17
updated: 2026-06-20
---

# Phase 4.1 CNN model 구현

## 1. 개요

CuPy 기반 CNN 모델을 구현한다. MLP와 동일한 외부 인터페이스(`forward`, `backward`, `params`, `grads`)를 유지하여 기존 `Trainer`, `Evaluator`, `Experiment` 코드를 수정 없이 재사용한다.

CuPy 미설치 환경에서는 NumPy로 자동 fallback하여 동일한 코드가 CPU에서도 실행된다.

## 2. 구현 파일

| 파일 | 역할 |
|---|---|
| `src/nn/layers.py` | `Module` 기반 클래스에 `training`, `train()`, `eval()` 추가 |
| `src/nn/conv.py` | `im2col`/`col2im` + `Conv2d`, `MaxPool2d`, `Flatten`, `Dropout` |
| `src/models/cnn.py` | CuPy 기반 CNN 모델 |
| `tests/stage4/test_cnn.py` | 42개 단위 테스트 |

## 3. 수정: `src/nn/layers.py`

기존 `Module` 기반 클래스에 `training` 상태와 `train()` / `eval()` 메서드를 추가했다.
`Sequential`은 `train()` / `eval()` 호출 시 하위 레이어로 상태를 전파한다.

```python
class Module:
    def __init__(self):
        self.params = []
        self.grads = []
        self.training = True  # 추가

    def train(self):          # 추가
        self.training = True

    def eval(self):           # 추가
        self.training = False

class Sequential(Module):
    def train(self):          # 추가 — 하위 레이어 전파
        self.training = True
        for layer in self.layers:
            layer.train()

    def eval(self):           # 추가 — 하위 레이어 전파
        self.training = False
        for layer in self.layers:
            layer.eval()
```

기존 `Linear`, `Sigmoid`, `ReLU`, `Sequential`의 forward/backward 코드는 변경하지 않는다.

## 4. 신규: `src/nn/conv.py`

### 4.1. im2col / col2im

| 함수 | 입력 | 출력 |
|---|---|---|
| `im2col(x, K, stride, padding, xp)` | `(B, C, H, W)` | `(B*out_h*out_w, C*K*K)`, `out_h`, `out_w` |
| `col2im(col, x_shape, K, stride, padding, xp)` | `(B*out_h*out_w, C*K*K)` 또는 `(B*out_h*out_w*C, K*K)` | `(B, C, H, W)` |

`col2im`은 `col.reshape(B, out_h, out_w, C, K, K)` 변환 시 두 형태 모두 동일하게 처리된다.
MaxPool2d backward가 `(B*out_h*out_w*C, K*K)` 형태로 호출하더라도 원소 배열 순서가 일치하므로 올바른 결과를 반환한다.

### 4.2. CNN 레이어

모든 레이어는 `Module`을 상속하며 `xp` 파라미터로 numpy 또는 cupy를 선택한다.

| 클래스 | 파라미터 | 비고 |
|---|---|---|
| `Conv2d(in_c, out_c, K, stride, padding, seed, xp)` | w `(out_c, in_c, K, K)`, b `(out_c,)` | He init; numpy로 생성 후 xp 변환 |
| `MaxPool2d(K, stride, padding, xp)` | 없음 | im2col 기반 max pooling |
| `Flatten()` | 없음 | reshape만 사용; xp 불필요 |
| `Dropout(p)` | 없음 | numpy random mask; training 상태 분기 |

**Conv2d 가중치 초기화**: numpy seed로 생성 후 `xp.asarray()`로 변환하여 CuPy/NumPy 모두 동일한 seed 재현성을 확보한다.

```python
rng = np.random.default_rng(seed)
scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))  # He init
w_np = rng.standard_normal(...).astype(np.float32) * scale
self.w = xp.asarray(w_np)
```

**Dropout**: inverted dropout 방식으로 학습 시 활성 뉴런을 `1 / (1 - p)`로 스케일 업한다.

## 5. 신규: `src/models/cnn.py`

### 5.1. 모델 구조

```
Input (N, 784)
  ↓ reshape to (N, 1, 28, 28)
Conv2d(1→32, K=3, pad=1)  →  (N, 32, 28, 28)
ReLU
MaxPool2d(2, 2)            →  (N, 32, 14, 14)
Conv2d(32→64, K=3, pad=1) →  (N, 64, 14, 14)
ReLU
MaxPool2d(2, 2)            →  (N, 64, 7, 7)
Flatten                    →  (N, 3136)
  ↓ CuPy → numpy 변환
Linear(3136→256)
ReLU
Dropout(0.5)
Linear(256→output_dim)
Output (N, output_dim)     ← numpy float32
```

### 5.2. CuPy/NumPy 경계 처리

MnistDataset은 `(N, 784)` numpy 배열을 반환한다. CNN 내부에서 CuPy 변환과 역변환을 처리하므로 DataLoader 및 손실 함수는 수정이 필요 없다.

| 지점 | 변환 |
|---|---|
| `forward()` 시작 | `xp.asarray(x).reshape(-1, 1, 28, 28)` — numpy → CuPy |
| `Flatten` 직후 | `np.asarray(x_xp)` — CuPy → numpy |
| `backward()` 시작 | `xp.asarray(grad_out)` — numpy → CuPy |

`conv_net`의 params/grads는 CuPy 배열, `fc_net`의 params/grads는 numpy 배열이다.
`SGD.step()`은 `param -= lr * grad`로 in-place 갱신하며, 두 타입 모두 지원한다.

### 5.3. CuPy fallback

```python
try:
    import cupy as _xp
except ImportError:
    import numpy as _xp
```

CuPy 미설치 환경에서 동일 코드가 NumPy로 실행된다.

## 6. 테스트 결과

```
pytest tests/stage4/test_cnn.py -v
42 passed in 1.02s
```

| 테스트 클래스 | 항목 수 | 검증 내용 |
|---|---|---|
| `TestIm2Col` | 4 | shape, padding, stride, 값 검증 |
| `TestCol2Im` | 2 | shape, 역변환 누적 검증 |
| `TestConv2d` | 7 | forward/backward shape, grad 갱신, params 등록, dtype |
| `TestMaxPool2d` | 5 | shape, max 선택, gradient 전달 위치 |
| `TestFlatten` | 3 | forward/backward shape, params 없음 |
| `TestDropout` | 5 | 마스크 적용, 스케일 업, eval 통과, backward mask |
| `TestCNNForward` | 6 | 3종 task shape, dtype, numpy 반환, batch=1 |
| `TestCNNBackward` | 2 | 실행 무오류, grad 갱신 확인 |
| `TestCNNParamsGrads` | 5 | list, 길이, shape 일치, SGD 갱신 |
| `TestCNNTrainEval` | 3 | eval 결정론적, train 확률적, flag 전파 |

## 7. 실행 명령

```bash
pytest tests/stage4/test_cnn.py -v
```
