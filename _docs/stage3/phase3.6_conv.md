---
tags: [docs, stage3]
created: 2026-06-17
updated: 2026-06-20
---

# Phase 3.6 conv 레이어 구현

## 1. 개요

`src/nn/conv.py`에 `im2col`/`col2im` 변환 함수와 `Conv2d`, `MaxPool2d`, `Flatten`, `Dropout` 레이어를 구현한다.
모든 레이어는 `Module`을 상속하며 `xp` 파라미터로 numpy 또는 cupy를 선택한다.

## 2. im2col / col2im

| 함수 | 입력 | 출력 |
|---|---|---|
| `im2col(x, K, stride, padding, xp)` | `(B, C, H, W)` | `(B*out_h*out_w, C*K*K)`, `out_h`, `out_w` |
| `col2im(col, x_shape, K, stride, padding, xp)` | `(B*out_h*out_w, C*K*K)` | `(B, C, H, W)` |

`col2im`은 `col.reshape(B, out_h, out_w, C, K, K)` 변환 시 두 형태 모두 동일하게 처리된다.
MaxPool2d backward가 `(B*out_h*out_w*C, K*K)` 형태로 호출하더라도 원소 배열 순서가 일치하므로 올바른 결과를 반환한다.

## 3. CNN 레이어

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

## 4. 테스트 결과

```
pytest tests/stage3/test_cnn.py -v
42 passed
```

| 테스트 클래스 | 항목 수 | 검증 내용 |
|---|---|---|
| `TestIm2Col` | 4 | shape, padding, stride, 값 검증 |
| `TestCol2Im` | 2 | shape, 역변환 누적 검증 |
| `TestConv2d` | 7 | forward/backward shape, grad 갱신, params 등록, dtype |
| `TestMaxPool2d` | 5 | shape, max 선택, gradient 전달 위치 |
| `TestFlatten` | 3 | forward/backward shape, params 없음 |
| `TestDropout` | 5 | 마스크 적용, 스케일 업, eval 통과, backward mask |

## 5. 실행 명령

```bash
pytest tests/stage3/test_cnn.py -v
```
