# conv.py: CNN 레이어 모듈 (Conv2d, MaxPool2d, Flatten, Dropout) + im2col/col2im

import numpy as np
from src.nn.layers import Module


def im2col(x, kernel_size, stride=1, padding=0, xp=np):
    """(B, C, H, W) → (B*out_h*out_w, C*K*K) 변환 (convolution 전처리).

    Returns:
        col: (B*out_h*out_w, C*K*K)
        out_h, out_w: 출력 공간 크기
    """
    B, C, H, W = x.shape
    K = kernel_size
    out_h = (H + 2 * padding - K) // stride + 1
    out_w = (W + 2 * padding - K) // stride + 1

    if padding > 0:
        x = xp.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                    mode="constant")

    col = xp.zeros((B, C, K, K, out_h, out_w), dtype=x.dtype)
    for kh in range(K):
        kh_max = kh + stride * out_h
        for kw in range(K):
            kw_max = kw + stride * out_w
            col[:, :, kh, kw, :, :] = x[:, :, kh:kh_max:stride, kw:kw_max:stride]

    # (B, C, K, K, out_h, out_w) → (B, out_h, out_w, C, K, K) → (B*out_h*out_w, C*K*K)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(B * out_h * out_w, -1)
    return col, out_h, out_w


def col2im(col, x_shape, kernel_size, stride=1, padding=0, xp=np):
    """(B*out_h*out_w, C*K*K) 또는 (B*out_h*out_w*C, K*K) → (B, C, H, W) 역변환.

    두 형태 모두 col.reshape(B, out_h, out_w, C, K, K) 로 동일하게 처리된다.
    """
    B, C, H, W = x_shape
    K = kernel_size
    out_h = (H + 2 * padding - K) // stride + 1
    out_w = (W + 2 * padding - K) // stride + 1

    col = col.reshape(B, out_h, out_w, C, K, K).transpose(0, 3, 4, 5, 1, 2)
    # shape: (B, C, K, K, out_h, out_w)

    H_pad, W_pad = H + 2 * padding, W + 2 * padding
    dx_padded = xp.zeros((B, C, H_pad, W_pad), dtype=col.dtype)
    for kh in range(K):
        kh_max = kh + stride * out_h
        for kw in range(K):
            kw_max = kw + stride * out_w
            dx_padded[:, :, kh:kh_max:stride, kw:kw_max:stride] += col[:, :, kh, kw, :, :]

    if padding > 0:
        return dx_padded[:, :, padding:padding + H, padding:padding + W]
    return dx_padded[:, :, :H, :W]


class Conv2d(Module):
    """2D 합성곱 레이어 (xp-agnostic: numpy 또는 cupy 지정 가능)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 seed=None, xp=np):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.xp = xp

        # He init — numpy로 생성 후 xp로 변환 (seed 재현성 확보)
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        w_np = (rng.standard_normal(
            (out_channels, in_channels, kernel_size, kernel_size)
        ) * scale).astype(np.float32)
        self.w = xp.asarray(w_np)
        self.b = xp.zeros(out_channels, dtype=np.float32)
        self.grad_w = xp.zeros_like(self.w)
        self.grad_b = xp.zeros_like(self.b)

        self.params = [self.w, self.b]
        self.grads = [self.grad_w, self.grad_b]
        self._x = None
        self._col_cache = None  # (col_x, out_h, out_w)

    def forward(self, x):
        B, C, H, W = x.shape
        self._x = x
        col_x, out_h, out_w = im2col(x, self.kernel_size, self.stride, self.padding,
                                     xp=self.xp)
        self._col_cache = (col_x, out_h, out_w)
        col_w = self.w.reshape(self.out_channels, -1)  # (out_c, in_c*K*K)
        out = col_x @ col_w.T + self.b               # (B*out_h*out_w, out_c)
        out = out.reshape(B, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        col_x, out_h, out_w = self._col_cache
        B = self._x.shape[0]
        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        self.grad_b[...] = dout_flat.sum(axis=0)
        self.grad_w[...] = (dout_flat.T @ col_x).reshape(self.grad_w.shape)
        col_w = self.w.reshape(self.out_channels, -1)
        dcol_x = dout_flat @ col_w
        dx = col2im(dcol_x, self._x.shape, self.kernel_size, self.stride, self.padding,
                    xp=self.xp)
        return dx


class MaxPool2d(Module):
    """2D 최대 풀링 레이어 (xp-agnostic)."""

    def __init__(self, kernel_size, stride=None, padding=0, xp=np):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.xp = xp
        self._cache = None   # reshaped col (B*out_h*out_w*C, K*K)
        self._max_indices = None
        self._input_shape = None
        self._out_h = None
        self._out_w = None

    def forward(self, x):
        B, C, H, W = x.shape
        self._input_shape = x.shape
        col_x, out_h, out_w = im2col(x, self.kernel_size, self.stride, self.padding,
                                     xp=self.xp)
        self._out_h, self._out_w = out_h, out_w
        col_x = col_x.reshape(-1, self.kernel_size * self.kernel_size)
        # shape: (B*out_h*out_w*C, K*K)
        self._max_indices = self.xp.argmax(col_x, axis=1)
        self._cache = col_x
        out = self.xp.max(col_x, axis=1).reshape(B, out_h, out_w, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        B, C, out_h, out_w = dout.shape
        dout_flat = dout.transpose(0, 2, 3, 1).flatten()
        dcol = self.xp.zeros_like(self._cache)
        dcol[self.xp.arange(self._max_indices.size), self._max_indices] = dout_flat
        dx = col2im(dcol, self._input_shape, self.kernel_size, self.stride, self.padding,
                    xp=self.xp)
        return dx


class Flatten(Module):
    """4D (B, C, H, W) → 2D (B, C*H*W) 변환 레이어."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        self._input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self._input_shape)


class Dropout(Module):
    """Dropout 레이어 — training 시 마스크 적용, eval 시 통과."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self._mask = None

    def forward(self, x):
        if self.training:
            self._mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype) / (1.0 - self.p)
            return x * self._mask
        return x

    def backward(self, dout):
        if self.training:
            return dout * self._mask
        return dout
