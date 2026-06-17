# test_cnn.py: CNN 레이어 및 모델 단위 테스트

import numpy as np
import pytest

from src.nn.conv import im2col, col2im, Conv2d, MaxPool2d, Flatten, Dropout
from src.models.cnn import CNN


# ---------------------------------------------------------------------------
# im2col / col2im
# ---------------------------------------------------------------------------

class TestIm2Col:
    def test_output_shape_no_padding(self):
        # B=2, C=3, H=5, W=5, K=3, stride=1, padding=0 → out=3x3
        x = np.ones((2, 3, 5, 5), dtype=np.float32)
        col, out_h, out_w = im2col(x, 3, stride=1, padding=0)
        assert col.shape == (2 * 3 * 3, 3 * 3 * 3)
        assert out_h == 3
        assert out_w == 3

    def test_output_shape_with_padding(self):
        # B=2, C=1, H=4, W=4, K=3, stride=1, padding=1 → out=4x4
        x = np.ones((2, 1, 4, 4), dtype=np.float32)
        col, out_h, out_w = im2col(x, 3, stride=1, padding=1)
        assert col.shape == (2 * 4 * 4, 1 * 3 * 3)
        assert out_h == 4
        assert out_w == 4

    def test_output_shape_stride2(self):
        # B=1, C=1, H=6, W=6, K=2, stride=2, padding=0 → out=3x3
        x = np.ones((1, 1, 6, 6), dtype=np.float32)
        col, out_h, out_w = im2col(x, 2, stride=2, padding=0)
        assert col.shape == (1 * 3 * 3, 1 * 2 * 2)
        assert out_h == 3
        assert out_w == 3

    def test_values_match_manual_extraction(self):
        # 단순 케이스: B=1, C=1, H=3, W=3, K=2, stride=1, padding=0
        x = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
        col, out_h, out_w = im2col(x, 2, stride=1, padding=0)
        # out: 2x2, col shape: (4, 4)
        assert col.shape == (4, 4)
        # 첫 번째 패치: (0,0)~(1,1) = [0,1,3,4]
        np.testing.assert_array_equal(col[0], [0, 1, 3, 4])


class TestCol2Im:
    def test_output_shape(self):
        x = np.ones((2, 3, 6, 6), dtype=np.float32)
        col, out_h, out_w = im2col(x, 3, stride=1, padding=1)
        dx = col2im(col, x.shape, 3, stride=1, padding=1)
        assert dx.shape == x.shape

    def test_roundtrip_ones(self):
        # 단일 채널 단일 배치에서 패딩 없이 각 위치 기여도 검증
        x = np.ones((1, 1, 4, 4), dtype=np.float32)
        col, out_h, out_w = im2col(x, 2, stride=1, padding=0)
        dx = col2im(col, x.shape, 2, stride=1, padding=0)
        assert dx.shape == x.shape
        # 모서리는 1번, 가장자리는 2번, 중앙은 4번 누적
        assert float(dx[0, 0, 0, 0]) == pytest.approx(1.0)
        assert float(dx[0, 0, 1, 1]) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Conv2d
# ---------------------------------------------------------------------------

class TestConv2d:
    def test_forward_shape_same_padding(self):
        layer = Conv2d(1, 8, 3, padding=1)
        x = np.random.randn(2, 1, 8, 8).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (2, 8, 8, 8)

    def test_forward_shape_valid_padding(self):
        layer = Conv2d(3, 16, 3, padding=0)
        x = np.random.randn(4, 3, 10, 10).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (4, 16, 8, 8)

    def test_backward_dx_shape(self):
        layer = Conv2d(1, 8, 3, padding=1)
        x = np.random.randn(2, 1, 8, 8).astype(np.float32)
        out = layer.forward(x)
        dout = np.ones_like(out)
        dx = layer.backward(dout)
        assert dx.shape == x.shape

    def test_backward_updates_grad_w(self):
        layer = Conv2d(1, 4, 3, padding=1, seed=0)
        x = np.random.randn(2, 1, 8, 8).astype(np.float32)
        out = layer.forward(x)
        layer.backward(np.ones_like(out))
        assert not np.all(np.asarray(layer.grad_w) == 0)

    def test_backward_updates_grad_b(self):
        layer = Conv2d(1, 4, 3, padding=1, seed=0)
        x = np.random.randn(2, 1, 8, 8).astype(np.float32)
        out = layer.forward(x)
        layer.backward(np.ones_like(out))
        assert not np.all(np.asarray(layer.grad_b) == 0)

    def test_params_grads_registered(self):
        layer = Conv2d(1, 4, 3)
        assert len(layer.params) == 2
        assert len(layer.grads) == 2

    def test_output_dtype_float32(self):
        layer = Conv2d(1, 4, 3, padding=1, seed=0)
        x = np.random.randn(2, 1, 8, 8).astype(np.float32)
        out = layer.forward(x)
        assert np.asarray(out).dtype == np.float32


# ---------------------------------------------------------------------------
# MaxPool2d
# ---------------------------------------------------------------------------

class TestMaxPool2d:
    def test_forward_shape_halves_spatial(self):
        layer = MaxPool2d(2, 2)
        x = np.random.randn(2, 4, 8, 8).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (2, 4, 4, 4)

    def test_forward_shape_odd_input(self):
        # H=7 → (7-2)//2 + 1 = 3
        layer = MaxPool2d(2, 2)
        x = np.random.randn(1, 1, 7, 7).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (1, 1, 3, 3)

    def test_backward_dx_shape(self):
        layer = MaxPool2d(2, 2)
        x = np.random.randn(2, 4, 8, 8).astype(np.float32)
        out = layer.forward(x)
        dout = np.ones_like(out)
        dx = layer.backward(dout)
        assert dx.shape == x.shape

    def test_forward_selects_max(self):
        # 2x2 영역에서 최대값만 선택되어야 함
        layer = MaxPool2d(2, 2)
        x = np.array([[[[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12],
                         [13, 14, 15, 16]]]], dtype=np.float32)
        out = layer.forward(x)
        expected = np.array([[[[6, 8], [14, 16]]]], dtype=np.float32)
        np.testing.assert_array_equal(np.asarray(out), expected)

    def test_backward_gradient_to_max_position(self):
        # gradient가 최대값 위치에만 전달되어야 함
        layer = MaxPool2d(2, 2)
        x = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        out = layer.forward(x)
        dx = layer.backward(np.ones_like(out))
        # max=4 위치 (0,0,1,1) 에만 gradient=1 전달
        assert float(dx[0, 0, 1, 1]) == pytest.approx(1.0)
        assert float(dx[0, 0, 0, 0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

class TestFlatten:
    def test_forward_shape(self):
        layer = Flatten()
        x = np.random.randn(3, 4, 7, 7).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (3, 4 * 7 * 7)

    def test_backward_restores_shape(self):
        layer = Flatten()
        x = np.random.randn(3, 4, 7, 7).astype(np.float32)
        out = layer.forward(x)
        dx = layer.backward(np.ones_like(out))
        assert dx.shape == x.shape

    def test_no_params(self):
        layer = Flatten()
        assert layer.params == []
        assert layer.grads == []


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

class TestDropout:
    def test_forward_training_applies_mask(self):
        np.random.seed(42)
        layer = Dropout(0.5)
        layer.training = True
        x = np.ones((100, 100), dtype=np.float32)
        out = layer.forward(x)
        assert np.any(out == 0.0)          # 일부 뉴런 비활성화

    def test_forward_training_scales_output(self):
        np.random.seed(0)
        layer = Dropout(0.5)
        layer.training = True
        x = np.ones((1000, 10), dtype=np.float32)
        out = layer.forward(x)
        # inverted dropout: 활성 뉴런은 1/(1-p)=2로 스케일 업
        active = out[out > 0]
        np.testing.assert_allclose(active, 2.0, atol=1e-5)

    def test_forward_eval_passthrough(self):
        layer = Dropout(0.5)
        layer.training = False
        x = np.random.randn(10, 10).astype(np.float32)
        out = layer.forward(x)
        np.testing.assert_array_equal(out, x)

    def test_backward_applies_same_mask(self):
        np.random.seed(1)
        layer = Dropout(0.5)
        layer.training = True
        x = np.ones((10, 10), dtype=np.float32)
        out = layer.forward(x)
        dout = np.ones_like(out)
        dx = layer.backward(dout)
        np.testing.assert_array_equal(dx, out)  # backward mask = forward mask

    def test_no_params(self):
        layer = Dropout(0.5)
        assert layer.params == []
        assert layer.grads == []


# ---------------------------------------------------------------------------
# CNN 모델
# ---------------------------------------------------------------------------

class TestCNNForward:
    @pytest.fixture
    def cnn(self):
        return CNN(task="multiclass", seed=42)

    def test_output_shape_multiclass(self, cnn):
        x = np.random.randn(4, 784).astype(np.float32)
        out = cnn.forward(x)
        assert out.shape == (4, 10)

    def test_output_shape_binary(self):
        model = CNN(task="binary", seed=0)
        x = np.random.randn(4, 784).astype(np.float32)
        out = model.forward(x)
        assert out.shape == (4, 1)

    def test_output_shape_regression(self):
        model = CNN(task="regression", seed=0)
        x = np.random.randn(4, 784).astype(np.float32)
        out = model.forward(x)
        assert out.shape == (4, 1)

    def test_output_dtype_float32(self, cnn):
        x = np.random.randn(2, 784).astype(np.float32)
        out = cnn.forward(x)
        assert out.dtype == np.float32

    def test_output_is_numpy(self, cnn):
        x = np.random.randn(2, 784).astype(np.float32)
        out = cnn.forward(x)
        assert isinstance(out, np.ndarray)

    def test_batch_size_1(self, cnn):
        x = np.random.randn(1, 784).astype(np.float32)
        out = cnn.forward(x)
        assert out.shape == (1, 10)


class TestCNNBackward:
    @pytest.fixture
    def cnn(self):
        return CNN(task="multiclass", seed=42)

    def test_backward_runs_without_error(self, cnn):
        x = np.random.randn(4, 784).astype(np.float32)
        out = cnn.forward(x)
        grad = np.ones_like(out)
        cnn.backward(grad)  # 예외 없이 실행

    def test_grads_updated_after_backward(self, cnn):
        x = np.random.randn(4, 784).astype(np.float32)
        out = cnn.forward(x)
        cnn.backward(np.ones_like(out))
        all_zero = all(np.all(np.asarray(g) == 0) for g in cnn.grads)
        assert not all_zero


class TestCNNParamsGrads:
    @pytest.fixture
    def cnn(self):
        return CNN(task="multiclass", seed=0)

    def test_params_is_list(self, cnn):
        assert isinstance(cnn.params, list)

    def test_grads_is_list(self, cnn):
        assert isinstance(cnn.grads, list)

    def test_params_grads_same_length(self, cnn):
        assert len(cnn.params) == len(cnn.grads)

    def test_params_shapes_match_grads(self, cnn):
        for p, g in zip(cnn.params, cnn.grads):
            assert np.asarray(p).shape == np.asarray(g).shape

    def test_params_updated_by_sgd(self, cnn):
        from src.core.optimizers import SGD
        optimizer = SGD(cnn, lr=0.01)
        x = np.random.randn(4, 784).astype(np.float32)

        params_before = [np.asarray(p).copy() for p in cnn.params]
        out = cnn.forward(x)
        cnn.backward(np.ones_like(out))
        optimizer.step()
        params_after = [np.asarray(p) for p in cnn.params]

        changed = any(
            not np.allclose(b, a) for b, a in zip(params_before, params_after)
        )
        assert changed


class TestCNNTrainEval:
    def test_eval_disables_dropout(self):
        model = CNN(task="multiclass", seed=0)
        x = np.random.randn(4, 784).astype(np.float32)

        model.eval()
        out1 = model.forward(x)
        out2 = model.forward(x)
        np.testing.assert_array_equal(out1, out2)  # eval 시 결정론적

    def test_train_enables_dropout(self):
        model = CNN(task="multiclass", seed=0)
        x = np.random.randn(4, 784).astype(np.float32)

        model.train()
        outs = [model.forward(x) for _ in range(10)]
        # training 시 dropout으로 인해 출력이 달라야 함
        all_same = all(np.allclose(outs[0], o) for o in outs[1:])
        assert not all_same

    def test_dropout_training_flag_propagated(self):
        model = CNN(task="multiclass", seed=0)
        model.eval()
        assert model.dropout.training is False
        model.train()
        assert model.dropout.training is True
