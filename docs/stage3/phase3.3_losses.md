---
tags: [stage3, nn, losses]
created: 2026-06-17
updated: 2026-06-17
---

# Phase 3.3 loss 및 metric 구현

## 1. 역할

`src/nn/losses.py`는 task별 손실 함수, gradient 함수, 평가 지표를 NumPy로 구현한다.
PyTorch의 `torch.nn` / `torch.nn.functional`에 대응하며, logit을 입력으로 받아 activation을 내부에서 처리한다.

## 2. 구현

### 2.1. 손실 함수와 gradient 함수 쌍

각 task에는 손실 함수와 그에 대응하는 gradient 함수가 쌍으로 제공된다.
gradient 함수는 `d(loss) / d(logits)`를 직접 계산하여 `trainer`에서 `backward`의 입력으로 전달한다.

task별 손실 함수와 gradient 함수의 관계는 다음과 같다.

| task | 손실 함수 | gradient 함수 | logit 입력 shape |
|---|---|---|---|
| `multiclass` | `cross_entropy(logits, targets)` | `cross_entropy_grad(logits, targets)` | `(N, C)` |
| `binary` | `binary_cross_entropy(logits, targets)` | `binary_cross_entropy_grad(logits, targets)` | `(N, 1)` |
| `regression` | `mse(preds, targets)` | `mse_grad(preds, targets)` | `(N, 1)` |

`targets`는 `MnistDataset`이 반환하는 task별 float32 배열이다.

### 2.2. cross_entropy / cross_entropy_grad

- `cross_entropy(logits, targets)` : logit에 softmax를 적용한 뒤 NLL(negative log-likelihood)을 계산한다. `targets`는 one-hot, shape `(N, C)`.
- `cross_entropy_grad(logits, targets)` : `(softmax(logits) - targets) / N`. softmax와 NLL의 복합 미분이 이 형태로 정리된다.

### 2.3. binary_cross_entropy / binary_cross_entropy_grad

- `binary_cross_entropy(logits, targets)` : logit에 sigmoid를 적용한 뒤 BCE를 계산한다. 수치 안정성을 위해 sigmoid 출력을 `[1e-8, 1-1e-8]`로 clip한다. `targets`는 shape `(N, 1)`.
- `binary_cross_entropy_grad(logits, targets)` : `(sigmoid(logits) - targets) / N`.

### 2.4. mse / mse_grad

- `mse(preds, targets)` : `mean((preds - targets)^2)`. regression task에서 logit이 그대로 예측값이므로 함수명과 달리 activation이 없다.
- `mse_grad(preds, targets)` : `2 * (preds - targets) / N`.

### 2.5. 평가 지표

평가 지표 함수는 logit 입력을 직접 처리한다.

| 함수 | 설명 | 비고 |
|---|---|---|
| `accuracy(logits, targets)` | `argmax` 일치율. `targets`: one-hot | softmax 불필요 - argmax는 단조 변환에 불변 |
| `binary_accuracy(logits, targets)` | `logit >= 0` 일치율. `targets`: `{0, 1}` | `sigmoid(x) >= 0.5 ↔ x >= 0` |
| `r2_score(preds, targets)` | 결정계수 `1 - SS_res / SS_tot` | `SS_tot + 1e-8`로 0 나누기 방지 |

### 2.6. 인터페이스

```python
from src.nn.losses import (
    cross_entropy, cross_entropy_grad,
    binary_cross_entropy, binary_cross_entropy_grad,
    mse, mse_grad,
    accuracy, binary_accuracy, r2_score,
)

# forward
loss = cross_entropy(logits, targets)        # scalar

# backward 입력 계산
grad_out = cross_entropy_grad(logits, targets)   # (N, C)
mlp.backward(grad_out)

# 평가
acc = accuracy(logits, targets)              # scalar
```

## 3. 테스트

테스트 파일: `tests/stage3/test_losses.py`

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestCrossEntropy` | 3 | scalar 반환, 비음수, 정답 logit 클 때 loss ≈ 0 |
| `TestCrossEntropyGrad` | 3 | grad shape, 행합 = 0, 스케일 1/N |
| `TestBinaryCrossEntropy` | 3 | scalar 반환, 비음수, 정답 logit 클 때 loss ≈ 0 |
| `TestBinaryCrossEntropyGrad` | 2 | grad shape, 부호 방향 |
| `TestMSE` | 3 | scalar 반환, 완벽 예측 시 0, 수치 검증 |
| `TestMSEGrad` | 2 | grad shape, 수치 검증 |
| `TestMetrics` | 5 | accuracy 완벽/0, binary_accuracy, r2 완벽, r2 상한 |

실행 명령:

```bash
conda run -n numpy_env pytest tests/stage3/test_losses.py -v
```

## 4. 설계 결정

- `cross_entropy`와 `binary_cross_entropy`가 logit을 입력으로 받고 activation을 내부에서 처리하는 방식은 PyTorch `CrossEntropyLoss`와 `BCEWithLogitsLoss`의 설계를 따른다. 이 방식은 수치 안정성이 높고 인터페이스를 단순하게 유지한다.
- `cross_entropy_grad`의 결과가 `(softmax - targets) / N`으로 정리되는 이유는 softmax + NLL의 복합 미분이 이 형태를 가지기 때문이다. 별도의 softmax 역전파 없이 logit에 대한 gradient를 직접 얻는다.
- `mse`는 regression task에서 logit이 예측값이므로 gradient 함수 이름을 `mse_grad`로 통일했다. 다른 task의 `*_grad`와 같은 방식으로 `trainer`에서 호출한다.
- `accuracy`에서 softmax를 생략한 것은 argmax가 단조 변환에 불변이므로 결과가 동일하기 때문이다.
