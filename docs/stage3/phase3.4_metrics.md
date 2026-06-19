---
tags: [stage3, nn, metrics]
created: 2026-06-19
updated: 2026-06-19
---

# Phase 3.4 metric 함수 구현

## 1. 역할

`src/nn/metrics.py`는 task별 평가 지표를 NumPy로 구현한다.
logit을 입력으로 받아 정확도, 결정계수 등 모델의 성능을 측정한다.

## 2. 구현

### 2.1. 함수 목록

평가 지표 함수는 logit 입력을 직접 처리하며, gradient가 필요하지 않다.

| 함수 | 설명 | logit 입력 shape | 반환 shape |
|---|---|---|---|
| `accuracy(logits, targets)` | `argmax` 일치율. `targets`: one-hot | `(N, C)` | scalar |
| `binary_accuracy(logits, targets)` | `logit >= 0` 일치율. `targets`: `{0, 1}` | `(N, 1)` | scalar |
| `r2_score(preds, targets)` | 결정계수 `1 - SS_res / SS_tot` | `(N, 1)` | scalar |

### 2.2. accuracy

```python
def accuracy(logits, targets):
	"""targets: one-hot (N, C). softmax 불필요 — argmax 는 단조 변환에 불변."""
	return (logits.argmax(axis=1) == targets.argmax(axis=1)).mean()
```

- `logits`: shape `(N, C)` 출력, C는 클래스 수
- `targets`: shape `(N, C)` one-hot 인코딩
- 반환: `[0, 1]` scalar — 정답 클래스 예측 비율

**설계 노트**: softmax를 생략한 이유는 argmax가 단조 변환에 불변이므로 logit의 argmax와 softmax 확률의 argmax가 항상 같기 때문이다.

### 2.3. binary_accuracy

```python
def binary_accuracy(logits, targets):
	"""sigmoid(x) >= 0.5 ↔ x >= 0."""
	return ((logits >= 0.0) == targets.astype(bool)).mean()
```

- `logits`: shape `(N, 1)` 출력
- `targets`: shape `(N, 1)`, `{0.0, 1.0}`
- 반환: `[0, 1]` scalar — 정답 클래스 예측 비율

**설계 노트**: sigmoid 함수의 성질 `sigmoid(x) >= 0.5 ↔ x >= 0`을 이용하여 sigmoid 계산을 생략한다.

### 2.4. r2_score

```python
def r2_score(preds, targets):
	ss_res = np.sum((preds - targets) ** 2)
	ss_tot = np.sum((targets - targets.mean()) ** 2)
	return 1.0 - ss_res / (ss_tot + 1e-8)
```

- `preds`: shape `(N, 1)` 또는 `(N,)` 예측값
- `targets`: shape `(N, 1)` 또는 `(N,)` 정답값
- 반환: `[-∞, 1]` scalar — 결정계수
  - 1.0: 완벽 예측
  - 0.0: 평균값만큼만 설명
  - < 0: 평균보다 나쁜 예측

**설계 노트**: `SS_tot + 1e-8`으로 분모를 안정화하여 0으로 나누기를 방지한다.

### 2.5. 인터페이스

```python
from src.nn.metrics import accuracy, binary_accuracy, r2_score

# 평가
acc = accuracy(logits, targets)           # scalar
bin_acc = binary_accuracy(logits, targets)   # scalar
r2 = r2_score(preds, targets)             # scalar
```

## 3. 테스트

테스트 파일: `tests/stage3/test_metrics.py`

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestMetrics` | 5 | accuracy 완벽/0, binary_accuracy, r2 완벽, r2 상한 |

실행 명령:

```bash
conda run -n numpy_py311 pytest tests/stage3/test_metrics.py -v
```

## 4. 사용 예시

### 4.1. trainer에서 사용

`src/core/trainer.py`는 epoch 중 배치별로 손실과 metric을 계산한다.

```python
from src.nn.metrics import accuracy, binary_accuracy, r2_score

_TASK_FNS = {
    "multiclass": (cross_entropy, cross_entropy_grad, accuracy),
    "binary": (binary_cross_entropy, binary_cross_entropy_grad, binary_accuracy),
    "regression": (mse, mse_grad, r2_score),
}
```

### 4.2. evaluator에서 사용

`src/core/evaluator.py`는 validation/test 중 배치별로 손실과 metric을 계산한다.

```python
from src.nn.metrics import accuracy, binary_accuracy, r2_score

_TASK_FNS = {
    "multiclass": (cross_entropy, accuracy),
    "binary": (binary_cross_entropy, binary_accuracy),
    "regression": (mse, r2_score),
}
```
