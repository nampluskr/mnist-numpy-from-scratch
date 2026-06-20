---
tags: [stage3, nn, activations]
created: 2026-06-17
updated: 2026-06-17
---

# Phase 3.1 activation 구현

## 1. 역할

`src/nn/activations.py`는 순전파 전용 활성화 함수 4종을 NumPy 배열 연산으로 구현한다.
PyTorch의 `torch.nn.functional`에 대응하는 함수형 인터페이스로, 레이어 모듈과 손실 함수에서 내부적으로 호출한다.

## 2. 구현

### 2.1. 공개 함수 목록

각 함수는 `np.ndarray`를 입력으로 받아 같은 shape의 `np.ndarray`를 반환한다.

| 함수 | 입력 | 출력 | 비고 |
|---|---|---|---|
| `sigmoid(x)` | `(...)` | `(...)` | 수치 안정 분기 처리 |
| `softmax(x)` | `(N, C)` | `(N, C)` | max subtraction으로 오버플로 방지 |
| `relu(x)` | `(...)` | `(...)` | `np.maximum(0, x)` |
| `identity(x)` | `(...)` | `(...)` | 입력을 그대로 반환 |

### 2.2. 수치 안정성 처리

`sigmoid`는 큰 음수 입력에서 `exp(-x)`가 오버플로하는 문제를 방지하기 위해 부호에 따라 두 가지 등가 수식으로 분기한다.

```python
# x >= 0: 표준 수식
out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))

# x < 0: 오버플로 방지 등가 수식
out[~pos] = np.exp(x[~pos]) / (1.0 + np.exp(x[~pos]))
```

`softmax`는 각 행에서 최대값을 빼고 지수를 계산하여 큰 logit에서 `exp` 오버플로가 발생하지 않도록 한다.

### 2.3. 사용 위치

활성화 함수는 두 곳에서 호출된다.

- `src/nn/layers.py` - `Sigmoid.forward`가 내부적으로 `sigmoid`를 호출한다.
- `src/nn/losses.py` - `cross_entropy`와 `cross_entropy_grad`가 `softmax`를, `binary_cross_entropy`와 `binary_cross_entropy_grad`가 `sigmoid`를 내부적으로 호출한다.

## 3. 테스트

테스트 파일: `tests/stage3/test_activations.py`

소규모 synthetic 배열만 사용하며 MNIST 데이터 의존성이 없다.

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestSigmoid` | 5 | shape 보존, 범위 (0, 1), 중점값 0.5, 대칭성, 큰 음수 유한값 |
| `TestSoftmax` | 4 | shape 보존, 행합 = 1, 모든 값 양수, 큰 logit 수치 안정성 |
| `TestReLU` | 3 | shape 보존, 음수 → 0, 양수 불변 |
| `TestIdentity` | 1 | 입력과 출력 배열 동일 |

실행 명령:

```bash
conda run -n numpy_env pytest tests/stage3/test_activations.py -v
```

## 4. 설계 결정

- 활성화 함수는 순전파 전용이다. backward 미분은 `layers.py`의 각 레이어 모듈 내부에서 처리한다.
- `losses.py`의 cross-entropy 계열 함수는 activation을 내부 처리하므로 호출부에서 별도로 activation을 적용하지 않는다.
- `identity`는 regression task의 output layer에서 "활성화 없음"을 명시적으로 표현하기 위한 자리표시자 함수이다.
