---
tags: [stage3, models, mlp]
created: 2026-06-17
updated: 2026-06-19
---

# Phase 3.5 MLP model 구현

## 1. 역할

`src/models/mlp.py`는 `src/nn/` 모듈을 조립하여 3층 MLP를 구현한다.
`forward()`는 raw logit을 반환하고, activation과 gradient 계산은 `src/nn/losses.py`가 담당한다.

## 2. 구현

### 2.1. MLP(task, seed)

`MLP`는 `Sequential`로 레이어를 조합한다. 구조는 고정이며 task는 output_dim 결정에만 영향을 준다.

```
Linear(784, 256) → Sigmoid → Linear(256, 128) → Sigmoid → Linear(128, output_dim)
```

재현성을 위해 `seed`에서 `np.random.default_rng`로 레이어별 seed를 파생하여 각 `Linear`에 전달한다.

`output_dim`은 `get_task_spec(task)["output_dim"]`에서 읽는다.

| task | output_dim |
|---|---|
| `multiclass` | 10 |
| `binary` | 1 |
| `regression` | 1 |

### 2.2. 공개 인터페이스

```python
from src.models.mlp import MLP
from src.nn.losses import cross_entropy, cross_entropy_grad

mlp = MLP(task="multiclass", seed=0)

# forward - raw logit 반환
logits = mlp.forward(x)      # x: (N, 784) float32 → (N, output_dim) float32

# backward - losses.*_grad 계산 결과를 전달
grad_out = cross_entropy_grad(logits, targets)
mlp.backward(grad_out)

# 파라미터 갱신 (SGD 예시)
lr = 0.01
for param, grad in zip(mlp.params, mlp.grads):
    param -= lr * grad
```

### 2.3. params / grads 구조

`params`와 `grads`는 `Sequential`이 수집한 참조 목록이다.

| 인덱스 | 대응 배열 | shape |
|---|---|---|
| `params[0]` / `grads[0]` | `Linear(784, 256).w` / `grad_w` | `(784, 256)` |
| `params[1]` / `grads[1]` | `Linear(784, 256).b` / `grad_b` | `(256,)` |
| `params[2]` / `grads[2]` | `Linear(256, 128).w` / `grad_w` | `(256, 128)` |
| `params[3]` / `grads[3]` | `Linear(256, 128).b` / `grad_b` | `(128,)` |
| `params[4]` / `grads[4]` | `Linear(128, output_dim).w` / `grad_w` | `(128, output_dim)` |
| `params[5]` / `grads[5]` | `Linear(128, output_dim).b` / `grad_b` | `(output_dim,)` |

`Sigmoid` 레이어는 학습 파라미터가 없으므로 목록에 포함되지 않는다.

## 3. 테스트

테스트 파일: `tests/stage3/test_mlp.py`

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestMLPInit` | 8 | task별 output_dim, 유효하지 않은 task 예외, params/grads 개수(6), 첫 weight shape, seed 재현성 |
| `TestMLPForward` | 5 | task별 output shape, dtype float32, row sum != 1 (raw logit 확인) |
| `TestMLPBackward` | 3 | backward grad_x shape, backward 후 grads 비영, grads 원소가 레이어 내부 참조 |
| `TestMLPTrainingLoop` | 1 | 20 step SGD 후 loss 감소 |

실행 명령:

```bash
conda run -n numpy_env pytest tests/stage3/test_mlp.py -v
```

Stage 3 전체 실행:

```bash
conda run -n numpy_env pytest tests/stage3/ -v
```

## 4. 설계 결정

- `MLP.forward()`는 raw logit만 반환한다. activation을 외부에서 처리하는 방식은 PyTorch 방식과 일치하며, `losses.py`가 activation을 내부 처리하므로 모델-손실 경계가 logit으로 고정된다.
- `params`와 `grads`를 property로 노출한다. `Sequential.params`를 직접 반환하므로 모델 인스턴스를 교체하지 않는 한 참조가 유지된다.
- hidden activation으로 `Sigmoid`를 선택한 것은 레거시 코드와의 일관성을 유지하기 위해서이다. Stage 4 이후 `ReLU`로 교체할 수 있다.
- `backward`의 `grad_out` 인자는 `losses.*_grad` 함수가 반환하는 `d(loss)/d(logits)` 배열이다. 모델 내부에서 activation의 역전파를 별도로 처리하지 않는 것은 logit 기반 gradient 함수가 이미 activation 미분을 접어 놓은(folded) 형태이기 때문이다.
