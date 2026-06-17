---
tags: [stage3, mlp, numpy]
created: 2026-06-17
updated: 2026-06-17
---

# Phase 3.1 — MLP

## 1. 목적

NumPy 전용 3층 MLP를 구현하고, manual backward + SGD update 흐름을 확인한다.

## 2. 구현 파일

| 파일 | 역할 |
|---|---|
| `src/models/mlp.py` | `MLP` 클래스 — forward, backward, update |
| `tests/stage3/test_mlp.py` | 생성, forward shape, backward/update 흐름 테스트 (14개) |

## 3. 설계 결정

### 3.1. 구조

```
784 → (W1, b1) → sigmoid → 256
    → (W2, b2) → sigmoid → 128
    → (W3, b3) → output_activation → output_dim
```

- hidden activation: `sigmoid`
- output activation: task별 분기 (`softmax` / `sigmoid` / `identity`)
- `output_dim`: `get_task_spec(task)["output_dim"]` 에서 읽음

### 3.2. 가중치 초기화

He 초기화(`sqrt(2 / fan_in)`)를 사용하고 bias는 0으로 초기화한다.

### 3.3. backward 입력 규약

`backward(grad_out)`의 `grad_out`은 loss에 대한 output 미분 `(preds - y) / batch_size` 형태를 호출부에서 계산하여 전달한다.
softmax/sigmoid의 미분은 이 형태로 이미 접힌(folded) 상태이므로 `backward` 내부에서 재계산하지 않는다.

### 3.4. 캐시 전략

`forward` 호출 시 `_cache`에 입력 및 각 층의 activation을 저장하고, `backward`에서 참조한다.
`_grads`는 `backward` 호출 후 `update`에서 소비된다.

## 4. 공개 인터페이스

```python
mlp = MLP(task="multiclass")          # task: "multiclass" | "binary" | "regression"
preds = mlp.forward(x)                # x: (N, 784) float32 → preds: (N, output_dim) float32
grad_x = mlp.backward(grad_out)       # grad_out: (N, output_dim) → grad_x: (N, 784)
mlp.update(lr=0.01)                   # SGD 1-step
```

## 5. 테스트 결과

```
14 passed in 0.42s
```

| 클래스 | 테스트 항목 |
|---|---|
| `TestMLPInit` | output_dim, invalid task, weight shapes |
| `TestMLPForward` | output shape (3 task), softmax sum=1, binary range [0,1], dtype float32 |
| `TestMLPBackwardUpdate` | backward grad_x shape, update changes weights, 20-step loss 감소 |
