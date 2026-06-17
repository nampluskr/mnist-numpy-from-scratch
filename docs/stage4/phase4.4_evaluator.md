---
tags: [stage4, core, evaluator]
created: 2026-06-17
updated: 2026-06-17
---

# Phase 4.4 Evaluator 구현

## 1. 역할

`src/core/evaluator.py`는 한 epoch 단위의 평가 루프를 실행한다.
`DataLoader`에서 배치를 순회하며 forward → loss/metric을 계산하고, 가중 평균 loss/metric을 요약 dict로 반환한다.
Trainer와 달리 backward와 optimizer.step()을 수행하지 않으므로 파라미터는 변경되지 않는다.

## 2. 구현

### 2.1. task별 함수 분기

Evaluator는 `task_spec["task"]` 값에 따라 `src.nn.losses`의 손실·지표 함수를 선택한다.

| task | loss | metric |
|---|---|---|
| `multiclass` | `cross_entropy` | `accuracy` |
| `binary` | `binary_cross_entropy` | `binary_accuracy` |
| `regression` | `mse` | `r2_score` |

함수 조합은 모듈 수준 dict `_TASK_FNS`에 정의하여 `__init__`에서 한 번만 조회한다.

### 2.2. Evaluator(model, task_spec)

| 인자 | 설명 |
|---|---|
| `model` | `forward`, `params`를 제공하는 모델 인스턴스 |
| `task_spec` | `get_task_spec(task)`가 반환한 dict |

### 2.3. evaluate(test_loader)

DataLoader를 순회하며 한 pass를 실행하고 요약 dict를 반환한다.

```text
for each (x, y) in test_loader:
    logits = model.forward(x)
    loss   = loss_fn(logits, y)
    metric = metric_fn(logits, y)

return {"loss": weighted_avg_loss, "metric": weighted_avg_metric, "num_samples": N}
```

loss/metric은 배치 크기로 가중 평균하므로, 마지막 배치가 작아도 전체 평균이 정확하게 계산된다.

### 2.4. 인터페이스

```python
from src.core.evaluator import Evaluator
from src.models.mlp import MLP
from src.task import get_task_spec

model = MLP(task="multiclass", seed=0)
evaluator = Evaluator(model, get_task_spec("multiclass"))

result = evaluator.evaluate(test_loader)
print(result)  # {"loss": 0.45, "metric": 0.89, "num_samples": 10000}
```

## 3. 테스트

테스트 파일: `tests/stage4/test_evaluator.py`

`TinyModel`(선형 레이어 1개), `SimpleLoader`를 사용하여 MLP와 DataLoader 없이 독립적으로 검증한다.

| 테스트 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestEvaluatorEvaluate` | 16 (5항목 × 3 task + partial\_last\_batch) | 반환 dict 키 집합, num\_samples 정확성, loss/metric float 타입, 파라미터 불변 확인, 불완전 마지막 배치 처리 |

실행 명령:

```bash
conda run -n numpy_env pytest tests/stage4/test_evaluator.py -v
```

## 4. 설계 결정

- `evaluate`는 파라미터를 변경하지 않는다. backward/optimizer.step() 호출이 없으므로 Trainer와 달리 optimizer 인자가 없다.
- Trainer와 동일한 가중 평균 방식을 사용하여 배치 크기 불균형에도 정확한 결과를 얻는다.
- `_TASK_FNS`에서 gradient 함수는 제외한다. 평가에서는 grad 계산이 불필요하다.
