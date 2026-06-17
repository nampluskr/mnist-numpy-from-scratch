---
tags: [stage4, core, trainer]
created: 2026-06-17
updated: 2026-06-17
---

# Phase 4.3 Trainer 구현: 학습 루프, DataLoader 수신, fit 인터페이스

## 1. 역할

`src/core/trainer.py`는 한 epoch 단위의 학습 루프를 실행한다.
`DataLoader`에서 배치를 순회하며 forward → loss/grad → backward → optimizer step을 수행하고, 가중 평균 loss/metric을 요약 dict로 반환한다.

## 2. 구현

### 2.1. task별 함수 분기

Trainer는 `task_spec["task"]` 값에 따라 `src.nn.losses`의 손실·gradient·지표 함수를 선택한다.

| task | loss | grad | metric |
|---|---|---|---|
| `multiclass` | `cross_entropy` | `cross_entropy_grad` | `accuracy` |
| `binary` | `binary_cross_entropy` | `binary_cross_entropy_grad` | `binary_accuracy` |
| `regression` | `mse` | `mse_grad` | `r2_score` |

함수 조합은 모듈 수준 dict `_TASK_FNS`에 정의하여 `__init__`에서 한 번만 조회한다.

### 2.2. Trainer(model, optimizer, task_spec)

| 인자 | 설명 |
|---|---|
| `model` | `forward`, `backward`, `params`, `grads`를 제공하는 모델 인스턴스 |
| `optimizer` | `step()`을 제공하는 옵티마이저 인스턴스 |
| `task_spec` | `get_task_spec(task)`가 반환한 dict |

### 2.3. fit(train_loader)

DataLoader를 순회하며 한 epoch를 실행하고 요약 dict를 반환한다.

```text
for each (x, y) in train_loader:
    logits = model.forward(x)
    loss   = loss_fn(logits, y)
    metric = metric_fn(logits, y)
    grad   = grad_fn(logits, y)
    model.backward(grad)
    optimizer.step()

return {"loss": weighted_avg_loss, "metric": weighted_avg_metric, "num_samples": N}
```

loss/metric은 배치 크기로 가중 평균하므로, 마지막 배치가 작아도 전체 평균이 정확하게 계산된다.

### 2.4. 인터페이스

```python
from src.core.trainer import Trainer
from src.core.optimizers import SGD
from src.models.mlp import MLP
from src.task import get_task_spec

model = MLP(task="multiclass", seed=0)
optimizer = SGD(model, lr=0.01)
trainer = Trainer(model, optimizer, get_task_spec("multiclass"))

log = trainer.fit(train_loader)
print(log)  # {"loss": 2.31, "metric": 0.12, "num_samples": 60000}
```

## 3. 테스트

테스트 파일: `tests/stage4/test_trainer.py`

`TinyModel`(선형 레이어 1개), `NoOpOptimizer`, `SimpleLoader`를 사용하여 MLP와 DataLoader 없이 독립적으로 검증한다.

| 테스트 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestTrainerFit` | 16 (5항목 × 3 task + partial\_last\_batch) | 반환 dict 키 집합, num\_samples 정확성, loss/metric float 타입, SGD로 파라미터 갱신 확인, 불완전 마지막 배치 처리 |

실행 명령:

```bash
conda run -n numpy_env pytest tests/stage4/test_trainer.py -v
```

## 4. 설계 결정

- `fit`은 1 epoch만 실행한다. 여러 epoch 반복은 `Experiment` 또는 클라이언트 스크립트가 담당한다.
- loss/metric을 `float(loss) * n`으로 누적한 뒤 `num_samples`로 나눈다. 배치 크기가 균일하지 않아도 올바른 가중 평균을 얻는다.
- `mse`와 `mse_grad`는 원래 `preds`를 받지만, regression에서는 identity activation이므로 `logits == preds`가 성립한다. 따라서 다른 task와 동일하게 `logits`를 전달한다.
- `_TASK_FNS` dict를 모듈 수준에 두어 인스턴스 생성 시 매번 import를 반복하지 않도록 한다.
