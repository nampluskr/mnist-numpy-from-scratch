---
tags: [docs, stage5, trainer, evaluator]
created: "2026-06-20"
updated: "2026-06-20"
---

# Trainer와 Evaluator 구현

## 1. 개요

`src/core/trainer.py`의 `Trainer`와 `src/core/evaluator.py`의 `Evaluator`는 학습과 평가 루프를 각각 캡슐화한 실행 객체이다. `Trainer.fit`은 `DataLoader`를 받아 epoch 단위 학습 루프를 실행하고, `Evaluator.evaluate`는 `DataLoader`를 받아 배치 단위 평가를 집계한다. 두 객체 모두 task를 기준으로 loss 함수와 metric 함수를 자동으로 선택하여 multiclass, binary, regression 세 가지 task를 동일한 인터페이스로 처리한다.

**목표**
- `Trainer.fit(train_loader)`으로 epoch 단위 학습 루프를 실행하고 epoch별 loss/metric 로그를 반환한다.
- `Evaluator.evaluate(test_loader)`으로 전체 test set에 대한 평균 loss와 metric을 집계한다.
- task 이름 하나로 loss, metric, gradient 함수를 자동으로 선택하는 dispatch 구조를 사용한다.

## 2. 개념

### 2.1. 학습 루프 구조

딥러닝 학습 루프는 epoch 반복과 batch 반복의 이중 구조로 이루어진다. 하나의 epoch는 전체 train set을 한 번 순회하는 단위이고, 하나의 batch는 DataLoader가 제공하는 소규모 샘플 묶음이다.

각 batch에서 실행되는 연산 순서는 다음과 같다.

```text
forward -> loss 계산 -> gradient 계산 -> backward -> optimizer.step
```

학습 루프는 이 흐름을 반복하며 epoch가 끝날 때마다 loss와 metric 평균을 집계한다. `Trainer`는 이 반복 구조를 `fit` 메서드 하나로 캡슐화하여 클라이언트 코드가 루프를 직접 작성하지 않아도 된다.

핵심 용어는 다음과 같다.

| 용어 | 의미 | 이 프로젝트에서의 역할 |
|---|---|---|
| epoch | 전체 데이터 1회 순회 | `Trainer.fit(epochs=10)` 형태로 지정 |
| batch | DataLoader가 반환하는 소규모 샘플 묶음 | `for images, targets in train_loader` 형태로 순회 |
| dispatch | task 이름으로 함수를 선택하는 구조 | `TASK_SPEC` dict로 loss/metric/grad 함수 매핑 |

### 2.2. task dispatch 구조

multiclass, binary, regression 세 task는 loss 함수, metric 함수, gradient 함수가 모두 다르다. 이를 `if/elif` 분기 대신 dict 기반 dispatch로 처리하면 새로운 task 추가 시 dict에 항목만 추가하면 된다.

task별 함수 매핑은 다음과 같다.

| task | loss | metric | grad |
|---|---|---|---|
| `multiclass` | `cross_entropy` | `accuracy` | `cross_entropy_grad` |
| `binary` | `binary_cross_entropy` | `binary_accuracy` | `binary_cross_entropy_grad` |
| `regression` | `mse` | `r2_score` | `mse_grad` |

`Trainer`와 `Evaluator`는 생성 시 task 이름을 받아 해당 함수를 `self.loss_fn`, `self.metric_fn`, `self.grad_fn`에 저장한다.

### 2.3. 평가 루프와 집계

`Evaluator`는 model을 evaluation mode로 전환한 뒤 gradient 계산 없이 forward만 수행한다. 배치별 loss와 metric 합계를 누적한 뒤 전체 샘플 수로 나누어 평균을 반환한다. batch 크기가 고르지 않을 수 있으므로 단순 배치 수 평균이 아닌 샘플 수 가중 평균을 사용한다.

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `Trainer` | 클래스 | `model`, `optimizer`, `task: str` | trainer instance | 학습 루프 실행 객체 |
| `fit` | 메서드 | `train_loader`, `epochs: int` | list of dict | epoch별 loss/metric 로그 |
| `Evaluator` | 클래스 | `model`, `task: str` | evaluator instance | 평가 루프 실행 객체 |
| `evaluate` | 메서드 | `loader` | dict | `loss`, `metric`, `num_samples` 포함 |

### 3.1. Trainer 구현

```python
TASK_SPEC = {
    "multiclass": {
        "loss": cross_entropy,
        "metric": accuracy,
        "grad": cross_entropy_grad,
    },
    "binary": {
        "loss": binary_cross_entropy,
        "metric": binary_accuracy,
        "grad": binary_cross_entropy_grad,
    },
    "regression": {
        "loss": mse,
        "metric": r2_score,
        "grad": mse_grad,
    },
}

class Trainer:
    def __init__(self, model, optimizer, task="multiclass"):
        spec = TASK_SPEC[task]
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = spec["loss"]
        self.metric_fn = spec["metric"]
        self.grad_fn = spec["grad"]

    def fit(self, train_loader, epochs=1):
        logs = []
        for epoch in range(1, epochs + 1):
            total_loss, total_metric, num_samples = 0.0, 0.0, 0
            self.model.train()
            for images, targets in train_loader:
                logits = self.model.forward(images)
                loss = self.loss_fn(logits, targets)
                metric = self.metric_fn(logits, targets)
                grad = self.grad_fn(logits, targets)
                self.model.backward(grad)
                self.optimizer.step()
                n = len(images)
                total_loss += loss * n
                total_metric += metric * n
                num_samples += n
            logs.append({
                "epoch": epoch,
                "loss": total_loss / num_samples,
                "metric": total_metric / num_samples,
            })
        return logs
```

`total_loss += loss * n`과 같이 배치 샘플 수를 곱한 합계를 누적한 뒤 전체 `num_samples`로 나누어 가중 평균을 계산한다. 이렇게 하면 마지막 batch 크기가 다를 때도 정확한 epoch 평균이 나온다.

### 3.2. Evaluator 구현

```python
class Evaluator:
    def __init__(self, model, task="multiclass"):
        spec = TASK_SPEC[task]
        self.model = model
        self.loss_fn = spec["loss"]
        self.metric_fn = spec["metric"]

    def evaluate(self, loader):
        total_loss, total_metric, num_samples = 0.0, 0.0, 0
        self.model.eval()
        for images, targets in loader:
            logits = self.model.forward(images)
            n = len(images)
            total_loss += self.loss_fn(logits, targets) * n
            total_metric += self.metric_fn(logits, targets) * n
            num_samples += n
        return {
            "loss": total_loss / num_samples,
            "metric": total_metric / num_samples,
            "num_samples": num_samples,
        }
```

`self.model.eval()`은 `Sequential`의 `training` 플래그를 `False`로 전환한다. `Dropout` 등 training/eval 동작이 다른 레이어가 이 플래그를 확인한다.

## 4. 사용법

최소 사용 예제는 다음과 같다.

```python
from src.models.mlp import MLP
from src.core.optimizers import SGD
from src.core.trainer import Trainer
from src.core.evaluator import Evaluator

model = MLP(task="multiclass", seed=42)
optimizer = SGD(model, lr=0.01)
trainer = Trainer(model, optimizer, task="multiclass")
evaluator = Evaluator(model, task="multiclass")

logs = trainer.fit(train_loader, epochs=5)
result = evaluator.evaluate(test_loader)

print(logs[-1])
print(result)
```

예상 출력은 다음과 같다.

```text
{"epoch": 5, "loss": 0.312, "metric": 0.891}
{"loss": 0.298, "metric": 0.905, "num_samples": 10000}
```

프로젝트 통합 예제는 다음과 같다. train/test DataLoader를 구성하여 학습하고 평가하는 전체 흐름이다.

```python
from src.data.mnist import MnistDataset
from src.data.dataloader import DataLoader
from src.models.mlp import MLP
from src.core.optimizers import Adam
from src.core.trainer import Trainer
from src.core.evaluator import Evaluator

task = "multiclass"
train_ds = MnistDataset(split="train", task=task)
test_ds = MnistDataset(split="test", task=task)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

model = MLP(task=task, seed=42)
optimizer = Adam(model, lr=0.001)
trainer = Trainer(model, optimizer, task=task)
evaluator = Evaluator(model, task=task)

logs = trainer.fit(train_loader, epochs=10)
result = evaluator.evaluate(test_loader)
```

## 5. 테스트

테스트 파일은 `tests/stage5/test_trainer.py`와 `tests/stage5/test_evaluator.py`이다.

```bash
conda run -n numpy_py311 pytest tests/stage5/test_trainer.py tests/stage5/test_evaluator.py -v
```

테스트 구성은 다음과 같다.

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestTrainer` | 4 | fit 반환 로그 길이, loss/metric 키 존재, epoch 후 params 변경 확인, 3 task 동작 |
| `TestEvaluator` | 3 | evaluate 반환 키 존재, num_samples 정확성, eval mode 전환 확인 |

## 6. 요약

`Trainer`는 `DataLoader`를 순회하며 forward, loss, gradient, backward, optimizer.step의 학습 루프를 epoch 단위로 실행하고 로그를 반환한다. `Evaluator`는 eval mode에서 forward와 loss/metric 집계만 수행하여 test set 성능을 반환한다. 두 객체 모두 `TASK_SPEC` dict dispatch로 task별 함수를 자동 선택하므로 클라이언트 코드에 task 분기가 없다.

다음 Phase에서는 [[phase5.3_predictor-visualizer]]을 다룬다.
