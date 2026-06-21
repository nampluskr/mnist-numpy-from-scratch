---
tags: [docs, stage5, trainer, evaluator]
created: "2026-06-20"
updated: "2026-06-21"
---

# Trainer와 Evaluator 구현

## 1. 개요

`src/core/trainer.py`의 `Trainer`와 `src/core/evaluator.py`의 `Evaluator`는 학습과 평가 루프를 각각 캡슐화한 실행 객체이다. `Trainer.fit`은 `Dataloader`를 받아 epoch 단위 학습 루프를 실행하고, `Evaluator.evaluate`는 `Dataloader`를 받아 배치 단위 평가를 집계한다. 두 객체 모두 task를 기준으로 loss 함수와 metric 함수를 자동으로 선택하여 multiclass, binary, regression 세 가지 task를 동일한 인터페이스로 처리한다.

**목표**
- `Trainer.fit(train_loader)`으로 epoch 단위 학습 루프를 실행하고 epoch별 loss/metric 로그를 반환한다.
- `Evaluator.evaluate(test_loader)`으로 전체 test set에 대한 평균 loss와 metric을 집계한다.
- task 이름 하나로 loss, metric, gradient 함수를 자동으로 선택하는 dispatch 구조를 사용한다.

## 2. 개념

### 2.1. 학습 루프 구조

딥러닝 학습은 전체 데이터셋을 여러 번 반복 순회하며 파라미터를 점진적으로 개선하는 과정이다. 학습 루프는 epoch 반복과 batch 반복의 이중 구조로 이루어진다.

- **epoch**: 전체 train set을 한 번 순회하는 단위. epoch가 끝날 때마다 loss/metric을 집계하여 학습 진행을 모니터링한다.
- **batch**: `Dataloader`가 한 번의 iteration에서 반환하는 소규모 샘플 묶음. 하나의 epoch는 `ceil(N / batch_size)`번의 batch iteration으로 구성된다.

각 batch에서 실행되는 연산 순서는 다음과 같다.

```text
forward → loss 계산 → gradient 계산 → backward → optimizer.step
```

이 흐름 하나가 파라미터 업데이트 한 스텝이다. `Trainer.fit`은 이 스텝을 한 epoch에 걸쳐 반복하고 평균 loss/metric을 반환한다.

### 2.2. 배치 학습 한 스텝 상세

한 batch $(x, y)$에서 파라미터 $\theta$를 업데이트하는 과정을 수식으로 풀어쓰면 다음과 같다.

**Step 1. Forward**

$$
\hat{y} = f_\theta(x), \quad x \in \mathbb{R}^{N \times D},\ \hat{y} \in \mathbb{R}^{N \times C}
$$

모델이 입력 배치 $x$를 받아 logit $\hat{y}$를 출력한다. $N$은 배치 크기, $D$는 입력 차원, $C$는 출력 차원이다.

**Step 2. Loss 계산**

$$
L = \frac{1}{N} \sum_{i=1}^{N} \ell(\hat{y}_i,\ y_i)
$$

task에 맞는 loss 함수 $\ell$로 배치 평균 loss $L$을 계산한다.

**Step 3. Gradient 계산**

$$
g = \frac{\partial L}{\partial \hat{y}} \in \mathbb{R}^{N \times C}
$$

loss의 logit에 대한 편미분을 계산한다. 이 프로젝트에서는 softmax/sigmoid를 loss 함수 내부에서 처리하므로, gradient 함수는 activation까지 포함한 합성 미분을 반환한다.

| task | 합성 gradient $\frac{\partial L}{\partial \hat{y}}$ |
|---|---|
| `multiclass` | $(\text{softmax}(\hat{y}) - y) / N$ |
| `binary` | $(\text{sigmoid}(\hat{y}) - y) / N$ |
| `regression` | $2(\hat{y} - y) / N$ |

**Step 4. Backward**

$$
\frac{\partial L}{\partial \theta} = \text{chain rule로 } g \text{를 모델 레이어를 역순으로 전파}
$$

`model.backward(grad)`는 $g$를 입력으로 받아 각 레이어의 chain rule을 역순으로 적용한다. 각 `Linear` 레이어는 `grad_w`, `grad_b`를 in-place로 업데이트한다.

**Step 5. Optimizer step**

$$
\theta \leftarrow \theta - \eta \cdot \frac{\partial L}{\partial \theta}
$$

`optimizer.step()`이 `model.params`/`model.grads`를 참조하여 in-place 업데이트한다.

### 2.3. 샘플 수 가중 평균 집계

한 epoch에서 배치별 loss/metric을 단순 평균하면 마지막 batch 크기가 다를 때 오차가 생긴다. `Trainer`와 `Evaluator`는 샘플 수 가중 합계를 누적한 뒤 전체 샘플 수로 나누는 가중 평균을 사용한다.

$$
\bar{L}_{\text{epoch}} = \frac{\displaystyle\sum_{b} L_b \cdot n_b}{\displaystyle\sum_{b} n_b}
$$

$L_b$는 배치 $b$의 loss, $n_b$는 배치 $b$의 샘플 수이다. 마지막 배치가 `batch_size`보다 작아도 정확한 epoch 평균이 보장된다.

구현에서는 누적 변수로 표현된다.

```text
total_loss += loss * n
total_metric += metric * n
total_samples += n
epoch_loss = total_loss / total_samples
```

### 2.4. task dispatch 구조

multiclass, binary, regression 세 task는 loss 함수, metric 함수, gradient 함수가 모두 다르다. `if/elif` 분기 대신 dict 기반 dispatch를 사용하면 `Trainer`와 `Evaluator` 내부에 task 분기가 없고, 새로운 task 추가 시 dict에 항목만 추가하면 된다.

```text
task 이름 → _TASK_FNS[task] → (loss_fn, grad_fn, metric_fn)
```

task별 함수 매핑은 다음과 같다.

| task | loss_fn | grad_fn | metric_fn |
|---|---|---|---|
| `multiclass` | `cross_entropy` | `cross_entropy_grad` | `accuracy` |
| `binary` | `binary_cross_entropy` | `binary_cross_entropy_grad` | `binary_accuracy` |
| `regression` | `mse` | `mse_grad` | `r2_score` |

`Trainer`와 `Evaluator`는 `task_spec["task"]` 키로 dict에서 함수를 꺼내 `self.loss_fn`, `self.grad_fn`, `self.metric_fn`에 저장한다. 이후 루프에서는 task를 전혀 참조하지 않는다.

### 2.5. Trainer와 Evaluator의 역할 분리

`Trainer`와 `Evaluator`는 동일한 task dispatch 구조를 공유하지만 역할이 다르다.

| 항목 | Trainer | Evaluator |
|---|---|---|
| 목적 | 파라미터 업데이트 | 모델 성능 측정 |
| gradient 계산 | 필요 (`grad_fn` 포함) | 불필요 (`grad_fn` 없음) |
| backward | `model.backward(grad)` 호출 | 호출하지 않음 |
| optimizer | `optimizer.step()` 호출 | 사용하지 않음 |
| Dataloader | train set | test/validation set |
| 반환값 | `loss`, `metric`, `num_samples` | `loss`, `metric`, `num_samples` |

`Evaluator`는 `grad_fn`을 보유하지 않는다. forward와 loss/metric 집계만 수행하므로 backward 경로가 없어 파라미터가 변경되지 않는다.

핵심 용어는 다음과 같다.

| 용어 | 의미 | 이 프로젝트에서의 역할 |
|---|---|---|
| epoch | 전체 데이터 1회 순회 | 클라이언트가 루프를 돌며 `Trainer.fit` 1회 = 1 epoch |
| batch | `Dataloader`가 반환하는 소규모 샘플 묶음 | `for x, y in train_loader` 형태로 순회 |
| dispatch | task 이름으로 함수를 선택하는 구조 | `_TASK_FNS` dict로 loss/grad/metric 함수 매핑 |
| 가중 평균 | 샘플 수 기반 집계 | `total_loss += loss * n; epoch_loss = total_loss / total_samples` |
| task_spec | task 정보를 담은 dict | `task_spec["task"]`로 `_TASK_FNS` 조회 |

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

프로젝트 통합 예제는 다음과 같다. train/test `Dataloader`를 구성하여 학습하고 평가하는 전체 흐름이다.

```python
from src.data.mnist import MNISTDataset
from src.data.dataloader import Dataloader
from src.models.mlp import MLP
from src.core.optimizers import Adam
from src.core.trainer import Trainer
from src.core.evaluator import Evaluator

task = "multiclass"
train_ds = MNISTDataset(split="train", task=task)
test_ds = MNISTDataset(split="test", task=task)
train_loader = Dataloader(train_ds, batch_size=128, shuffle=True)
test_loader = Dataloader(test_ds, batch_size=256, shuffle=False)

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

`Trainer`는 `Dataloader`를 순회하며 forward, loss, gradient, backward, optimizer.step의 학습 루프를 epoch 단위로 실행하고 로그를 반환한다. `Evaluator`는 eval mode에서 forward와 loss/metric 집계만 수행하여 test set 성능을 반환한다. 두 객체 모두 `TASK_SPEC` dict dispatch로 task별 함수를 자동 선택하므로 클라이언트 코드에 task 분기가 없다.

다음 Phase에서는 [[phase5.3_predictor-visualizer]]을 다룬다.
