---
tags: [stage5, core, experiment]
created: 2026-06-17
updated: 2026-06-20
---

# Phase 5.6 Experiment 구현

## 1. 역할

`src/core/experiment.py`는 config를 기준으로 dataset, dataloader, model, optimizer, trainer, evaluator, predictor를 조립하는 최상위 진입점이다.
클라이언트 스크립트는 내부 구현 모듈을 직접 조립하지 않고 Experiment를 통해 학습·평가·예측 객체에 접근한다.

## 2. 구현

### 2.1. Experiment(config=None)

`config`가 없으면 `get_default_config()`를 사용한다.

| config 키 | 역할 |
|---|---|
| `task` | task 선택 (`multiclass` / `binary` / `regression`) |
| `dataset_dir` | MNIST gz 파일 경로 (없으면 default config 값 사용) |
| `batch_size` | DataLoader 배치 크기 |
| `num_epochs` | `run()` 반복 횟수 |
| `seed` | MLP 초기화 시드 |
| `lr` | SGD 학습률 (없으면 0.01) |

### 2.2. 조립 순서

```text
MnistDataset("train", task) + MnistDataset("test", task)
    → DataLoader(train, shuffle=True) + DataLoader(test, shuffle=False)
MLP(task, seed)
    → SGD(model, lr)
    → Trainer(model, optimizer, task_spec)
    → Evaluator(model, task_spec)
    → Predictor(model, task_spec)
```

### 2.3. run()

`num_epochs` 횟수만큼 학습·평가 루프를 실행하고 per-epoch 로그 목록을 반환한다.

```text
for epoch in 1..num_epochs:
    train_log = trainer.fit(train_loader)
    test_log  = evaluator.evaluate(test_loader)
    logs.append({"epoch": epoch, "train": train_log, "test": test_log})
return logs
```

### 2.4. 인터페이스

```python
from src.core.experiment import Experiment

exp = Experiment({"task": "multiclass", "dataset_dir": "/mnt/d/datasets/mnist",
                  "batch_size": 64, "num_epochs": 10, "seed": 42})
logs = exp.run()

# 예측
result = exp.predictor.predict(images)
```

## 3. 테스트

테스트 파일: `tests/stage5/test_experiment.py`

synthetic MNIST gz 파일을 생성하여 실제 MNIST 없이 전체 조립 흐름을 검증한다.

| 테스트 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestExperimentAssembly` | 6 | train/test loader, model/trainer/evaluator/predictor 인스턴스 타입 |
| `TestExperimentRun` | 18 (6항목 × 3 task) | 반환 list 타입, 길이, epoch 번호, train/test 키, 로그 키 집합 |

실행 명령:

```bash
conda run -n numpy_env pytest tests/stage5/test_experiment.py -v
```

## 4. 설계 결정

- `run()`은 epoch 루프를 직접 실행하며 per-epoch 로그를 누적한다. 여러 epoch 반복의 책임을 Experiment가 갖는다.
- Trainer와 달리 Experiment는 optimizer를 SGD로 고정한다. 옵티마이저 선택은 이후 config 확장 시 추가할 수 있다.
- `lr` 키가 config에 없을 때 `0.01`로 대체한다. 필수 키에 포함하지 않아 default config 호환성을 유지한다.
- 클라이언트 코드는 `exp.trainer`, `exp.evaluator`, `exp.predictor` 속성을 직접 참조하여 추가 제어가 가능하다.
