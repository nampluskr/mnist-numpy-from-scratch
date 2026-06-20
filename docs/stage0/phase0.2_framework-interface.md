---
tags: [docs, stage0, framework-interface]
created: "2026-06-20"
updated: "2026-06-20"
---

# 후속 프레임워크 공통 인터페이스 규약

## 1. 개요

이 프로젝트는 `numpy -> pytorch -> tensorflow -> jax` 순서로 이어질 딥러닝 프레임워크 비교 시리즈의 첫 번째 기준 구현이다. 후속 프로젝트가 동일한 문제, 동일한 프로젝트 구조, 동일한 CLI 사용법을 유지하려면 공개 함수명, 클래스명, 입력·출력 규약을 이 프로젝트에서 먼저 확정해야 한다. 이 문서는 Stage 2부터 Stage 6까지 구현에서 사용할 공통 진입점과 입력·출력 규약을 정의한다.

**목표**
- 후속 프레임워크 프로젝트와 공유할 공개 함수명·클래스명을 확정한다.
- 각 진입점의 입력·출력 타입과 shape 규약을 확정한다.
- task별 target 변환, loss, metric, 예측 후처리 규약을 단일 문서로 정리한다.

## 2. 개념

### 2.1. 인터페이스 통일의 필요성

후속 프레임워크 프로젝트에서는 `src/nn/` 하위 구현만 프레임워크 API로 교체된다. `src/data/`, `src/models/`의 최상위 구조, `src/core/`의 실행 객체 인터페이스, `scripts/`의 CLI 형식은 동일하게 유지한다. 인터페이스가 통일되어 있으면 프레임워크별 구현 차이를 직접 비교할 수 있다.

비교 기준이 되는 항목은 다음과 같다.

| 비교 항목 | 동일하게 유지 | 프레임워크마다 교체 |
|---|---|---|
| 파일 구조 | `src/data/`, `src/models/`, `src/core/` | `src/nn/` 내부 구현 |
| 함수명·클래스명 | `load_mnist`, `MnistDataset`, `DataLoader`, `MLP` 등 | 없음 |
| CLI 인터페이스 | `--task`, `--model`, `--epochs`, `--lr` 인자 | 없음 |
| 데이터 파이프라인 | `(N, 784)` float32 입력, task별 target 형식 | 없음 |

### 2.2. task 규약 중앙 관리

task별로 다른 output_dim, loss, metric, 예측 후처리는 `src/data/mnist.py`의 `get_task_spec(task)` 함수로 통합 관리한다. `Trainer`, `Evaluator`, `Predictor`는 task spec dict를 수신하여 task별 분기를 처리하므로 코드 중복이 없다.

## 3. 구현

### 3.1. 공개 진입점과 입력·출력 규약

공개 진입점 전체 목록과 규약은 아래와 같다.

| 파일 | 진입점 | 입력 | 출력 |
|---|---|---|---|
| `src/data/mnist.py` | `load_mnist(split)` | `split: str` | `(images, labels)` - `(N,28,28)` uint8, `(N,)` uint8 |
| `src/data/mnist.py` | `get_task_spec(task)` | `task: str` | `dict` |
| `src/data/mnist.py` | `transform_targets(labels, task)` | `labels: np.ndarray`, `task: str` | `np.ndarray` |
| `src/data/mnist.py` | `MnistDataset` | `split: str`, `task: str` | dataset instance |
| `src/data/dataloader.py` | `DataLoader` | `dataset`, `batch_size: int`, `shuffle: bool` | dataloader instance |
| `src/nn/activations.py` | `sigmoid`, `softmax`, `identity`, `relu` | `np.ndarray` | `np.ndarray` |
| `src/nn/layers.py` | `Linear`, `Sigmoid`, `ReLU`, `Sequential` | 차원 또는 없음 | layer instance |
| `src/nn/losses.py` | `cross_entropy`, `binary_cross_entropy`, `mse` | `logits, targets: np.ndarray` | scalar |
| `src/nn/losses.py` | `cross_entropy_grad`, `binary_cross_entropy_grad`, `mse_grad` | `logits, targets: np.ndarray` | `np.ndarray` |
| `src/nn/metrics.py` | `accuracy`, `binary_accuracy`, `r2_score` | `logits, targets: np.ndarray` | scalar |
| `src/models/mlp.py` | `MLP` | `task: str`, `seed: int` | model instance |
| `src/core/optimizers.py` | `SGD`, `Adam` | model instance, `lr: float` | optimizer instance |
| `src/core/trainer.py` | `Trainer` | model, optimizer, task spec | trainer instance |
| `src/core/evaluator.py` | `Evaluator` | model, task spec | evaluator instance |
| `src/core/predictor.py` | `Predictor` | model, task spec | predictor instance |
| `src/utils/checkpoints.py` | `save_checkpoint`, `load_checkpoint` | model, path | 없음 / model params |

### 3.2. 입력·출력 세부 규약

각 진입점의 입력·출력 타입과 shape 규약은 아래와 같다.

`split` 값은 `"train"` 또는 `"test"`만 허용한다. `task` 값은 `"multiclass"`, `"binary"`, `"regression"`만 허용한다.

`load_mnist(split)` 반환값은 다음과 같다.

- `images`: `(N, 28, 28)` uint8 원본 배열
- `labels`: `(N,)` uint8 원본 배열

`MnistDataset(split, task)` 내부 배열 규약은 다음과 같다.

- `images`: `(N, 784)` float32 (`reshape(-1, 784)` + `/255` 정규화 완료)
- `targets`: task별 float32 배열 (아래 표 참조)
- `__getitem__(idx)`: `(image, target)` 단일 샘플 tuple 반환

`get_task_spec(task)` 반환 dict는 최소한 아래 키를 포함한다.

| 키 | 의미 |
|---|---|
| `task` | task 이름 문자열 |
| `output_dim` | 모델 출력 차원 수 |
| `target_dtype` | target 배열의 dtype |
| `prediction_mode` | 예측 후처리 방식 |

### 3.3. task별 규약

task별로 다른 항목의 전체 규약은 아래와 같다.

| 구분 | multiclass | binary | regression |
|---|---|---|---|
| target 변환 | `one_hot(labels, 10)` | `(labels % 2).reshape(-1, 1)` | `labels / 9.0` |
| target shape | `(N, 10)` float32 | `(N, 1)` float32 | `(N, 1)` float32 |
| output_dim | 10 | 1 | 1 |
| MLP 출력 활성화 | softmax (losses.py 내부 처리) | sigmoid (losses.py 내부 처리) | identity |
| 손실 함수 | `cross_entropy` | `binary_cross_entropy` | `mse` |
| 평가 지표 | `accuracy` | `binary_accuracy` | `r2_score` |
| 예측 후처리 | argmax | `prob >= 0.5` -> 0/1 | `round(clip(raw * 9.0, 0, 9))` |

### 3.4. 메서드 반환값 규약

실행 객체의 주요 메서드 반환값 규약은 아래와 같다.

`MLP.forward(x)` 반환값과 `params`/`grads` 구조는 다음과 같다.

- `forward(x)`: `(N, output_dim)` raw logit 배열 반환 (activation은 losses.py에서 처리)
- `params`: list - `params[0]`이 첫 번째 `Linear`의 `w`
- `grads`: list - `params`와 동일 순서

`SGD.step()`, `Adam.step()` 반환값은 없다. `model.params`를 in-place 업데이트한다.

`Trainer.fit(train_loader)` 반환값은 epoch별 로그 dict 목록이다.

`Evaluator.evaluate(test_loader)` 반환 dict는 `loss`, `metric`, `num_samples` 키를 포함한다.

`Predictor.predict(images)` 반환 dict는 raw prediction과 decoded prediction을 함께 포함한다.

## 4. 사용법

`MnistDataset`과 `DataLoader`를 조합하는 최소 예제는 다음과 같다.

```python
from src.data.mnist import MnistDataset
from src.data.dataloader import DataLoader

dataset = MnistDataset(split="train", task="multiclass")
loader = DataLoader(dataset, batch_size=64, shuffle=True)

for images, targets in loader:
    print(images.shape)   # (64, 784) float32
    print(targets.shape)  # (64, 10)  float32
    break
```

`get_task_spec()`을 사용하는 예제는 다음과 같다.

```python
from src.data.mnist import get_task_spec

spec = get_task_spec("multiclass")
print(spec["output_dim"])       # 10
print(spec["prediction_mode"])  # "argmax"
```

## 5. 테스트

Phase 0.2는 계획·규약 수립 단계이므로 대응하는 테스트 파일이 없다. 각 규약의 동작 검증은 Stage 2부터 시작하는 구현 단계에서 테스트 코드로 확인한다.

## 6. 요약

공개 함수명·클래스명·입력·출력 규약을 이 문서에서 확정하면 Stage 2부터 구현할 때 인터페이스 변경 없이 파일 단위로 작업할 수 있다. task 규약은 `get_task_spec(task)`로 중앙 관리하여 `Trainer`, `Evaluator`, `Predictor`가 task 분기를 통일된 방식으로 처리한다. 이 규약은 후속 PyTorch, TensorFlow, JAX 프로젝트에서도 동일하게 유지되어 프레임워크 간 비교 기준이 된다.

다음 Phase에서는 [[phase0.3_test-plan]]을 다룬다.
