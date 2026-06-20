---
tags: [project, stage0]
created: 2026-06-17
updated: 2026-06-17
---

# Phase 0.3 테스트 계획 수립

이 문서는 tests 폴더 구조, 파일별 공개 인터페이스 규약, TDD 원칙, pytest 실행 명령을 확정한다.

## 1. tests 폴더 구조

테스트 코드는 Stage 단위로 폴더를 분리하고, `__init__.py`를 두지 않는다.

```text
tests/
├── conftest.py
├── stage1/
│   ├── test_config.py
│   ├── test_task.py
│   ├── test_batching.py
│   ├── test_random.py
│   └── test_io.py
├── stage2/
│   ├── test_mnist.py
│   ├── test_dataset.py
│   └── test_dataloader.py
├── stage3/
│   ├── test_mlp.py
│   ├── test_layers.py
│   ├── test_activations.py
│   └── test_losses.py
├── stage4/
│   ├── test_optimizers.py
│   ├── test_checkpoints.py
│   ├── test_trainer.py
│   ├── test_evaluator.py
│   ├── test_predictor.py
│   └── test_experiment.py
└── stage5/
    ├── test_train.py
    ├── test_evaluate.py
    ├── test_predict.py
    └── test_visualize.py
```

`__init__.py`를 두지 않는 이유는 `pyproject.toml`의 `pythonpath` 설정으로 import를 해결하기 때문이다. `conftest.py`는 tests 루트에 하나만 두고 재사용 fixture만 관리한다.

## 2. 파일별 공개 인터페이스 규약

각 구현 파일의 공개 진입점과 입력·출력 규약을 확정한다.

### 2.1. Stage 1 - 기본 설정 · 과제 규약

Stage 1 파일의 공개 인터페이스는 아래와 같다.

| 파일 | 공개 진입점 | 입력 | 출력 |
|---|---|---|---|
| `src/config.py` | `get_default_config()` | 없음 | `dict` |
| `src/task.py` | `get_task_spec(task)` | `task: str` | `dict` |
| `src/task.py` | `transform_targets(labels, task)` | `labels: np.ndarray`, `task: str` | `np.ndarray` |
| `src/utils/batching.py` | `get_batches(n, batch_size, shuffle)` | `int`, `int`, `bool` | list of index arrays |
| `src/utils/random.py` | `set_seed(seed)` | `seed: int` | 없음 |
| `src/utils/io.py` | `save_array`, `load_array` | 경로, 배열 | 배열 또는 없음 |

`get_default_config()`가 반환하는 dict는 최소한 `dataset_dir`, `seed`, `batch_size`, `num_epochs`, `task`, `split` 키를 포함한다.

`get_task_spec(task)`가 반환하는 dict는 최소한 `task`, `output_dim`, `target_dtype`, `prediction_mode` 키를 포함한다.

`task` 값은 `"multiclass"`, `"binary"`, `"regression"` 세 가지만 허용한다.

### 2.2. Stage 2 - MNIST 데이터 로더

Stage 2 파일의 공개 인터페이스는 아래와 같다.

| 파일 | 공개 진입점 | 입력 | 출력 |
|---|---|---|---|
| `src/data/mnist.py` | `load_mnist(split)` | `split: str` | `(images, labels)` tuple |
| `src/data/mnist.py` | `MnistDataset` | `split: str`, `task: str` | dataset instance |
| `src/data/dataloader.py` | `DataLoader` | `dataset`, `batch_size: int`, `shuffle: bool` | dataloader instance |

`load_mnist(split)`의 `images`는 `(N, 28, 28)` uint8, `labels`는 `(N,)` uint8 원본 배열을 반환한다.

`split` 값은 `"train"` 또는 `"test"`만 허용한다.

`MnistDataset(split, task)`의 `images`는 `(N, 784)` float32 (reshape + `/255` 정규화 완료), `targets`는 task별 float32 배열이다.

`MnistDataset.__getitem__(idx)`는 `(image, target)` 단일 샘플 tuple을 반환한다.

task별 target 변환 규약은 아래와 같다.

| task | 변환 | target shape |
|---|---|---|
| `multiclass` | `one_hot(labels, num_classes=10)` | `(N, 10)` float32 |
| `binary` | `(labels % 2).reshape(-1, 1)` (홀수=1, 짝수=0) | `(N, 1)` float32 |
| `regression` | `labels / 9.0` | `(N, 1)` float32 |

`DataLoader(dataset, batch_size, shuffle)`의 `__iter__`는 `(images_batch, targets_batch)` tuple을 yield한다. `__len__`·`__getitem__`을 구현한 Dataset이면 종류에 관계없이 수용한다.

### 2.3. Stage 3 - NumPy MLP 및 구성요소

Stage 3 파일의 공개 인터페이스는 아래와 같다.

| 파일 | 공개 진입점 | 입력 | 출력 |
|---|---|---|---|
| `src/models/mlp.py` | `MLP` | config 또는 명시적 차원 인자 | model instance |
| `src/models/layers.py` | `Linear`, `Sigmoid`, `ReLU`, `Sequential` | 차원 또는 없음 | layer instance |
| `src/models/activations.py` | `sigmoid`, `softmax`, `identity`, `relu` | `np.ndarray` | `np.ndarray` |
| `src/models/losses.py` | `cross_entropy`, `binary_cross_entropy`, `mse` | `preds, targets: np.ndarray` | scalar |
| `src/models/losses.py` | `accuracy`, `binary_accuracy`, `r2_score` | `preds, targets: np.ndarray` | scalar |

`MLP.forward(x)`는 `(N, output_dim)` prediction 배열을 반환한다.

`Linear.backward(dout)`는 상위 레이어로 전달할 gradient를 반환하며, `grad_w`와 `grad_b`를 인스턴스에 in-place 저장한다.

### 2.4. Stage 4 - 실행 객체

Stage 4 파일의 공개 인터페이스는 아래와 같다.

| 파일 | 공개 진입점 | 입력 | 출력 |
|---|---|---|---|
| `src/core/optimizers.py` | `SGD`, `Adam` | model instance, `lr: float` | optimizer instance |
| `src/core/checkpoints.py` | `save`, `load` | 경로, model | 없음 또는 model |
| `src/core/trainer.py` | `Trainer` | model, optimizer, task spec | trainer instance |
| `src/core/evaluator.py` | `Evaluator` | model, task spec | evaluator instance |
| `src/core/predictor.py` | `Predictor` | model, task spec | predictor instance |
| `src/core/experiment.py` | `Experiment` | config | experiment instance |

`SGD.step()`, `Adam.step()`은 `model.params`를 in-place 업데이트하며 반환값이 없다.

`Trainer.fit(train_loader)`는 `DataLoader`를 수신하며 epoch별 로그 dict 목록 또는 요약 dict를 반환한다.

`Evaluator.evaluate(test_loader)`는 `DataLoader`를 수신하며 `loss`, `metric`, `num_samples`를 포함한 dict를 반환한다.

`Predictor.predict(images)`는 raw prediction과 decoded prediction을 함께 담은 dict를 반환한다.

`Experiment`는 config를 기준으로 dataset, dataloader, task spec, model, optimizer, trainer, evaluator, predictor를 조립하는 최상위 진입점 역할을 한다.

## 3. TDD 원칙

Stage 1부터는 구현보다 테스트를 먼저 작성한다. 아래 원칙을 기준으로 유지한다.

테스트 작성 순서는 아래와 같다.

- 구현 파일 작성 전에 대응 테스트 파일을 먼저 작성한다.
- 테스트가 실패하는 것을 확인한 뒤 구현 파일을 작성한다.
- 구현 후 테스트가 통과하면 다음 파일로 이동한다.

테스트 범위와 방식은 아래 기준을 따른다.

- public interface 기준으로 테스트를 작성하고, 내부 helper는 필요한 경우에만 간접 검증한다.
- unit test를 먼저 작성하고, data-task 또는 model-core 조합이 필요한 경우에만 통합 테스트를 추가한다.
- MNIST 원본 전체 의존 테스트는 최소화하고, 작은 synthetic array 기반 테스트를 우선 작성한다.
- 숫자 비교는 부동소수점 오차를 고려하여 exact equality 대신 tolerance 비교(`np.allclose`, `pytest.approx`)를 사용한다.
- fixture는 재사용 가치가 있는 경우에만 `tests/conftest.py`로 올리고, 파일 전용 데이터는 각 테스트 파일 내부에 둔다.

테스트 우선순위는 아래와 같다.

| 우선순위 | 테스트 파일 | 대상 |
|---|---|---|
| 1 | `tests/stage1/test_task.py` | task별 target 변환과 task spec |
| 2 | `tests/stage2/test_mnist.py` | 로컬 MNIST 로딩, split 처리, shape 검증 |
| 3 | `tests/stage3/test_mlp.py` | MLP 생성, forward shape, parameter update 흐름 |
| 4 | `tests/stage4/test_optimizers.py` | SGD, Adam 파라미터 업데이트 |
| 5 | `tests/stage4/test_trainer.py` | 학습 루프와 batch 처리 |
| 6 | `tests/stage4/test_evaluator.py` | 평가 루프와 metric 집계 |
| 7 | `tests/stage4/test_experiment.py` | data, task, model, core 실행 객체 조립 |

## 4. pytest 실행 명령

pytest는 파일 단위 구현 흐름에 맞춰 아래 순서로 사용한다.

| 목적 | 명령 |
|---|---|
| Stage 1 전체 | `pytest tests/stage1/ -q` |
| Stage 2 전체 | `pytest tests/stage2/ -q` |
| Stage 3 전체 | `pytest tests/stage3/ -q` |
| Stage 4 전체 | `pytest tests/stage4/ -q` |
| Stage 5 전체 | `pytest tests/stage5/ -q` |
| 단일 파일 | `pytest tests/stage3/test_mlp.py -q` |
| 전체 스모크 확인 | `pytest tests/ -q` |

`pyproject.toml`에 `pythonpath = ["."]` 또는 `pythonpath = ["src"]` 설정을 두어 테스트 파일이 `src` 모듈을 직접 import할 수 있도록 한다. `__init__.py` 없이도 import가 해결되는 이유이다.
