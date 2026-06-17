---
tags: [project, docs]
created: 2026-06-08
updated: 2026-06-17
---

# project-spec.md

이 프로젝트의 목적, 배경, 범위, 제약 사항, 진행 단계를 정의한다.
사용자가 작성하며, 에이전트는 요청 시에만 갱신한다.

## 1. 목적

이 프로젝트의 목적은 NumPy와 CuPy만으로 MNIST 기반 딥러닝 모델 학습 과정을 구현하고, 이후 PyTorch, TensorFlow, JAX 프로젝트와 동일한 모듈 및 함수 인터페이스를 유지할 수 있는 기준 구현을 작성하는 것이다.

## 2. 배경

이 프로젝트는 `numpy -> pytorch -> tensorflow -> jax` 순서로 진행되는 딥러닝 프레임워크 학습 시리즈의 첫 번째 프로젝트이다. 프레임워크별 구현을 비교 학습하려면 동일한 문제, 동일한 구조, 동일한 사용법을 갖는 프로젝트 구성이 필요하다.

## 3. 범위

이 프로젝트에서 수행할 작업 범위는 MNIST 데이터셋 기반 태스크, 모델 구현, 문서화, 테스트 구조를 모두 포함한다.

- MNIST 데이터셋을 대상으로 실험 환경을 구성한다.
- Multiclass Classification, Binary Classification, Regression 과제를 수행한다.
- MLP 모델은 CPU 기반 NumPy 구현으로 학습한다.
- CNN 모델은 GPU 기반 CuPy 구현으로 학습한다.
- 후속 PyTorch, TensorFlow, JAX 프로젝트와 호환되도록 모듈명, 함수명, 사용법을 통일한다.
- `_core/legacy/src/` 폴더에 사용자가 제공하는 기존 3가지 작업의 레거시 코드를 보관하고 분석한다.
- `src/` 폴더에 실제 소스 코드를 구현한다.
- `scripts/` 폴더에 클라이언트 코드 `train.py`, `evaluate.py`, `predict.py`, `visualize.py`를 작성한다.
- `tests/` 폴더 기반으로 `pytest`를 이용한 TDD 구조를 구현한다.
- 튜토리얼 문서와 실행 가능한 예제 코드를 함께 작성한다.

## 4. 제약 사항

이 프로젝트의 제약 사항은 시리즈 기준 구현으로서의 일관성과 실행 환경의 역할 분리를 중심으로 정의한다.

- 첫 번째 프로젝트이므로 후속 PyTorch, TensorFlow, JAX 프로젝트의 기준 구조 역할을 해야 한다.
- 프레임워크별 프로젝트에서 작성되는 모듈과 함수는 같은 이름과 사용법을 가져야 한다.
- MLP는 CPU 학습, CNN은 GPU 학습을 기준으로 한다.
- 프로젝트 문서 초안을 먼저 작성한 뒤 `src/` 폴더 구조를 설계하고 검토해야 한다.
- 테스트는 `pytest` 기반 TDD 흐름으로 구현해야 한다.

## 5. 진행 단계

프로젝트는 Stage 0에서 기존 설계를 재검토한 뒤, Stage 1부터 각 파일마다 코드 작성 Task와 테스트 작성 Task를 분리하여 진행한다.

### 5.1. Stage 0 프로젝트 설계 재검토

- Phase 0.1 기존 계획 및 레거시 입력 재검토
- Phase 0.2 공통 `src` 및 `tests` 구조 확정
- Phase 0.3 파일 단위 코드·테스트 구현 순서 확정

### 5.2. Stage 1 기본 설정 및 과제 규약 구현

- Phase 1.1 `config.py` 코드 작성 및 `test_config.py` 테스트 작성
- Phase 1.2 `task.py` 코드 작성 및 `test_task.py` 테스트 작성
- Phase 1.3 `utils` 코드 작성 및 대응 테스트 작성

### 5.3. Stage 2 MNIST 데이터 로더 구현

- Phase 2.1 `data/mnist.py` 코드 작성 및 테스트 작성 — `load_mnist()` raw 로딩
- Phase 2.2 `data/mnist.py` 확장 및 테스트 작성 — `MnistDataset` 클래스 추가
- Phase 2.3 `data/dataloader.py` 코드 작성 및 테스트 작성 — 범용 `DataLoader` 클래스

### 5.4. Stage 3 NumPy 기반 MLP 구현

- Phase 3.1 `models/mlp.py` 코드 작성 및 `models/test_mlp.py` 테스트 작성
- Phase 3.2 MLP 학습에 필요한 모델 하위 구현과 대응 테스트 작성

### 5.5. Stage 4 실행 객체 구현

- Phase 4.1 `core/checkpoints.py` 코드 작성 및 `core/test_checkpoints.py` 테스트 작성
- Phase 4.2 `core/trainer.py` 코드 작성 및 `core/test_trainer.py` 테스트 작성
- Phase 4.3 `core/evaluator.py` 코드 작성 및 `core/test_evaluator.py` 테스트 작성
- Phase 4.4 `core/predictor.py` 코드 작성 및 `core/test_predictor.py` 테스트 작성
- Phase 4.5 `core/experiment.py` 코드 작성 및 `core/test_experiment.py` 테스트 작성
- Phase 4.6 `core/visualizer.py` 코드 작성 및 `core/test_visualizer.py` 테스트 작성

### 5.6. Stage 5 클라이언트 코드 구현

- Phase 5.1 `scripts/train.py` 코드 작성 및 `tests/scripts/test_train.py` 테스트 작성
- Phase 5.2 `scripts/evaluate.py` 코드 작성 및 `tests/scripts/test_evaluate.py` 테스트 작성
- Phase 5.3 `scripts/predict.py` 코드 작성 및 `tests/scripts/test_predict.py` 테스트 작성
- Phase 5.4 `scripts/visualize.py` 코드 작성 및 `tests/scripts/test_visualize.py` 테스트 작성

### 5.7. Stage 6 CuPy 기반 CNN 구현

- Phase 6.1 `models/cnn.py` 코드 작성 및 `models/test_cnn.py` 테스트 작성
- Phase 6.2 CNN 실행 객체 연동 테스트

### 5.8. Stage 7 튜토리얼 문서화 및 전체 검증

- Phase 7.1 Jupyter Book 챕터 작성
- Phase 7.2 실행 예제와 결과 정리
- Phase 7.3 후속 프레임워크 프로젝트 연계 기준 정리

## 6. 확정 구조

프로젝트의 소스 코드와 테스트 코드는 후속 PyTorch, TensorFlow, JAX 프로젝트와 같은 최상위 구조를 유지하도록 설계한다.

### 6.1. 데이터셋 기준

MNIST 데이터셋은 로컬 저장소에 보관된 4개 압축 파일을 사용하며, 다운로드 기능은 기본 범위에 포함하지 않는다.

- 기본 데이터셋 경로는 `DATASET_DIR = "/mnt/d/datasets/mnist"`로 사용한다.
- 데이터 split은 `split = "train"` 또는 `split = "test"` 값으로 선택한다.
- 과제 유형은 `task = "multiclass"`, `task = "binary"`, `task = "regression"` 값으로 선택한다.
- `task`별 target 변환, output dimension, loss, metric 규약은 단일 파일 `src/task.py`에서 관리한다.

MNIST 로컬 저장소는 다음 4개 파일을 포함해야 한다.

```text
/mnt/d/datasets/mnist/
├── train-images-idx3-ubyte.gz
├── train-labels-idx1-ubyte.gz
├── t10k-images-idx3-ubyte.gz
└── t10k-labels-idx1-ubyte.gz
```

### 6.2. `src` 패키지 구조

`src` 패키지는 프레임워크별 구현 차이를 하위 구현으로 흡수하고, 최상위 구조와 클라이언트 사용법은 동일하게 유지한다.

```text
src/
├── __init__.py
├── config.py
├── data/
│   ├── __init__.py
│   ├── mnist.py
│   └── dataloader.py
├── task.py
├── models/
│   ├── __init__.py
│   ├── mlp.py
│   └── cnn.py
├── core/
│   ├── __init__.py
│   ├── experiment.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── predictor.py
│   ├── visualizer.py
│   └── checkpoints.py
└── utils/
    ├── __init__.py
    ├── batching.py
    ├── random.py
    └── io.py
```

각 위치의 책임은 다음 기준으로 분리한다.

| 위치 | 책임 |
| --- | --- |
| `src/config.py` | 기본 경로, 기본 split, 기본 task, 기본 실행 설정을 정의한다. |
| `src/data/mnist.py` | 로컬 MNIST `*.gz` 파일 로딩(`load_mnist`)과 `MnistDataset` 클래스를 제공한다. task별 target 변환은 `MnistDataset` 내부에서 처리한다. |
| `src/data/dataloader.py` | 범용 `DataLoader` 클래스를 제공한다. `__len__`과 `__getitem__`을 구현한 Dataset이면 모두 수용한다. |
| `src/task.py` | task별 output_dim, loss, metric, prediction_mode 규약을 단일 진입점으로 관리한다. `transform_targets`는 각 Dataset 클래스가 내부에서 호출하는 헬퍼로 사용한다. |
| `src/models/` | MLP, CNN 모델 구조를 배치하며 프레임워크별 하위 구현 차이를 흡수한다. |
| `src/core/` | `scripts/`에서 참조하는 실행 객체를 배치한다. |
| `src/utils/` | batching, random seed, file I/O 등 공통 보조 기능을 배치한다. |

### 6.3. `scripts`와 `core` 관계

클라이언트 코드는 내부 구현 모듈을 직접 조립하지 않고 `core`의 실행 객체를 참조한다.

| 클라이언트 파일 | 참조 대상 |
| --- | --- |
| `scripts/train.py` | `src/core/experiment.py`, `src/core/trainer.py` |
| `scripts/evaluate.py` | `src/core/experiment.py`, `src/core/evaluator.py` |
| `scripts/predict.py` | `src/core/experiment.py`, `src/core/predictor.py` |
| `scripts/visualize.py` | `src/core/experiment.py`, `src/core/visualizer.py` |

### 6.4. `tests` 폴더 구조

테스트 코드는 `src`와 `scripts`의 테스트 대상별로 파일을 분리하여 작성한다.

```text
tests/
├── __init__.py
├── conftest.py
├── test_config.py
├── test_task.py
├── data/
│   ├── __init__.py
│   └── test_mnist.py
├── models/
│   ├── __init__.py
│   ├── test_mlp.py
│   └── test_cnn.py
├── core/
│   ├── __init__.py
│   ├── test_experiment.py
│   ├── test_trainer.py
│   ├── test_evaluator.py
│   ├── test_predictor.py
│   ├── test_visualizer.py
│   └── test_checkpoints.py
├── scripts/
│   ├── __init__.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   ├── test_predict.py
│   └── test_visualize.py
└── utils/
    ├── __init__.py
    ├── test_batching.py
    ├── test_random.py
    └── test_io.py
```

초기 테스트 작성 우선순위는 데이터와 task 규약에서 시작하여 모델, 실행 객체, 클라이언트 코드로 확장한다.

| 우선순위 | 테스트 파일 | 대상 |
| --- | --- | --- |
| 1 | `tests/test_task.py` | task별 target 변환과 task spec |
| 2 | `tests/data/test_mnist.py` | 로컬 MNIST 로딩, split 처리, shape 검증 |
| 3 | `tests/models/test_mlp.py` | MLP 생성, forward shape, parameter update 흐름 |
| 4 | `tests/core/test_trainer.py` | 학습 루프와 batch 처리 |
| 5 | `tests/core/test_evaluator.py` | 평가 루프와 metric 집계 |
| 6 | `tests/core/test_experiment.py` | data, task, model, core 실행 객체 조립 |
| 7 | `tests/scripts/test_train.py` | train 클라이언트 인자와 실행 객체 호출 |

### 6.5. 레거시 코드와 구현 파일 매핑

레거시 코드 3개 파일은 공통 파이프라인과 task별 차이로 분리하여 Stage 1 이후 구현 파일에 매핑한다.

공통 파이프라인은 모든 task에서 동일하게 유지한다.

- 데이터 로딩: `train`/`test` split 기준으로 이미지와 레이블을 로드한 뒤 이미지는 `float32`와 `[0, 1]` 범위로 정규화한다.
- 모델 구조: `784 -> 256 -> 128 -> output_dim` MLP 구조와 hidden activation `sigmoid`를 공통으로 사용한다.
- 학습 루프: epoch 단위 반복, mini-batch shuffle, forward, loss/metric 계산, manual backward, SGD 업데이트 흐름을 공통으로 유지한다.
- 평가 루프: 학습 파라미터를 재사용하여 test split 전체를 batch 단위로 순회하고 평균 loss/metric을 집계한다.
- 예측 루프: test 샘플 일부를 대상으로 task별 후처리를 적용한 예측 결과를 출력한다.

task별 차이와 구현 대상 파일의 매핑 기준은 아래와 같다.

| 구분 | multiclass 레거시 | binary 레거시 | regression 레거시 | 구현 대상 파일 |
| --- | --- | --- | --- | --- |
| target 변환 | one-hot, shape `(N, 10)` | 홀수/짝수 이진화, shape `(N, 1)` | `label / 9.0`, shape `(N, 1)` | `src/task.py` |
| output dimension | `10` | `1` | `1` | `src/task.py`, `src/models/mlp.py` |
| output activation | `softmax` | `sigmoid` | `identity` | `src/task.py`, `src/models/mlp.py` |
| loss | `cross_entropy` | `binary_cross_entropy` | `mse` | `src/task.py`, `src/core/trainer.py`, `src/core/evaluator.py` |
| metric | `accuracy` | `binary_accuracy` | `r2_score` | `src/task.py`, `src/core/trainer.py`, `src/core/evaluator.py` |
| output gradient | `(preds - y) / batch_size` | `(preds - y) / batch_size` | `2 * (preds - y) / batch_size` | `src/models/` 하위 구현 |
| prediction 후처리 | `argmax` | `prob >= 0.5` | `round(clip(raw * 9.0, 0, 9))` | `src/core/predictor.py` |

파일 단위 구현 계획은 공통 책임과 task 규약을 분리하는 방향으로 유지한다.

- `src/data/mnist.py`: gzip 파일 읽기, split 선택, 이미지/레이블 원본 로딩 담당
- `src/task.py`: target 변환, output dimension, loss, metric, prediction 후처리 규약 담당
- `src/models/mlp.py`: forward, backward, parameter update에 필요한 NumPy MLP 구현 담당
- `src/core/trainer.py`: batch 반복, 학습 loss/metric 집계 담당
- `src/core/evaluator.py`: 평가 loss/metric 집계 담당
- `src/core/predictor.py`: task별 prediction 후처리와 출력 형식 담당

### 6.6. 공통 함수명과 입력·출력 규약

Stage 1 초기 구현에서 사용할 공통 진입점은 후속 프레임워크 프로젝트와 같은 이름을 유지해야 한다.

우선 확정한 파일별 공개 함수와 클래스는 아래와 같다.

| 파일 | 공개 진입점 | 입력 | 출력 | 책임 |
| --- | --- | --- | --- | --- |
| `src/config.py` | `get_default_config()` | 없음 | `dict` | 기본 경로, seed, batch size, epoch, task, split 반환 |
| `src/data/mnist.py` | `load_mnist(split)` | `split: str` | `(images, labels)` tuple | 로컬 MNIST 원본 배열 로딩 |
| `src/data/mnist.py` | `MnistDataset` | `split: str`, `task: str` | dataset instance | MNIST 로딩·정규화·task별 target 변환 담당 |
| `src/data/dataloader.py` | `DataLoader` | `dataset`, `batch_size: int`, `shuffle: bool` | dataloader instance | 범용 배치·셔플 이터레이터 (`__len__`+`__getitem__` 프로토콜 요구) |
| `src/task.py` | `get_task_spec(task)` | `task: str` | `dict` | `output_dim`, activation, loss, metric, prediction_mode 반환 |
| `src/task.py` | `transform_targets(labels, task)` | `labels: np.ndarray`, `task: str` | `np.ndarray` | task별 target 변환 — 각 Dataset 클래스 내부에서 호출 |
| `src/models/mlp.py` | `MLP` | config 또는 명시적 차원 인자 | model instance | NumPy 기반 MLP 생성 |
| `src/core/trainer.py` | `Trainer` | model, task spec, config | trainer instance | 학습 루프 실행 |
| `src/core/evaluator.py` | `Evaluator` | model, task spec, config | evaluator instance | 평가 루프 실행 |
| `src/core/predictor.py` | `Predictor` | model, task spec, config | predictor instance | 예측 및 후처리 실행 |
| `src/core/experiment.py` | `Experiment` | config | experiment instance | data, task, model, 실행 객체 조립 |

초기 구현에서 통일할 입력·출력 규약은 아래 기준을 따른다.

- `split` 값은 `"train"` 또는 `"test"`만 허용한다.
- `task` 값은 `"multiclass"`, `"binary"`, `"regression"`만 허용한다.
- `load_mnist(split)`의 `images`는 `(N, 28, 28)` `uint8`, `labels`는 `(N,)` `uint8` 원본 배열을 반환한다.
- `MnistDataset(split, task)`의 `images`는 `(N, 784)` `float32` (reshape + /255 정규화 완료), `targets`는 task별 `float32` 배열이다.
- `MnistDataset.__getitem__(idx)`는 `(image, target)` 단일 샘플 tuple을 반환한다.
- `MnistDataset`의 task별 target 변환 규약은 다음과 같다.
  - `multiclass`: `one_hot(labels, num_classes=10)` → shape `(N, 10)`
  - `binary`: `(labels % 2)` (홀수=1, 짝수=0) → shape `(N, 1)`
  - `regression`: `labels / 9.0` → shape `(N, 1)`
- `DataLoader(dataset, batch_size, shuffle)`의 `__iter__`는 `(images_batch, targets_batch)` tuple을 yield한다.
- `DataLoader`는 `__len__`과 `__getitem__`을 구현한 Dataset이면 종류에 관계없이 수용한다.
- `get_task_spec(task)`는 최소한 `task`, `output_dim`, `target_dtype`, `prediction_mode` 키를 포함한다.
- `transform_targets(labels, task)`는 `task.py`에 유지하며 각 Dataset 클래스 내부에서 호출한다.
- `MLP.forward(x)`는 `(N, output_dim)` prediction 배열을 반환한다.
- `Trainer.fit(train_loader)`는 `DataLoader`를 수신하며 epoch별 로그 dict 목록 또는 요약 dict를 반환한다.
- `Evaluator.evaluate(test_loader)`는 `DataLoader`를 수신하며 `loss`, `metric`, `num_samples`를 포함한 dict를 반환한다.
- `Predictor.predict(images)`는 raw prediction과 decoded prediction을 함께 담은 dict를 반환한다.
- `Experiment`는 config를 기준으로 dataset, dataloader, task spec, model, trainer, evaluator, predictor를 조립하는 최상위 진입점 역할을 한다.

### 6.7. 실패 테스트 작성 원칙과 `pytest` 실행 기준

Stage 1부터는 구현보다 테스트가 먼저 작성되도록 TDD 순서를 고정한다.

실패 테스트 작성 원칙은 아래 기준을 따른다.

- public interface 기준으로 테스트를 먼저 작성하고, 내부 helper는 필요한 경우에만 간접 검증한다.
- 한 번에 하나의 파일 단위를 구현하며 해당 파일의 대응 테스트가 먼저 실패하는지 확인한다.
- unit test를 먼저 작성하고, data-task 또는 model-core 조합이 필요한 경우에만 통합 테스트를 추가한다.
- fixture는 재사용 가치가 있는 경우에만 `tests/conftest.py`로 올리고, 파일 전용 데이터는 각 테스트 파일 내부에 둔다.
- 숫자 비교는 부동소수점 오차를 고려하여 exact equality 대신 tolerance 비교를 사용한다.
- MNIST 원본 전체 의존 테스트는 최소화하고, 작은 synthetic array 기반 테스트를 우선 작성한다.

`pytest` 실행은 파일 단위 구현 흐름에 맞춰 아래 순서로 사용한다.

| 목적 | 명령 |
| --- | --- |
| config 단일 파일 테스트 | `pytest tests/test_config.py -q` |
| task 단일 파일 테스트 | `pytest tests/test_task.py -q` |
| utils 단일 파일 테스트 | `pytest tests/utils/test_batching.py -q` |
| data 단일 파일 테스트 | `pytest tests/data/test_mnist.py -q` |
| core 단일 파일 테스트 | `pytest tests/core/test_trainer.py -q` |
| 전체 테스트 스모크 확인 | `pytest tests -q` |
