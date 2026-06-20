---
tags: [docs, stage0, implementation-plan]
created: "2026-06-20"
updated: "2026-06-20"
---

# 구현 계획 수립

## 1. 개요

레거시 common 모듈 6개를 분석한 결과를 바탕으로 새로운 `src/` 패키지의 파일 구조, 각 파일의 책임 범위, Stage 1부터 Stage 6까지의 구현 순서를 확정한다. 이 계획이 확정되면 Stage 1부터 파일 단위로 TDD 순서에 따라 구현을 시작할 수 있다. 후속 PyTorch, TensorFlow, JAX 프로젝트와 동일한 최상위 구조를 유지하도록 설계한다.

**목표**
- 레거시 common 모듈 6개와 새로운 `src/` 파일의 1:1 매핑을 확정한다.
- `src/` 패키지의 하위 폴더 구조와 각 파일의 단일 책임 범위를 확정한다.
- Stage 1부터 Stage 6까지의 구현 순서와 Phase 단위 분할을 확정한다.

## 2. 개념

### 2.1. 레거시와 신규 구현의 차이

레거시 코드는 task별로 6개의 스크립트를 각각 완성된 형태로 작성한다. 공통 로직이 common 모듈로 분리되어 있지만, task 규약(target 변환, output dim, loss, metric)은 스크립트 내부에서 처리하거나 `trainer.py`의 Classifier 클래스에 분산되어 있다.

새로운 `src/` 구조는 세 가지 방향으로 개선한다.

- `common/functions.py`처럼 활성화, 손실, 지표가 혼재된 파일을 역할별로 분리한다.
- task 규약은 데이터와 가장 밀접한 `src/data/mnist.py`의 `get_task_spec()`으로 통합 관리한다.
- `trainer.py`의 학습·평가·예측 역할을 `Trainer`, `Evaluator`, `Predictor`로 분리한다.

### 2.2. 구현 순서 원칙

Stage는 의존성 순서를 따른다. 하위 레이어가 완성된 뒤 상위 레이어를 구현한다.

의존성 방향은 다음과 같다.

```text
utils -> data -> nn -> models -> core -> scripts/experiments
```

각 Stage 내에서도 의존성이 없는 파일을 먼저 구현하고, 의존하는 파일을 나중에 구현한다. 각 파일 구현 전에 대응 테스트 파일을 먼저 작성한다.

## 3. 구현

### 3.1. 레거시 common 모듈 → src 파일 매핑

레거시 6개 common 모듈과 새로운 `src/` 파일의 매핑은 아래와 같다.

`common/functions.py`는 활성화, 손실, 지표, gradient 함수가 혼합되어 있으므로 역할별로 3개 파일로 분리한다. `common/trainer.py`는 학습·평가·예측 역할이 혼합되어 있으므로 3개 파일로 분리한다.

| 레거시 | src 대응 | 변경 사항 |
|---|---|---|
| `common/mnist.py` | `src/data/mnist.py` | `load_images`/`load_labels` -> `load_mnist`로 통합, `MnistDataset` 추가 |
| `common/dataloader.py` | `src/data/dataloader.py` | 배열 직접 수신 -> Dataset 프로토콜 기반으로 전환 |
| `common/functions.py` (활성화) | `src/nn/activations.py` | `sigmoid`, `softmax`, `identity`, `relu` |
| `common/functions.py` (손실·gradient) | `src/nn/losses.py` | 손실 3종 + gradient 3종 |
| `common/functions.py` (지표) | `src/nn/metrics.py` | `accuracy`, `binary_accuracy`, `r2_score` |
| `common/modules.py` | `src/nn/layers.py` | `Module`, `Linear`, `Sigmoid`, `ReLU`, `Sequential` |
| `common/optimizers.py` | `src/core/optimizers.py` | `SGD`, `Adam` - 인터페이스 유지 |
| `common/trainer.py` (학습 루프) | `src/core/trainer.py` | `Trainer.fit(train_loader)` |
| `common/trainer.py` (평가 루프) | `src/core/evaluator.py` | `Evaluator.evaluate(test_loader)` |
| `common/trainer.py` (예측) | `src/core/predictor.py` | `Predictor.predict(images)` |
| - | `src/nn/conv.py` | `im2col`/`col2im`, `Conv2d`, `MaxPool2d`, `Flatten`, `Dropout` (신설) |
| - | `src/models/cnn.py` | CuPy 기반 CNN (신설) |
| - | `src/core/visualizer.py` | 예측 결과·샘플 시각화 (신설) |
| - | `src/core/logger.py` | epoch별 loss/metric 로그 (신설) |
| - | `src/utils/batching.py` | mini-batch 인덱스 생성·shuffle (신설) |
| - | `src/utils/random.py` | 난수 시드 고정 (신설) |
| - | `src/utils/io.py` | 파일 저장·로딩 보조 (신설) |
| - | `src/utils/checkpoints.py` | 모델 파라미터 `.npz` 저장·로딩 (신설) |
| - | `src/utils/training_plots.py` | 학습 곡선 PNG 저장 (신설) |

### 3.2. src 패키지 구조

`src` 패키지는 `nn`, `data`, `models`, `core`, `utils` 5개 하위 패키지로 구성된다.

```text
src/
├── __init__.py
├── nn/
│   ├── __init__.py
│   ├── activations.py
│   ├── layers.py
│   ├── losses.py
│   ├── metrics.py
│   └── conv.py
├── data/
│   ├── __init__.py
│   ├── mnist.py
│   └── dataloader.py
├── models/
│   ├── __init__.py
│   ├── mlp.py
│   └── cnn.py
├── core/
│   ├── __init__.py
│   ├── optimizers.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── predictor.py
│   ├── visualizer.py
│   └── logger.py
└── utils/
    ├── __init__.py
    ├── batching.py
    ├── random.py
    ├── io.py
    ├── checkpoints.py
    └── training_plots.py
```

### 3.3. 파일별 책임 범위

각 파일의 역할을 단일 책임 기준으로 확정한다.

| 파일 | 책임 |
|---|---|
| `src/nn/activations.py` | `sigmoid`, `softmax`, `identity`, `relu` - forward 전용 |
| `src/nn/layers.py` | `Module`, `Linear`, `Sigmoid`, `ReLU`, `Sequential` |
| `src/nn/losses.py` | 손실 함수 3종(`cross_entropy`, `binary_cross_entropy`, `mse`) + gradient 함수 3종 |
| `src/nn/metrics.py` | `accuracy`, `binary_accuracy`, `r2_score` |
| `src/nn/conv.py` | `im2col`/`col2im` + `Conv2d`, `MaxPool2d`, `Flatten`, `Dropout` |
| `src/data/mnist.py` | 로컬 gzip 로딩(`load_mnist`), `get_task_spec()`, `transform_targets()`, `MnistDataset` |
| `src/data/dataloader.py` | `__len__`·`__getitem__` 프로토콜 기반 범용 `DataLoader` |
| `src/models/mlp.py` | `src/nn/` 모듈을 조립한 NumPy 기반 MLP |
| `src/models/cnn.py` | CuPy 기반 CNN |
| `src/core/optimizers.py` | `SGD`, `Adam` - `model.params`/`grads` 기반 in-place 업데이트 |
| `src/core/trainer.py` | `Trainer.fit(train_loader)` - epoch·batch 학습 루프 |
| `src/core/evaluator.py` | `Evaluator.evaluate(test_loader)` - 평가 루프 |
| `src/core/predictor.py` | `Predictor.predict(images)` - task별 예측 후처리 |
| `src/core/visualizer.py` | 예측 결과·샘플 이미지 grid 시각화 |
| `src/core/logger.py` | epoch별 loss/metric 로그 기록, CSV 또는 dict 반환 |
| `src/utils/batching.py` | mini-batch 인덱스 생성, shuffle |
| `src/utils/random.py` | NumPy/CuPy 난수 시드 고정 |
| `src/utils/io.py` | 파일 저장·로딩 보조 함수 |
| `src/utils/checkpoints.py` | 모델 파라미터를 `.npz` 파일로 저장·로딩 |
| `src/utils/training_plots.py` | 학습 로그 loss/metric 곡선을 PNG로 저장 |

### 3.4. Stage별 구현 순서

구현은 Stage 0 설계 확정 후 Stage 1부터 파일 단위로 진행한다.

Stage 1부터 Stage 6까지의 Phase 분할은 아래와 같다.

| Stage | Phase | 대상 파일 | 테스트 파일 |
|---|---|---|---|
| 1 | 1.1 배치·난수 | `batching.py`, `random.py` | `test_batching.py`, `test_random.py` |
| 1 | 1.2 파일 입출력 | `io.py`, `checkpoints.py` | `test_io.py`, `test_checkpoints.py` |
| 1 | 1.3 시각화 유틸리티 | `training_plots.py` | `test_training_plots.py` |
| 2 | 2.1 MNIST 로딩 | `data/mnist.py` (`load_mnist`) | `test_mnist.py` |
| 2 | 2.2 Dataset | `data/mnist.py` (`MnistDataset`) | `test_dataset.py` |
| 2 | 2.3 DataLoader | `data/dataloader.py` | `test_dataloader.py` |
| 3 | 3.1 activation | `nn/activations.py` | `test_activations.py` |
| 3 | 3.2 loss | `nn/losses.py` | `test_losses.py` |
| 3 | 3.3 metric | `nn/metrics.py` | `test_metrics.py` |
| 3 | 3.4 layer | `nn/layers.py` | `test_layers.py` |
| 3 | 3.5 conv | `nn/conv.py` | `test_conv.py` |
| 4 | 4.1 MLP | `models/mlp.py` | `test_mlp.py` |
| 4 | 4.2 CNN | `models/cnn.py` | `test_cnn.py` |
| 5 | 5.1 optimizer | `core/optimizers.py` | `test_optimizers.py` |
| 5 | 5.2 Trainer·Evaluator | `core/trainer.py`, `core/evaluator.py` | `test_trainer.py`, `test_evaluator.py` |
| 5 | 5.3 Predictor·Visualizer | `core/predictor.py`, `core/visualizer.py` | `test_predictor.py`, `test_visualizer.py` |
| 5 | 5.4 Logger | `core/logger.py` | `test_logger.py` |
| 6 | 6.1 학습·평가 스크립트 | `scripts/train.py`, `scripts/evaluate.py` | `test_train.py`, `test_evaluate.py` |
| 6 | 6.2 예측·시각화 스크립트 | `scripts/predict.py`, `scripts/visualize.py` | `test_predict.py`, `test_visualize.py` |
| 6 | 6.3 배치 실험 스크립트 | `experiments/run_all.py` | - |

## 4. 사용법

각 Phase는 아래 절차를 반복하여 구현한다.

```text
1. 테스트 파일 작성 (tests/stageN/test_*.py)
2. pytest 실행 → 실패 확인
3. 구현 파일 작성 (src/...)
4. pytest 실행 → 통과 확인
5. 다음 Phase로 이동
```

Stage별 테스트 실행 예시는 다음과 같다.

```bash
conda run -n numpy_py311 pytest tests/stage1/ -q
conda run -n numpy_py311 pytest tests/stage2/ -q
conda run -n numpy_py311 pytest tests/stage3/ -q
```

## 5. 테스트

Phase 0.2는 계획 수립 단계이므로 대응하는 테스트 파일이 없다. TDD 절차와 테스트 원칙은 [[phase0.3_test-plan]]에서 상세히 정의한다.

## 6. 요약

레거시 common 모듈 6개는 새로운 `src/` 파일 20개에 역할별로 분리 매핑된다. `src/` 패키지는 `nn`, `data`, `models`, `core`, `utils` 5개 하위 패키지로 구성되며, Stage 1부터 Stage 6까지 의존성 순서에 따라 파일 단위로 구현한다. 각 파일은 구현 전에 테스트를 먼저 작성하는 TDD 순서를 따른다.

다음 Phase에서는 [[phase0.3_test-plan]]을 다룬다.
