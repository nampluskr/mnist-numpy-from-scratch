---
tags: [project, stage0]
created: 2026-06-17
updated: 2026-06-17
---

# Phase 0.2 구현 계획 수립

이 문서는 레거시 common 모듈과 src 파일의 1:1 매핑, src 패키지 구조와 각 파일의 책임 범위, Stage 1~7 구현 순서와 Phase 단위 분할을 확정한다.

## 1. 레거시 common 모듈 → src 파일 매핑

레거시 6개 common 모듈을 src 구현 파일에 대응시킨다.

`common/functions.py`는 활성화 함수, 손실 함수, 평가 지표, 기울기 함수를 모두 포함하므로 역할별로 분리하여 여러 src 파일에 매핑한다. `common/trainer.py`는 학습·평가·예측 역할이 혼합되어 있으므로 세 개의 src 파일로 분리한다.

| 레거시 | src 대응 | 비고 |
|---|---|---|
| `common/mnist.py` | `src/data/mnist.py` | 함수명 변경: `load_images`/`load_labels` → `load_mnist` 내부 처리, `MnistDataset` 추가 |
| `common/dataloader.py` | `src/data/dataloader.py` | 이미지·레이블 배열 직접 수신 → Dataset 프로토콜 기반으로 전환 |
| `common/functions.py` (활성화) | `src/models/activations.py` | `sigmoid`, `softmax`, `identity`, `relu` |
| `common/functions.py` (손실·지표) | `src/models/losses.py` | `cross_entropy`, `binary_cross_entropy`, `mse`, `accuracy`, `binary_accuracy`, `r2_score` |
| `common/modules.py` | `src/models/layers.py` | `Linear`, `Sigmoid`, `ReLU`, `Sequential` |
| `common/modules.py` + 스크립트 조합 | `src/models/mlp.py` | forward, backward, update를 캡슐화한 MLP |
| `common/optimizers.py` | `src/core/optimizers.py` | `SGD`, `Adam` - 레거시와 동일 인터페이스 유지 |
| `common/trainer.py` (학습 루프) | `src/core/trainer.py` | `Trainer.fit(train_loader)` |
| `common/trainer.py` (평가 루프) | `src/core/evaluator.py` | `Evaluator.evaluate(test_loader)` |
| `common/trainer.py` (예측) | `src/core/predictor.py` | `Predictor.predict(images)` |
| 스크립트 하이퍼파라미터 | `src/config.py` | `get_default_config()` |
| 스크립트 target 변환 | `src/task.py` | `get_task_spec(task)`, `transform_targets(labels, task)` |
| - | `src/utils/batching.py` | mini-batch 인덱스·shuffle (레거시에 없는 신설) |
| - | `src/utils/random.py` | 난수 시드 고정 (레거시에 없는 신설) |
| - | `src/utils/io.py` | 파일 저장·로딩 보조 (레거시에 없는 신설) |
| - | `src/core/checkpoints.py` | 모델 파라미터 저장·로딩 (레거시에 없는 신설) |
| - | `src/core/experiment.py` | 실행 객체 조립 최상위 진입점 (레거시에 없는 신설) |
| - | `src/core/visualizer.py` | 학습 로그·예측 결과 시각화 (레거시에 없는 신설) |

## 2. src 패키지 구조

`src` 패키지는 data, models, core, utils 4개 하위 패키지로 구성된다.

```text
src/
├── __init__.py
├── config.py
├── task.py
├── data/
│   ├── __init__.py
│   ├── mnist.py
│   └── dataloader.py
├── models/
│   ├── __init__.py
│   ├── mlp.py
│   ├── cnn.py
│   ├── layers.py
│   ├── activations.py
│   └── losses.py
├── core/
│   ├── __init__.py
│   ├── optimizers.py
│   ├── checkpoints.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── predictor.py
│   ├── visualizer.py
│   └── experiment.py
└── utils/
    ├── __init__.py
    ├── batching.py
    ├── random.py
    └── io.py
```

## 3. 각 파일의 책임 범위

각 파일의 역할을 단일 책임 기준으로 확정한다.

| 파일 | 책임 |
|---|---|
| `src/config.py` | 기본 경로, seed, batch_size, num_epochs, task, split 반환 |
| `src/task.py` | task별 output_dim, activation, loss, metric, prediction_mode 규약; target 변환 헬퍼 |
| `src/data/mnist.py` | 로컬 gzip 로딩(`load_mnist`), `MnistDataset` (정규화·target 변환 내장) |
| `src/data/dataloader.py` | `__len__`·`__getitem__` 프로토콜 기반 범용 `DataLoader` |
| `src/models/mlp.py` | NumPy 기반 MLP forward, backward, update 캡슐화 |
| `src/models/cnn.py` | CuPy 기반 CNN forward, backward, update 캡슐화 |
| `src/models/layers.py` | `Linear`, `Sigmoid`, `ReLU`, `Sequential` from-scratch 구현 |
| `src/models/activations.py` | `sigmoid`, `softmax`, `identity`, `relu` - forward 전용 |
| `src/models/losses.py` | 손실 함수 3종 + 평가 지표 3종 |
| `src/core/optimizers.py` | `SGD`, `Adam` - `model.params`/`grads` 기반 in-place 업데이트 |
| `src/core/checkpoints.py` | 모델 파라미터를 파일로 저장·로딩 |
| `src/core/trainer.py` | `Trainer.fit(train_loader)` - epoch·batch 학습 루프 |
| `src/core/evaluator.py` | `Evaluator.evaluate(test_loader)` - 평가 루프 |
| `src/core/predictor.py` | `Predictor.predict(images)` - task별 예측 후처리 |
| `src/core/visualizer.py` | 학습 로그·예측 결과·샘플 이미지 시각화 |
| `src/core/experiment.py` | dataset, dataloader, task spec, model, optimizer, 실행 객체 조립 |
| `src/utils/batching.py` | mini-batch 인덱스 생성, shuffle |
| `src/utils/random.py` | 난수 시드 고정 |
| `src/utils/io.py` | 파일 저장·로딩 보조 함수 |

## 4. Stage별 구현 순서 및 Phase 단위 분할

구현은 Stage 0 설계 확정 후 Stage 1부터 파일 단위로 진행한다.

### 4.1. Stage 1 기본 설정 · 과제 규약

후속 Stage의 의존성 기반이 되는 설정 파일과 task 규약을 먼저 구현한다.

| Phase | 대상 파일 | 테스트 파일 |
|---|---|---|
| 1.1 config | `src/config.py` | `tests/stage1/test_config.py` |
| 1.2 task | `src/task.py` | `tests/stage1/test_task.py` |
| 1.3 utils | `src/utils/batching.py`, `random.py`, `io.py` | `tests/stage1/test_batching.py`, `test_random.py`, `test_io.py` |

### 4.2. Stage 2 MNIST 데이터 로더

데이터 로딩과 배치 처리 구조를 확립한다.

| Phase | 대상 파일 | 테스트 파일 |
|---|---|---|
| 2.1 mnist | `src/data/mnist.py` (`load_mnist`) | `tests/stage2/test_mnist.py` |
| 2.2 dataset | `src/data/mnist.py` (`MnistDataset` 추가) | `tests/stage2/test_dataset.py` |
| 2.3 dataloader | `src/data/dataloader.py` | `tests/stage2/test_dataloader.py` |

### 4.3. Stage 3 NumPy MLP

NumPy 기반 MLP와 from-scratch 구성 요소를 구현한다.

| Phase | 대상 파일 | 테스트 파일 |
|---|---|---|
| 3.1 mlp | `src/models/mlp.py` | `tests/stage3/test_mlp.py` |
| 3.2 layers · activations · losses | `src/models/layers.py`, `activations.py`, `losses.py` | `tests/stage3/test_layers.py`, `test_activations.py`, `test_losses.py` |

### 4.4. Stage 4 실행 객체

학습·평가·예측·저장·시각화를 담당하는 core 실행 객체를 구현한다.

| Phase | 대상 파일 | 테스트 파일 |
|---|---|---|
| 4.1 optimizers | `src/core/optimizers.py` | `tests/stage4/test_optimizers.py` |
| 4.2 checkpoints | `src/core/checkpoints.py` | `tests/stage4/test_checkpoints.py` |
| 4.3 trainer | `src/core/trainer.py` | `tests/stage4/test_trainer.py` |
| 4.4 evaluator | `src/core/evaluator.py` | `tests/stage4/test_evaluator.py` |
| 4.5 predictor | `src/core/predictor.py` | `tests/stage4/test_predictor.py` |
| 4.6 experiment | `src/core/experiment.py` | `tests/stage4/test_experiment.py` |
| 4.7 visualizer | `src/core/visualizer.py` | `tests/stage4/test_visualizer.py` |

### 4.5. Stage 5 클라이언트 코드

`core` 실행 객체를 조합하는 CLI 진입점을 구현한다.

| Phase | 대상 파일 | 테스트 파일 |
|---|---|---|
| 5.1 train | `scripts/train.py` | `tests/stage5/test_train.py` |
| 5.2 evaluate | `scripts/evaluate.py` | `tests/stage5/test_evaluate.py` |
| 5.3 predict | `scripts/predict.py` | `tests/stage5/test_predict.py` |
| 5.4 visualize | `scripts/visualize.py` | `tests/stage5/test_visualize.py` |

### 4.6. Stage 6 CuPy CNN

GPU 기반 CNN을 구현하고 기존 core 실행 객체와 연동을 검증한다.

| Phase | 대상 파일 | 테스트 파일 |
|---|---|---|
| 6.1 cnn | `src/models/cnn.py` | `tests/stage6/test_cnn.py` |
| 6.2 CNN-core 연동 | - | `tests/stage6/test_experiment.py` |

### 4.7. Stage 7 문서화 · 전체 검증

Jupyter Book 챕터 작성, 실행 예제 정리, 후속 프레임워크 연계 기준을 수립한다.

| Phase | 대상 |
|---|---|
| 7.1 튜토리얼 | `docs/stage7/` 챕터 |
| 7.2 실행 예제 | `outputs/` 결과 정리, `docs/stage7/phase7.2_results.md` |
| 7.3 프레임워크 연계 | `docs/stage7/phase7.3_framework-checklist.md` |
