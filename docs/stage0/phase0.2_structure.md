---
tags: [stage0, structure, src, tests]
created: 2026-06-15
updated: 2026-06-17
---

# Phase 0.2 src · tests 구조 확정

## 1. 설계 원칙

프레임워크별 구현 차이를 하위에서 흡수하고 최상위 구조와 클라이언트 사용법은 동일하게 유지하는 원칙을 다음과 같이 확정한다.

- 4개 프레임워크(NumPy/CuPy, PyTorch, TensorFlow, JAX) 프로젝트가 동일한 `src/` 최상위 구조를 공유한다.
- 실행 객체는 `training/` 대신 `core/` 에 배치한다.
- 프레임워크별 차이는 `models/` 하위 구현에서 흡수하며 최상위 `nn/` 폴더는 사용하지 않는다.
- 레거시 `common/optimizers.py` 에 대응하는 `optimizers.py` 는 `core/` 에 배치한다.

## 2. src 패키지 구조

`src/` 패키지의 확정 구조는 다음과 같다.

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

각 위치의 책임은 다음과 같다.

| 위치 | 책임 |
|---|---|
| `src/config.py` | 기본 경로, seed, batch size, epoch, task, split 기본값 |
| `src/task.py` | task별 target 변환, output_dim, loss, metric, 후처리 규약 |
| `src/data/mnist.py` | 로컬 gz 파일 로딩, split 선택, MnistDataset |
| `src/data/dataloader.py` | 범용 DataLoader — shuffle, drop_last 지원 |
| `src/models/mlp.py` | NumPy MLP 생성, forward, backward, update |
| `src/models/cnn.py` | CuPy CNN 생성, forward, backward, update |
| `src/models/layers.py` | Linear 레이어 from-scratch 구현 |
| `src/models/activations.py` | sigmoid, softmax, identity, relu 활성화 함수 |
| `src/models/losses.py` | cross_entropy, binary_cross_entropy, mse 손실 함수 및 지표 |
| `src/core/optimizers.py` | SGD, Adam 옵티마이저 — model.params/grads 기반 in-place 업데이트 |
| `src/core/checkpoints.py` | 파라미터 저장·로딩 |
| `src/core/trainer.py` | 학습 루프, Dataloader 수신, loss/metric 집계 |
| `src/core/evaluator.py` | 평가 루프, Dataloader 수신, loss/metric 집계 |
| `src/core/predictor.py` | task별 예측 후처리 |
| `src/core/visualizer.py` | 학습 로그, 예측 결과, 샘플 이미지 시각화 |
| `src/core/experiment.py` | data, task, model, core 실행 객체 조립 |
| `src/utils/batching.py` | mini-batch 인덱스 생성, shuffle |
| `src/utils/random.py` | 난수 시드 고정 |
| `src/utils/io.py` | 파일 저장·로딩 보조 함수 |

## 3. tests 폴더 구조

테스트 코드는 `src/` 와 `scripts/` 의 대상별로 파일을 분리한다.

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

## 4. scripts와 core 관계

`scripts/` 클라이언트 코드는 내부 구현 모듈을 직접 조립하지 않고 `src/core/` 실행 객체를 참조한다.

| 클라이언트 파일 | 참조 대상 |
|---|---|
| `scripts/train.py` | `src/core/experiment.py`, `src/core/trainer.py` |
| `scripts/evaluate.py` | `src/core/experiment.py`, `src/core/evaluator.py` |
| `scripts/predict.py` | `src/core/experiment.py`, `src/core/predictor.py` |
| `scripts/visualize.py` | `src/core/experiment.py`, `src/core/visualizer.py` |
