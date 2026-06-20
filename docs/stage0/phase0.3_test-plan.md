---
tags: [docs, stage0, test-plan]
created: "2026-06-20"
updated: "2026-06-20"
---

# TDD 테스트 계획 수립

## 1. 개요

Stage 1부터 Stage 6까지 각 구현 파일에 대응하는 테스트 파일을 먼저 작성하는 TDD 원칙을 적용한다. 이 문서는 `tests/` 폴더 구조, 파일별 공개 인터페이스 규약, TDD 원칙, `pytest` 실행 명령을 확정한다. 확정된 계획은 Stage 1부터 구현 작업을 시작할 때 그대로 사용한다.

**목표**
- `tests/` 폴더를 Stage 단위로 구성하고 `__init__.py` 없이 동작하는 구조를 확정한다.
- 파일별 공개 인터페이스의 입력·출력 규약을 테스트 관점에서 확정한다.
- synthetic array 우선, tolerance 비교 기준 TDD 원칙을 확정한다.

## 2. 개념

### 2.1. TDD 순서 원칙

TDD는 테스트 실패 → 구현 → 테스트 통과의 짧은 사이클을 반복한다. 한 번에 하나의 파일 단위를 구현하며, 해당 파일의 대응 테스트가 먼저 실패하는 것을 확인한 뒤 구현한다.

단계별 절차는 다음과 같다.

```text
1. tests/stageN/test_*.py 작성  (테스트 실패 확인)
2. src/**/*.py 구현              (테스트 통과 확인)
3. pytest tests/stageN/ -q       (전체 Stage 검증)
4. 다음 Phase로 이동
```

### 2.2. 테스트 범위 원칙

MNIST 원본 전체(60,000장)에 의존하는 테스트는 실행 속도가 느리고 환경 의존성이 높다. 대신 작은 synthetic array를 직접 만들어 테스트를 작성하면 빠르고 결정론적인 검증이 가능하다.

적용 원칙은 다음과 같다.

- public interface 기준으로 테스트를 작성하고, 내부 helper는 필요한 경우에만 간접 검증한다.
- MNIST 원본 전체 의존 테스트는 최소화하고, synthetic array 기반 테스트를 우선 작성한다.
- 숫자 비교는 부동소수점 오차를 고려하여 `np.allclose` 또는 `pytest.approx`를 사용한다.
- unit test를 먼저 작성하고, 여러 모듈 조합이 필요한 경우에만 통합 테스트를 추가한다.
- fixture는 재사용 가치가 있는 경우에만 `tests/conftest.py`로 올리고, 파일 전용 데이터는 각 테스트 파일 내부에 둔다.

## 3. 구현

### 3.1. tests 폴더 구조

테스트 코드는 Stage 단위로 폴더를 분리하고, `__init__.py`를 두지 않는다.

```text
tests/
├── conftest.py
├── stage1/
│   ├── test_batching.py
│   ├── test_random.py
│   ├── test_io.py
│   ├── test_checkpoints.py
│   └── test_training_plots.py
├── stage2/
│   ├── test_mnist.py
│   ├── test_dataset.py
│   └── test_dataloader.py
├── stage3/
│   ├── test_activations.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   ├── test_layers.py
│   └── test_conv.py
├── stage4/
│   ├── test_mlp.py
│   └── test_cnn.py
├── stage5/
│   ├── test_optimizers.py
│   ├── test_trainer.py
│   ├── test_evaluator.py
│   ├── test_predictor.py
│   └── test_visualizer.py
└── stage6/
    ├── test_train.py
    ├── test_evaluate.py
    ├── test_predict.py
    └── test_visualize.py
```

`__init__.py`를 두지 않는 이유는 `pyproject.toml`의 `pythonpath` 설정으로 `src` 모듈 import를 해결하기 때문이다. `conftest.py`는 `tests/` 루트에 하나만 두고 재사용 fixture만 관리한다.

### 3.2. 파일별 공개 인터페이스 테스트 규약

각 구현 파일의 공개 진입점을 기준으로 테스트 항목을 확정한다.

Stage 1 테스트 대상은 아래와 같다.

| 테스트 파일 | 검증 대상 | 주요 검증 내용 |
|---|---|---|
| `test_batching.py` | `get_batches` | 배치 수, 배치 크기, shuffle 여부, 전체 커버리지 |
| `test_random.py` | `set_seed` | 동일 seed로 동일 배열 생성 여부 |
| `test_io.py` | `save_array`, `load_array` | 저장·로딩 왕복, shape·dtype 보존 |
| `test_checkpoints.py` | `save_checkpoint`, `load_checkpoint` | `.npz` 저장·로딩, CuPy array NumPy 변환 |
| `test_training_plots.py` | `plot_training_curves` | PNG 파일 생성 여부, epoch 수 일치 |

Stage 2 테스트 대상은 아래와 같다.

| 테스트 파일 | 검증 대상 | 주요 검증 내용 |
|---|---|---|
| `test_mnist.py` | `load_mnist` | images shape `(N,28,28)` uint8, labels shape `(N,)` uint8, split 분리 |
| `test_dataset.py` | `MnistDataset` | task별 target shape·dtype, `__len__`, `__getitem__` tuple 반환 |
| `test_dataloader.py` | `DataLoader` | 배치 shape, shuffle, `__len__`, 전체 반복 커버리지 |

Stage 3 테스트 대상은 아래와 같다.

| 테스트 파일 | 검증 대상 | 주요 검증 내용 |
|---|---|---|
| `test_activations.py` | `sigmoid`, `softmax`, `identity`, `relu` | 출력 shape 보존, 값 범위, 수치 안정성 |
| `test_losses.py` | 손실 3종 + gradient 3종 | scalar 반환, gradient shape, 양수 값 |
| `test_metrics.py` | `accuracy`, `binary_accuracy`, `r2_score` | 완전 정답·완전 오답 경계 조건, scalar 반환 |
| `test_layers.py` | `Linear`, `Sigmoid`, `ReLU`, `Sequential` | forward shape, backward gradient shape, in-place grad 저장 |
| `test_conv.py` | `Conv2d`, `MaxPool2d`, `Flatten`, `Dropout` | im2col shape, forward·backward 일관성 |

Stage 4 테스트 대상은 아래와 같다.

| 테스트 파일 | 검증 대상 | 주요 검증 내용 |
|---|---|---|
| `test_mlp.py` | `MLP` | 생성, forward shape `(N, output_dim)`, backward 후 grad 변화, params/grads 리스트 |
| `test_cnn.py` | `CNN` | forward shape, CuPy array 처리, GPU 미사용 환경 skip |

Stage 5 테스트 대상은 아래와 같다.

| 테스트 파일 | 검증 대상 | 주요 검증 내용 |
|---|---|---|
| `test_optimizers.py` | `SGD`, `Adam` | `step()` 후 params 변화, lr 민감도, 반환값 없음 |
| `test_trainer.py` | `Trainer.fit` | 1 epoch 실행, 로그 dict 반환, DataLoader 수신 |
| `test_evaluator.py` | `Evaluator.evaluate` | loss·metric·num_samples 키 포함 dict 반환 |
| `test_predictor.py` | `Predictor.predict` | task별 후처리 결과 shape·값 범위 |
| `test_visualizer.py` | `Visualizer` | 이미지 파일 생성 여부 |

Stage 6 테스트 대상은 아래와 같다.

| 테스트 파일 | 검증 대상 | 주요 검증 내용 |
|---|---|---|
| `test_train.py` | `scripts/train.py` | CLI 인자 처리, checkpoint 파일 생성 |
| `test_evaluate.py` | `scripts/evaluate.py` | checkpoint 로딩, 결과 출력 |
| `test_predict.py` | `scripts/predict.py` | 예측 결과 형식 |
| `test_visualize.py` | `scripts/visualize.py` | 이미지 파일 저장 |

### 3.3. 테스트 우선순위

구현 순서에 맞춘 테스트 우선순위는 아래와 같다.

| 순위 | 테스트 파일 | 이유 |
|---|---|---|
| 1 | `tests/stage1/test_batching.py` | 데이터 순회의 기반, 가장 간단한 입력 |
| 2 | `tests/stage2/test_mnist.py` | 실제 데이터 로딩 검증, 이후 모든 Stage의 의존 대상 |
| 3 | `tests/stage3/test_layers.py` | `Linear` forward/backward, in-place grad 저장 |
| 4 | `tests/stage4/test_mlp.py` | MLP 생성, forward shape, parameter update 흐름 |
| 5 | `tests/stage5/test_optimizers.py` | SGD, Adam params in-place 업데이트 |
| 6 | `tests/stage5/test_trainer.py` | 학습 루프와 batch 처리 |
| 7 | `tests/stage5/test_evaluator.py` | 평가 루프와 metric 집계 |

## 4. 사용법

`pyproject.toml`에 `pythonpath` 설정이 필요하다. 프로젝트 루트에 아래 내용이 있어야 한다.

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
```

이 설정으로 `tests/` 하위 파일에서 `from src.data.mnist import load_mnist`처럼 직접 import할 수 있다.

Stage별 테스트 실행 명령은 다음과 같다.

```bash
conda run -n numpy_py311 pytest tests/stage1/ -q
conda run -n numpy_py311 pytest tests/stage2/ -q
conda run -n numpy_py311 pytest tests/stage3/ -q
conda run -n numpy_py311 pytest tests/stage4/ -q
conda run -n numpy_py311 pytest tests/stage5/ -q
conda run -n numpy_py311 pytest tests/stage6/ -q
```

단일 파일 실행과 전체 스모크 확인은 다음과 같다.

```bash
conda run -n numpy_py311 pytest tests/stage3/test_layers.py -q
conda run -n numpy_py311 pytest tests/ -q
```

## 5. 테스트

Phase 0.3은 계획 수립 단계이므로 대응하는 테스트 파일이 없다. Stage 1부터 각 Phase 구현 시작 전에 대응 테스트 파일을 먼저 작성한다.

## 6. 요약

`tests/` 폴더는 Stage 1부터 Stage 6까지 6개 하위 폴더로 구성하며 `__init__.py`를 두지 않는다. 각 테스트 파일은 구현 파일보다 먼저 작성하고, synthetic array 기반으로 MNIST 원본 의존성을 최소화한다. 숫자 비교는 tolerance 비교를 기준으로 하며, 재사용 fixture만 `conftest.py`로 올린다.

다음 Phase에서는 [[phase1.1_batching-and-random]]을 다룬다.
