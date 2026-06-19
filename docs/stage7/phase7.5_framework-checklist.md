---
tags: [stage7, framework, checklist, migration]
created: 2026-06-18
updated: 2026-06-20
---

# Phase 7.5 framework 연계 준비

## 1. 목적

이 문서는 NumPy/CuPy 기준 구현을 후속 PyTorch, TensorFlow, JAX 프로젝트로 옮길 때 유지해야 하는 인터페이스와 검증 기준을 정리한다.
후속 프로젝트는 내부 구현체만 각 프레임워크에 맞게 교체하고, 외부 사용법과 문서 흐름은 가능한 한 동일하게 유지한다.

## 2. 현재 기준 구현 상태

Stage 7 기준으로 MLP 산출물과 MLP 튜토리얼은 완료되었다.
CNN 실험 산출물과 CNN 튜토리얼은 CUDA 실행 환경 문제로 평가를 중지한 상태이며, 프레임워크 연계 체크리스트는 MLP 검증 결과와 Stage 1~6 테스트 통과 구조를 기준으로 작성한다.

현재 상태는 아래와 같다.

| 항목 | 상태 | 비고 |
|---|---|---|
| Stage 1~5 | 완료 | 설정, task, data, MLP, core, CLI 구현 |
| Stage 6 | 완료 | CuPy CNN 구현과 synthetic 통합 테스트 완료 |
| Stage 7.1 | 완료 | `scripts/*.py` `--model` 플래그 추가 |
| Stage 7.2 | 부분 완료 | MLP 3종 산출물 완료, CNN 산출물 미진행 |
| Stage 7.3~7.5 | 부분 완료 | MLP 튜토리얼 완료, CNN 튜토리얼 미진행 |
| Stage 7.6 | 완료 기준 | 후속 프레임워크 연계 규약 정리 |

## 3. 유지해야 하는 최상위 구조

후속 프레임워크 프로젝트는 아래 최상위 구조를 유지한다.
프레임워크별 차이는 `src/nn/`, `src/models/`, 일부 `src/core/` 내부 구현으로 흡수한다.

```text
src/
├── config.py
├── task.py
├── data/
├── nn/
├── models/
├── core/
└── utils/
scripts/
tests/
configs/
docs/
outputs/
```

각 위치의 역할은 아래 기준을 유지한다.

| 위치 | 유지할 역할 |
|---|---|
| `src/config.py` | 기본 경로, 실행 기본값, seed, hyperparameter 제공 |
| `src/task.py` | task별 output dimension, loss, metric, prediction mode 규약 관리 |
| `src/data/` | MNIST 로딩, dataset, dataloader 제공 |
| `src/nn/` | 프레임워크별 layer, activation, loss 구성요소 제공 |
| `src/models/` | `MLP`, `CNN` 모델 클래스 제공 |
| `src/core/` | trainer, evaluator, predictor, visualizer, experiment 실행 객체 제공 |
| `scripts/` | CLI 진입점 제공 |
| `tests/` | public interface 기준 검증 |

## 4. 유지해야 하는 task 규약

세 가지 task 이름과 target 변환은 후속 프로젝트에서도 변경하지 않는다.
데이터셋 계층에서 target을 변환하고, 모델과 core 객체는 변환된 target만 받는 구조를 유지한다.

| task | target 변환 | output_dim | prediction_mode | metric |
|---|---|---:|---|---|
| `multiclass` | one-hot digit label | 10 | `argmax` | accuracy |
| `binary` | `label % 2` | 1 | `threshold` | accuracy |
| `regression` | `label / 9.0` | 1 | `round_clip` | r2 score |

손실 함수 대응은 아래 기준을 유지한다.

| task | loss |
|---|---|
| `multiclass` | cross entropy |
| `binary` | binary cross entropy |
| `regression` | mean squared error |

## 5. 유지해야 하는 public interface

후속 프로젝트의 테스트와 튜토리얼이 같은 방식으로 작성되려면 아래 public interface를 우선 보존한다.
내부 구현이 tensor 기반으로 바뀌더라도 입력 인자 이름, 반환 dict 키, CLI 인자 이름은 최대한 동일하게 둔다.

| 대상 | interface | 유지 기준 |
|---|---|---|
| config | `get_default_config()` | dict 반환 |
| task | `get_task_spec(task)` | task spec dict 반환 |
| task | `transform_targets(labels, task)` | task별 target 배열 또는 tensor 반환 |
| data | `MnistDataset(split, task, ...)` | `__len__`, `__getitem__` 제공 |
| data | `DataLoader(dataset, batch_size, shuffle)` | batch iteration 제공 |
| models | `MLP(...)`, `CNN(...)` | `forward`, parameter 접근 방식 제공 |
| core | `Trainer.fit(train_loader)` | epoch log 반환 |
| core | `Evaluator.evaluate(test_loader)` | `loss`, `metric`, `num_samples` 반환 |
| core | `Predictor.predict(images)` | raw prediction과 decoded prediction 반환 |
| core | `Experiment(config)` | data, model, core 객체 조립 |
| scripts | `train.py`, `evaluate.py`, `predict.py`, `visualize.py` | `--task`, `--model`, path 인자 유지 |

## 6. PyTorch 마이그레이션 체크리스트

PyTorch 프로젝트를 시작할 때는 아래 순서로 이식한다.
각 단계는 NumPy/CuPy 기준 구현의 public interface와 테스트를 먼저 맞춘 뒤, PyTorch 관용 구현으로 내부를 정리한다.

### 6.1. 기본 구조

기본 구조 이식 항목은 아래와 같다.

| 항목 | 체크 |
|---|---|
| `src/config.py`의 config key 유지 | [ ] |
| `src/task.py`의 task 이름과 target 변환 유지 | [ ] |
| `scripts/*.py`의 CLI 인자 이름 유지 | [ ] |
| `tests/` stage 단위 구조 유지 | [ ] |
| `outputs/{task}/{model}/` 산출물 경로 유지 | [ ] |

### 6.2. 데이터 계층

데이터 계층 이식 항목은 아래와 같다.

| 항목 | 체크 |
|---|---|
| 로컬 MNIST gzip 파일 경로 규약 유지 | [ ] |
| image shape `(784,)` 또는 모델 입력 직전 reshape 규약 결정 | [ ] |
| task별 target dtype과 shape 유지 | [ ] |
| PyTorch `Dataset`으로 `MnistDataset` 이식 | [ ] |
| PyTorch `DataLoader` 사용 시 기존 batch 반환 형태와 호환 확인 | [ ] |

### 6.3. 모델 계층

모델 계층 이식 항목은 아래와 같다.

| 항목 | 체크 |
|---|---|
| `MLP` hidden 구조 `784 -> 256 -> 128 -> output_dim` 유지 | [ ] |
| `CNN` 입력 reshape 또는 channel dimension 규약 유지 | [ ] |
| task별 final layer output dimension 유지 | [ ] |
| loss 함수가 logit 입력을 받는지 확정 | [ ] |
| checkpoint 저장·로드 시 model class와 config 연결 방식 확정 | [ ] |

### 6.4. core 계층

core 계층 이식 항목은 아래와 같다.

| 항목 | 체크 |
|---|---|
| `Trainer.fit()` 반환 log key 유지 | [ ] |
| `Evaluator.evaluate()` 반환 key 유지 | [ ] |
| `Predictor.predict()` decoded prediction 규약 유지 | [ ] |
| `Visualizer`가 기존 output 파일명 유지 | [ ] |
| `Experiment`가 `config["model"]`로 `mlp`와 `cnn`을 분기 | [ ] |

### 6.5. 테스트와 검증

테스트 이식 항목은 아래와 같다.

| 항목 | 체크 |
|---|---|
| Stage 1 task spec 테스트 먼저 통과 | [ ] |
| Stage 2 synthetic dataset 또는 작은 MNIST fixture 테스트 통과 | [ ] |
| Stage 3 model forward shape 테스트 통과 | [ ] |
| Stage 4 trainer/evaluator/predictor 반환 dict 테스트 통과 | [ ] |
| Stage 5 CLI config assembly 테스트 통과 | [ ] |
| GPU 의존 테스트는 사용 가능 여부에 따라 skip 조건 명시 | [ ] |

## 7. 프레임워크별 주의사항

후속 프로젝트별로 아래 차이를 먼저 확인한다.
공통 interface를 유지하되, framework-specific 구현은 각 프로젝트 문서에 별도로 기록한다.

| 프레임워크 | 주의사항 |
|---|---|
| PyTorch | `torch.nn.Module`, `torch.optim`, autograd 기반으로 manual backward 제거 |
| TensorFlow | `tf.data`, `tf.keras.Model`, `GradientTape` 사용 여부 결정 |
| JAX | functional parameter tree, `jit`, `grad`, PRNG key 관리 방식 확정 |

## 8. 완료 기준

Phase 7.6은 후속 프레임워크 프로젝트를 시작하기 위한 interface 기준을 확정하면 완료로 본다.
CNN 실험 산출물은 현재 평가 중지 상태이므로 Phase 7.6 완료 조건에 포함하지 않는다.

완료 기준은 아래와 같다.

| 기준 | 상태 |
|---|---|
| 유지할 폴더 구조 정리 | 완료 |
| task 규약 정리 | 완료 |
| public interface 정리 | 완료 |
| PyTorch 마이그레이션 체크리스트 작성 | 완료 |
| TensorFlow, JAX 주의사항 기록 | 완료 |
| CNN 평가 중지 상태 명시 | 완료 |
