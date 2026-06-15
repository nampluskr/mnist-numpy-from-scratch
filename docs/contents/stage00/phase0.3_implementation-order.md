---
tags: [stage00, implementation, tdd, conventions]
created: 2026-06-15
updated: 2026-06-15
---

# Phase 0.3 구현 순서 확정

## 1. Stage별 구현 순서

Stage 1부터 Stage 7까지의 구현 순서는 다음과 같다. 각 Phase는 소스 코드, 테스트 코드, 마크다운 보고서를 하나의 단위로 묶어 진행한다.

| Stage | 내용 | 핵심 파일 |
|---|---|---|
| Stage 1 | 기본 설정 · 과제 규약 | `config.py`, `task.py`, `utils/` |
| Stage 2 | MNIST 데이터 로더 | `data/mnist.py` |
| Stage 3 | NumPy MLP | `models/mlp.py`, `layers.py`, `activations.py`, `losses.py` |
| Stage 4 | 실행 객체 | `core/` 6개 파일 |
| Stage 5 | 클라이언트 코드 | `scripts/` 4개 파일 |
| Stage 6 | CuPy CNN | `models/cnn.py` |
| Stage 7 | 문서화 · 검증 | `docs/contents/` |

## 2. 공통 함수명 및 입출력 규약

파일별 공개 진입점과 입출력 규약은 다음과 같다. 이 규약은 후속 PyTorch, TensorFlow, JAX 프로젝트와 동일하게 유지한다.

| 파일 | 공개 진입점 | 입력 | 출력 |
|---|---|---|---|
| `src/config.py` | `get_default_config()` | 없음 | `dict` |
| `src/data/mnist.py` | `load_mnist(split)` | `split: str` | `(images, labels)` |
| `src/task.py` | `get_task_spec(task)` | `task: str` | `dict` |
| `src/task.py` | `transform_targets(labels, task)` | `labels: ndarray`, `task: str` | `ndarray` |
| `src/models/mlp.py` | `MLP` | config 또는 차원 인자 | model instance |
| `src/core/trainer.py` | `Trainer` | model, task spec, config | trainer instance |
| `src/core/evaluator.py` | `Evaluator` | model, task spec, config | evaluator instance |
| `src/core/predictor.py` | `Predictor` | model, task spec, config | predictor instance |
| `src/core/experiment.py` | `Experiment` | config | experiment instance |

입출력 세부 기준은 다음과 같다.

- `split` 허용값: `"train"`, `"test"`
- `task` 허용값: `"multiclass"`, `"binary"`, `"regression"`
- `load_mnist(split)` 반환: `images (N, 28, 28) uint8`, `labels (N,) uint8`
- `transform_targets()` 반환: 학습에 바로 사용 가능한 `float32` 배열
- `get_task_spec(task)` 최소 키: `task`, `output_dim`, `target_dtype`, `prediction_mode`
- `MLP.forward(x)` 반환: `(N, output_dim)` 배열
- `Trainer.fit()` 반환: epoch별 로그 dict 목록
- `Evaluator.evaluate()` 반환: `loss`, `metric`, `num_samples` 포함 dict
- `Predictor.predict()` 반환: raw prediction과 decoded prediction 포함 dict

## 3. TDD 원칙

Stage 1부터 구현보다 테스트를 먼저 작성한다. 실패 테스트 작성 원칙은 다음과 같다.

- public interface 기준으로 테스트를 먼저 작성하고, 내부 helper는 필요한 경우에만 간접 검증한다.
- 파일 단위로 진행하며, 해당 파일의 대응 테스트가 먼저 실패하는지 확인한 뒤 구현한다.
- unit test를 먼저 작성하고, 조합이 필요한 경우에만 통합 테스트를 추가한다.
- fixture는 재사용 가치가 있을 때만 `tests/conftest.py` 로 올리고, 파일 전용 데이터는 각 테스트 파일에 둔다.
- 숫자 비교는 exact equality 대신 tolerance 비교를 사용한다.
- MNIST 원본 전체 의존 테스트는 최소화하고 synthetic array 기반 테스트를 우선한다.

## 4. pytest 실행 명령

파일 단위 구현 흐름에서 사용하는 pytest 실행 명령은 다음과 같다.

| 목적 | 명령 |
|---|---|
| config 단일 파일 | `pytest tests/test_config.py -q` |
| task 단일 파일 | `pytest tests/test_task.py -q` |
| utils 단일 파일 | `pytest tests/utils/test_batching.py -q` |
| data 단일 파일 | `pytest tests/data/test_mnist.py -q` |
| models 단일 파일 | `pytest tests/models/test_mlp.py -q` |
| core 단일 파일 | `pytest tests/core/test_trainer.py -q` |
| 전체 스모크 확인 | `pytest tests/ -q` |
