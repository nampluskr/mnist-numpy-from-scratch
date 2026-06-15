# Deep Learning from Scratch with Numpy/Cupy 진행 이력

이 문서는 프로젝트의 Stage - Phase - Task 단위 실제 진행 내용을 기록한다.
PROJECT-TODO.md 체크박스 항목과 1:1 대응하며, 완료된 Task 에만 내용을 기입한다.

> 생성일시: 260608-133851
> 수정일시: 260608-142159
> 주제: Machine Learning

**목차**

1. [Stage 0 프로젝트 설계 재검토](#1-stage-0-프로젝트-설계-재검토)

## 1. Stage 0 프로젝트 설계 재검토

### 1.1. Phase 0.1 기존 계획 및 레거시 입력 재검토

#### [x] `_project/sessions/260608-133851_session-handoff.md` 기준의 기존 Stage - Phase - Task 구성을 재검토한다.

> 참조 세션: `_project/sessions/260608-140139_session-handoff.md`

**진행 내용**

기존 프로젝트 초기화 세션에서 작성된 Stage - Phase - Task 구성을 실제 구현 흐름에 맞게 재검토하였다.

- 기존 계획은 설계 검토, 데이터 구성, 모델 구현, 문서화, 테스트 정리가 Stage 1부터 Stage 6까지 나뉘어 있었다.
- 사용자의 추가 지시에 따라 프로젝트 설계 재검토 관련 내용은 Stage 0으로 이동하는 것이 적절하다고 판단하였다.
- Stage 1부터는 실제 코드 파일과 대응 테스트 파일을 함께 작성하는 단위로 진행해야 한다고 재정의하였다.

**산출물**

이 Task 에서 갱신된 문서는 기존 계획의 재정의 결과를 반영한다.

| 파일/산출물 | 내용 |
| --- | --- |
| `_project/PROJECT.md` | Stage 0 설계 재검토와 Stage 1 이후 파일·테스트 단위 진행 단계 반영 |
| `_project/PROJECT-TODO.md` | 기존 Stage - Phase - Task 구성을 새 체계로 재작성 |
| `_project/sessions/260608-140139_session-handoff.md` | 다음 작업 목록과 새 세션 시작 지시문 갱신 |

**결정사항**

이번 재검토에서 확정한 진행 체계는 다음 구현 단계의 기준이 된다.

| 항목 | 결정 내용 |
| --- | --- |
| 설계 재검토 위치 | 프로젝트 설계 재검토 관련 내용은 Stage 0으로 이동한다. |
| 구현 시작 위치 | Stage 1부터 실제 코드 작업과 테스트를 진행한다. |
| Task 단위 | Task는 `소스 파일 / 테스트 파일 테스트` 형식으로 정의한다. |

#### [x] `legacy/src/`의 MNIST MLP 레거시 코드 3개 파일을 구현 입력으로 확인한다.

> 참조 세션: `_project/sessions/260608-133851_session-handoff.md`

**진행 내용**

사용자가 제공한 기존 3가지 MNIST MLP 작업 코드가 `legacy/src/` 폴더에 배치되었음을 확인하였다.

- `mnist-multiclass-mlp.py`는 softmax와 cross entropy를 사용하는 10-class classification 레거시 코드이다.
- `mnist-binary-mlp.py`는 홀수·짝수 변환 규칙과 binary cross entropy를 사용하는 binary classification 레거시 코드이다.
- `mnist-regression-mlp.py`는 레이블을 0에서 1 사이 값으로 정규화하고 MSE를 사용하는 regression 레거시 코드이다.

**산출물**

이 Task 에서 확인한 레거시 파일은 다음 구현 단계의 분석 입력으로 사용한다.

| 파일/산출물 | 내용 |
| --- | --- |
| `legacy/src/mnist-multiclass-mlp.py` | MNIST multiclass MLP 레거시 코드 |
| `legacy/src/mnist-binary-mlp.py` | MNIST binary MLP 레거시 코드 |
| `legacy/src/mnist-regression-mlp.py` | MNIST regression MLP 레거시 코드 |

#### [x] 레거시 코드의 데이터 처리, 모델, 학습, 평가 흐름을 파일 단위 구현 계획에 매핑한다.

> 참조 세션: `_project/sessions/260608-140139_session-handoff.md`

**진행 내용**

레거시 코드 3개 파일을 공통 파이프라인과 task별 차이로 분리하여 Stage 1 이후 구현 파일에 매핑하였다.

- 세 레거시 코드는 모두 `train/test` split 로딩, 이미지 정규화, 3-layer MLP, mini-batch SGD, test 평가, 샘플 예측 출력 흐름을 공유한다고 정리하였다.
- multiclass는 one-hot target, `softmax`, `cross_entropy`, `accuracy` 조합으로 정리하였다.
- binary는 odd/even 이진화 target, `sigmoid`, `binary_cross_entropy`, `binary_accuracy` 조합으로 정리하였다.
- regression은 `label / 9.0` target, `identity`, `mse`, `r2_score` 조합으로 정리하였다.
- 공통 로딩 책임은 `src/data/mnist.py`, task별 규약은 `src/task.py`, 학습/평가 루프는 `src/core/` 계층, manual gradient 구현은 `src/models/` 하위로 배치하기로 정리하였다.

**산출물**

이 Task 에서 갱신된 문서는 레거시 구현의 파일 단위 매핑 기준을 포함한다.

| 파일/산출물 | 내용 |
| --- | --- |
| `_project/PROJECT.md` | 레거시 코드와 구현 파일 매핑 기준 추가 |
| `_project/PROJECT-TODO.md` | Phase 0.1 마지막 미완료 Task 완료 반영 |

### 1.2. Phase 0.2 공통 `src` 및 `tests` 구조 확정

#### [x] 4개 프레임워크 프로젝트가 공유할 `src` 최상위 구조를 확정한다.

> 참조 세션: `_project/sessions/260608-140139_session-handoff.md`

**진행 내용**

NumPy/CuPy, PyTorch, TensorFlow, JAX 프로젝트가 동일하게 유지할 `src` 최상위 구조를 확정하였다.

- 공통 최상위 구조는 `config.py`, `data/`, `task.py`, `models/`, `core/`, `utils/`로 정리하였다.
- 프레임워크별 차이는 `models/` 하위 파일 또는 하위 폴더에서 흡수하기로 하였다.
- `src` 구조는 후속 프로젝트 초기화 시 기준 구조로 재사용한다.

**산출물**

이 Task 에서 갱신된 문서는 공통 `src` 구조를 명시한다.

| 파일/산출물 | 내용 |
| --- | --- |
| `_project/PROJECT.md` | 확정된 `src` 패키지 구조 문서화 |
| `_project/PROJECT-TODO.md` | Stage 0 공통 구조 확정 Task 완료 상태 반영 |

#### [x] `training/` 대신 `core/`를 사용하는 실행 객체 배치 원칙을 확정한다.

> 참조 세션: `_project/sessions/260608-140139_session-handoff.md`

**진행 내용**

`scripts/`에서 직접 참조할 실행 객체를 `core/`에 배치하는 원칙을 확정하였다.

- `training/`은 학습 루프에 의미가 치우치므로 평가, 예측, 시각화 실행 객체까지 포괄하는 이름으로 `core/`를 선택하였다.
- `core/`에는 `experiment.py`, `trainer.py`, `evaluator.py`, `predictor.py`, `visualizer.py`, `checkpoints.py`를 배치한다.
- `scripts/`는 내부 구현 모듈을 직접 조립하지 않고 `core` 실행 객체를 호출한다.

**결정사항**

실행 계층은 클라이언트 코드의 공통 사용법을 유지하는 기준 역할을 한다.

| 항목 | 결정 내용 |
| --- | --- |
| 실행 객체 폴더 | `src/core/` |
| 클라이언트 참조 | `scripts/`는 `src/core/` 객체를 참조한다. |

#### [x] 최상위 `nn/` 폴더를 제외하고 프레임워크별 차이를 `models/` 하위에서 흡수하는 원칙을 확정한다.

> 참조 세션: `_project/sessions/260608-140139_session-handoff.md`

**진행 내용**

4개 프레임워크 프로젝트가 같은 `src` 최상위 구조를 갖도록 최상위 `nn/` 폴더를 제외하였다.

- PyTorch 프로젝트에서는 `torch.nn`을 사용하므로 최상위 `nn/` 폴더를 강제하지 않는 것이 적절하다고 판단하였다.
- NumPy/CuPy에서 필요한 from-scratch 하위 구현은 `models/` 내부에 배치할 수 있도록 하였다.
- `models/` 하위 폴더 구조는 프레임워크별로 달라질 수 있도록 허용하였다.

**결정사항**

공통 구조는 최상위 폴더 수준에서 유지하고, 프레임워크별 구현 차이는 하위 구조에서 처리한다.

| 항목 | 결정 내용 |
| --- | --- |
| 최상위 `nn/` | 사용하지 않는다. |
| 구현 차이 흡수 위치 | `src/models/` 하위 구조 |

#### [x] 테스트 대상별 파일이 생성되도록 `tests` 폴더 구조를 확정한다.

> 참조 세션: `_project/sessions/260608-140139_session-handoff.md`

**진행 내용**

테스트 코드는 대상 파일과 대응되는 테스트 파일을 두는 방식으로 구조를 확정하였다.

- `tests/test_config.py`, `tests/test_task.py`는 `src/config.py`, `src/task.py`에 대응한다.
- `tests/data/`, `tests/models/`, `tests/core/`, `tests/utils/`는 `src` 하위 폴더와 대응한다.
- `tests/scripts/`는 `scripts/` 클라이언트 코드와 대응한다.
- 공통 fixture는 `tests/conftest.py`에 배치한다.

**산출물**

이 Task 에서 갱신된 문서는 테스트 폴더 구조를 포함한다.

| 파일/산출물 | 내용 |
| --- | --- |
| `_project/PROJECT.md` | 확정된 `tests` 폴더 구조 문서화 |
| `_project/PROJECT-TODO.md` | 테스트 구조 확정 Task 완료 상태 반영 |

### 1.3. Phase 0.3 파일 단위 코드·테스트 구현 순서 확정

#### [x] Stage 1 이후 Task 단위를 `소스 파일 코드 작성` / `테스트 파일 테스트 작성` 형식으로 재정의한다.

> 참조 세션: `_project/sessions/260608-140139_session-handoff.md`

**진행 내용**

Stage 1 이후의 모든 실제 구현 Task를 코드 작성 Task와 테스트 작성 Task로 분리하여 재정의하였다.

- Stage 1은 `src/config.py` 코드 작성, `tests/test_config.py` 테스트 작성, `src/task.py` 코드 작성, `tests/test_task.py` 테스트 작성, `utils` 코드 및 대응 테스트 작성으로 구성하였다.
- Stage 2는 `src/data/mnist.py` 코드 작성과 `tests/data/test_mnist.py` 테스트 작성 중심의 데이터 로더 구현으로 구성하였다.
- Stage 3은 `src/models/mlp.py` 코드 작성과 `tests/models/test_mlp.py` 테스트 작성 중심의 NumPy MLP 구현으로 구성하였다.
- Stage 4는 `src/core/` 실행 객체 코드 작성 Task와 `tests/core/` 대응 테스트 작성 Task로 구성하였다.
- Stage 5는 `scripts/` 클라이언트 코드 작성 Task와 `tests/scripts/` 대응 테스트 작성 Task로 구성하였다.
- Stage 6은 `src/models/cnn.py` 코드 작성과 `tests/models/test_cnn.py` 테스트 작성 중심의 CuPy CNN 구현으로 구성하였다.

**산출물**

이 Task 에서 갱신된 작업 목록은 다음 구현 세션의 직접 기준이다.

| 파일/산출물 | 내용 |
| --- | --- |
| `_project/PROJECT.md` | 새 진행 단계 요약 반영 |
| `_project/PROJECT-TODO.md` | Stage 1 이후 코드 작성 Task와 테스트 작성 Task 분리 반영 |

**결정사항**

구현 Task는 기능 묶음이 아니라 검증 가능한 파일 단위로 관리한다.

| 항목 | 결정 내용 |
| --- | --- |
| Task 표기 | `소스 파일 코드 작성` / `테스트 파일 테스트 작성` |
| 진행 기준 | 코드와 테스트를 별도 체크 가능한 단위로 분리하여 Stage 1 이후 Task로 배치한다. |

#### [x] 공통 함수명과 입력·출력 규약을 파일별로 정의한다.

> 참조 세션: `_project/sessions/260608-140139_session-handoff.md`

**진행 내용**

Stage 1 초기 구현에서 바로 사용할 공통 공개 진입점과 입력·출력 규약을 파일별로 정의하였다.

- `src/config.py`는 `get_default_config()`를 공개 진입점으로 두고 기본 경로, seed, batch size, epoch, task, split 값을 반환하도록 정리하였다.
- `src/data/mnist.py`는 `load_mnist(split)`를 기준으로 원본 `images`, `labels` 배열을 반환하도록 정리하였다.
- `src/task.py`는 `get_task_spec(task)`와 `transform_targets(labels, task)`를 기준 인터페이스로 확정하였다.
- `src/models/mlp.py`, `src/core/trainer.py`, `src/core/evaluator.py`, `src/core/predictor.py`, `src/core/experiment.py`는 클래스 기반 공개 진입점을 사용하는 방향으로 정리하였다.
- `split`, `task`, target dtype, prediction 반환 형식 등 공통 입력·출력 규약을 문서화하였다.

**산출물**

이 Task 에서 갱신된 문서는 파일별 공개 진입점과 반환 규약을 포함한다.

| 파일/산출물 | 내용 |
| --- | --- |
| `_project/PROJECT.md` | 공통 함수명과 입력·출력 규약 추가 |
| `_project/PROJECT-TODO.md` | Phase 0.3 함수명/규약 정의 Task 완료 반영 |

#### [x] 구현 전 실패 테스트 작성 원칙과 `pytest` 실행 명령을 정리한다.

> 참조 세션: `_project/sessions/260608-140139_session-handoff.md`

**진행 내용**

TDD 기반 구현을 바로 시작할 수 있도록 실패 테스트 작성 원칙과 기본 `pytest` 실행 명령을 정리하였다.

- public interface 기준으로 실패 테스트를 먼저 작성하고, 파일 단위로 구현을 진행하는 원칙을 확정하였다.
- unit test 우선, 필요 시 통합 테스트 확장, synthetic array 우선 사용, tolerance 비교 사용 원칙을 정리하였다.
- `tests/test_config.py`, `tests/test_task.py`, `tests/data/test_mnist.py`, `tests/core/test_trainer.py`, `tests -q` 실행 명령을 초기 기준으로 정리하였다.

**산출물**

이 Task 에서 갱신된 문서는 TDD 실행 기준을 포함한다.

| 파일/산출물 | 내용 |
| --- | --- |
| `_project/PROJECT.md` | 실패 테스트 작성 원칙과 `pytest` 실행 기준 추가 |
| `_project/PROJECT-TODO.md` | Phase 0.3 테스트 원칙 정리 Task 완료 반영 |
