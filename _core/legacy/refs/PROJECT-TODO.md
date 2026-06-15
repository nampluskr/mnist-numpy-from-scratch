# Deep Learning from Scratch with Numpy/Cupy 작업 목록

프로젝트 진행 현황 대시보드이다. Stage - Phase - Task 단위로 체크박스를 관리하며, Stage 0은 설계 재검토, Stage 1부터는 각 파일마다 코드 작성 Task와 테스트 작성 Task를 분리하여 관리한다.

> 생성일시: 260608-133040
> 수정일시: 260608-142159
> 주제: Machine Learning

## Stage 0 프로젝트 설계 재검토

### Phase 0.1 기존 계획 및 레거시 입력 재검토

이 Phase는 기존 세션에서 생성된 계획과 레거시 입력을 실제 구현 가능한 단위로 재정렬하기 위한 준비 단계이다.

- [x] `_project/sessions/260608-133851_session-handoff.md` 기준의 기존 Stage - Phase - Task 구성을 재검토한다.
- [x] `legacy/src/`의 MNIST MLP 레거시 코드 3개 파일을 구현 입력으로 확인한다.
- [x] 레거시 코드의 데이터 처리, 모델, 학습, 평가 흐름을 파일 단위 구현 계획에 매핑한다.

### Phase 0.2 공통 `src` 및 `tests` 구조 확정

이 Phase는 NumPy/CuPy, PyTorch, TensorFlow, JAX 프로젝트가 공유할 최상위 구조를 확정하기 위한 단계이다.

- [x] 4개 프레임워크 프로젝트가 공유할 `src` 최상위 구조를 확정한다.
- [x] `training/` 대신 `core/`를 사용하는 실행 객체 배치 원칙을 확정한다.
- [x] 최상위 `nn/` 폴더를 제외하고 프레임워크별 차이를 `models/` 하위에서 흡수하는 원칙을 확정한다.
- [x] 테스트 대상별 파일이 생성되도록 `tests` 폴더 구조를 확정한다.

### Phase 0.3 파일 단위 코드·테스트 구현 순서 확정

이 Phase는 Stage 1 이후의 모든 작업을 코드 작성 Task와 테스트 작성 Task로 나누어 추적하기 위한 단계이다.

- [x] Stage 1 이후 Task 단위를 `소스 파일 코드 작성` / `테스트 파일 테스트 작성` 형식으로 재정의한다.
- [x] 공통 함수명과 입력·출력 규약을 파일별로 정의한다.
- [x] 구현 전 실패 테스트 작성 원칙과 `pytest` 실행 명령을 정리한다.

## Stage 1 기본 설정 및 과제 규약 구현

### Phase 1.1 `config.py` 코드 작성 및 `test_config.py` 테스트 작성

이 Phase는 프로젝트 기본 경로와 실행 기본값을 코드로 고정하기 위한 단계이다.

- [ ] `src/config.py` 코드 작성
- [ ] `tests/test_config.py` 테스트 작성

### Phase 1.2 `task.py` 코드 작성 및 `test_task.py` 테스트 작성

이 Phase는 `task = "multiclass"`, `task = "binary"`, `task = "regression"` 규약을 단일 파일에 구현하기 위한 단계이다.

- [ ] `src/task.py` 코드 작성
- [ ] `tests/test_task.py` 테스트 작성

### Phase 1.3 `utils` 코드 작성 및 대응 테스트 작성

이 Phase는 데이터 배치, 재현성, 파일 입출력 보조 기능을 테스트와 함께 구현하기 위한 단계이다.

- [ ] `src/utils/batching.py` 코드 작성
- [ ] `tests/utils/test_batching.py` 테스트 작성
- [ ] `src/utils/random.py` 코드 작성
- [ ] `tests/utils/test_random.py` 테스트 작성
- [ ] `src/utils/io.py` 코드 작성
- [ ] `tests/utils/test_io.py` 테스트 작성

## Stage 2 MNIST 데이터 로더 구현

### Phase 2.1 `data/mnist.py` 코드 작성 및 `test_mnist.py` 테스트 작성

이 Phase는 `/mnt/d/datasets/mnist`에 저장된 로컬 MNIST `*.gz` 파일 4개를 읽는 데이터 로더를 구현하기 위한 단계이다.

- [ ] `src/data/mnist.py` 코드 작성
- [ ] `tests/data/test_mnist.py` 테스트 작성

### Phase 2.2 데이터 로딩과 task 규약 통합 테스트

이 Phase는 MNIST 로더와 task 변환 규약이 함께 동작하는지 확인하기 위한 단계이다.

- [ ] `src/data/mnist.py`, `src/task.py`, `tests/data/test_mnist.py`, `tests/test_task.py` 통합 테스트 작성

## Stage 3 NumPy 기반 MLP 구현

### Phase 3.1 `models/mlp.py` 코드 작성 및 `test_mlp.py` 테스트 작성

이 Phase는 NumPy 기반 MLP 모델의 생성, forward, backward, parameter update 흐름을 구현하기 위한 단계이다.

- [ ] `src/models/mlp.py` 코드 작성
- [ ] `tests/models/test_mlp.py` 테스트 작성

### Phase 3.2 MLP 하위 구현과 대응 테스트 작성

이 Phase는 NumPy/CuPy 프로젝트에서만 필요한 from-scratch 모델 하위 구현을 `models/` 내부에 배치하고 테스트하기 위한 단계이다.

- [ ] `src/models/` 하위 구현 파일 코드 작성
- [ ] `tests/models/` 대응 테스트 작성

## Stage 4 실행 객체 구현

### Phase 4.1 `core/checkpoints.py` 코드 작성 및 `test_checkpoints.py` 테스트 작성

이 Phase는 모델 파라미터와 학습 산출물의 저장·로딩 흐름을 구현하기 위한 단계이다.

- [ ] `src/core/checkpoints.py` 코드 작성
- [ ] `tests/core/test_checkpoints.py` 테스트 작성

### Phase 4.2 `core/trainer.py` 코드 작성 및 `test_trainer.py` 테스트 작성

이 Phase는 학습 루프와 batch 처리 흐름을 구현하기 위한 단계이다.

- [ ] `src/core/trainer.py` 코드 작성
- [ ] `tests/core/test_trainer.py` 테스트 작성

### Phase 4.3 `core/evaluator.py` 코드 작성 및 `test_evaluator.py` 테스트 작성

이 Phase는 validation과 test 평가 흐름을 구현하기 위한 단계이다.

- [ ] `src/core/evaluator.py` 코드 작성
- [ ] `tests/core/test_evaluator.py` 테스트 작성

### Phase 4.4 `core/predictor.py` 코드 작성 및 `test_predictor.py` 테스트 작성

이 Phase는 단일 입력과 batch 입력에 대한 예측 흐름을 구현하기 위한 단계이다.

- [ ] `src/core/predictor.py` 코드 작성
- [ ] `tests/core/test_predictor.py` 테스트 작성

### Phase 4.5 `core/experiment.py` 코드 작성 및 `test_experiment.py` 테스트 작성

이 Phase는 config, data, task, model, core 실행 객체를 조립하는 상위 실행 객체를 구현하기 위한 단계이다.

- [ ] `src/core/experiment.py` 코드 작성
- [ ] `tests/core/test_experiment.py` 테스트 작성

### Phase 4.6 `core/visualizer.py` 코드 작성 및 `test_visualizer.py` 테스트 작성

이 Phase는 예측 결과, 학습 로그, 샘플 이미지 시각화 흐름을 구현하기 위한 단계이다.

- [ ] `src/core/visualizer.py` 코드 작성
- [ ] `tests/core/test_visualizer.py` 테스트 작성

## Stage 5 클라이언트 코드 구현

### Phase 5.1 `scripts/train.py` 코드 작성 및 `test_train.py` 테스트 작성

이 Phase는 학습 클라이언트 CLI를 구현하고 `core` 실행 객체 호출을 검증하기 위한 단계이다.

- [ ] `scripts/train.py` 코드 작성
- [ ] `tests/scripts/test_train.py` 테스트 작성

### Phase 5.2 `scripts/evaluate.py` 코드 작성 및 `test_evaluate.py` 테스트 작성

이 Phase는 평가 클라이언트 CLI를 구현하고 `core` 실행 객체 호출을 검증하기 위한 단계이다.

- [ ] `scripts/evaluate.py` 코드 작성
- [ ] `tests/scripts/test_evaluate.py` 테스트 작성

### Phase 5.3 `scripts/predict.py` 코드 작성 및 `test_predict.py` 테스트 작성

이 Phase는 예측 클라이언트 CLI를 구현하고 `core` 실행 객체 호출을 검증하기 위한 단계이다.

- [ ] `scripts/predict.py` 코드 작성
- [ ] `tests/scripts/test_predict.py` 테스트 작성

### Phase 5.4 `scripts/visualize.py` 코드 작성 및 `test_visualize.py` 테스트 작성

이 Phase는 시각화 클라이언트 CLI를 구현하고 `core` 실행 객체 호출을 검증하기 위한 단계이다.

- [ ] `scripts/visualize.py` 코드 작성
- [ ] `tests/scripts/test_visualize.py` 테스트 작성

## Stage 6 CuPy 기반 CNN 구현

### Phase 6.1 `models/cnn.py` 코드 작성 및 `test_cnn.py` 테스트 작성

이 Phase는 CuPy 기반 CNN 모델의 생성과 forward 흐름을 구현하기 위한 단계이다.

- [ ] `src/models/cnn.py` 코드 작성
- [ ] `tests/models/test_cnn.py` 테스트 작성

### Phase 6.2 CNN 실행 객체 연동 테스트

이 Phase는 CNN 모델이 기존 `core` 실행 객체와 같은 인터페이스로 연동되는지 확인하기 위한 단계이다.

- [ ] `src/models/cnn.py`, `src/core/experiment.py`, `tests/models/test_cnn.py`, `tests/core/test_experiment.py` 통합 테스트 작성

## Stage 7 튜토리얼 문서화 및 전체 검증

### Phase 7.1 Jupyter Book 챕터 작성

이 Phase는 구현된 코드와 테스트 결과를 튜토리얼 문서로 정리하기 위한 단계이다.

- [ ] `docs/contents/` 튜토리얼 챕터 작성

### Phase 7.2 실행 예제와 결과 정리

이 Phase는 각 task별 실행 명령, 로그, 평가 결과, 시각화 결과를 산출물로 정리하기 위한 단계이다.

- [ ] `outputs/` 기준 학습 로그, 평가 결과, 시각화 결과 정리

### Phase 7.3 후속 프레임워크 프로젝트 연계 기준 정리

이 Phase는 PyTorch, TensorFlow, JAX 프로젝트에서 재사용할 구조와 인터페이스 기준을 문서화하기 위한 단계이다.

- [ ] 후속 프로젝트 초기화 시 참조할 체크리스트 작성
