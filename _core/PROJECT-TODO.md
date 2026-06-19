---
tags: [project, docs]
created: 2026-06-08
updated: 2026-06-20
---

# PROJECT-TODO.md

이 프로젝트의 진행 현황을 관리한다.
Stage - Phase - Task 단위로 체크박스를 관리한다.
Stage 및 Phase 구성은 `PROJECT-SPEC.md`의 진행 단계를 기준으로 작성한다.
Stage - Phase 수준의 제목은 키워드 중심의 개조식 표현으로 작성한다.

## 1. Stage 0 계획 수립

### 1.1. Phase 0.1 레거시 코드 분석

레거시 src/ 코드 6개 스크립트와 common 모듈을 읽고, manual/module 두 패턴을 비교하며 task별 차이를 도출한다.

- [x] `_core/legacy/src/` 레거시 코드 구조 파악 (task 스크립트 6개 + common 모듈 6개)
- [x] common 모듈별 제공 요소 정리 (functions, modules, optimizers, dataloader, trainer)
- [x] manual 및 module 두 가지 구현 패턴 비교 분석
- [x] task별 차이 도출 (target 변환, output dim, loss, metric, gradient, 후처리)
- [x] [[docs/stage0/phase0.1_legacy-analysis|phase0.1_legacy-analysis.md]] 문서 작성

### 1.2. Phase 0.2 구현 계획 수립

레거시 common 모듈을 src 파일로 1:1 매핑하고, 패키지 구조와 Stage 1-7 구현 순서를 확정한다.

- [x] 레거시 common 모듈에서 `src` 파일로 1:1 매핑 확정
- [x] `src` 패키지 구조 확정 (`data/`, `models/`, `core/`, `utils/`)
- [x] 각 파일의 책임 범위 확정
- [x] Stage 1-7 구현 순서 및 Phase 단위 분할 확정
- [x] [[docs/stage0/phase0.2_implementation-plan|phase0.2_implementation-plan.md]] 문서 작성

### 1.3. Phase 0.3 테스트 계획 수립

tests/ 폴더 구조와 파일별 공개 인터페이스 규약을 정하고, synthetic array 기반 TDD 원칙과 pytest 실행 방법을 확정한다.

- [x] `tests` 폴더 구조 확정 (stage 단위, `__init__.py` 없음)
- [x] 파일별 공개 인터페이스 규약 확정 (진입점, 입력, 출력)
- [x] TDD 원칙 확정 (synthetic array 우선, tolerance 비교 등)
- [x] `pytest` 실행 명령 정리 (stage 단위, 단일 파일, 전체)
- [x] [[docs/stage0/phase0.3_test-plan|phase0.3_test-plan.md]] 문서 작성

## 2. Stage 1 기본 설정 및 과제 규약 구현

### 2.1. Phase 1.1 config 구성

데이터 경로, 학습률·배치 크기 등 hyperparameter 기본값을 `src/config.py` dict로 정의하고 테스트로 검증한다.

- [x] `requirements.txt`
- [x] `src/config.py`
- [x] `tests/stage1/test_config.py`
- [x] [[docs/stage1/phase1.1_config|phase1.1_config.md]] 문서 작성

### 2.2. Phase 1.2 과제 규약 정의

multiclass, binary, regression 세 가지 task에 대해 target 변환, loss 함수, metric 계산 방식을 `src/task.py`에 규약으로 정의하고, task별 변환 결과를 테스트로 검증한다.

- [x] `src/task.py`
- [x] `tests/stage1/test_task.py`
- [x] [[docs/stage1/phase1.2_task|phase1.2_task.md]] 문서 작성

### 2.3. Phase 1.3 utility 구현

배치 분할(`batching.py`), random seed 고정(`random.py`), 모델 저장/로드(`io.py`) 세 가지 공통 유틸리티를 구현한다.

- [x] `src/utils/batching.py`
- [x] `tests/stage1/test_batching.py`
- [x] `src/utils/random.py`
- [x] `tests/stage1/test_random.py`
- [x] `src/utils/io.py`
- [x] `tests/stage1/test_io.py`
- [x] [[docs/stage1/phase1.3_utils|phase1.3_utils.md]] 문서 작성

### 2.4. Phase 1.4 Stage 1 노트북 작성

Stage 1에서 구현한 config, task spec, utility를 실습하는 교육용 노트북을 작성한다.

- [x] `notebooks/stage1/stage1-1_config-and-task.ipynb` 작성

## 3. Stage 2 MNIST 데이터 로더 구현

### 3.1. Phase 2.1 MNIST raw data loading

MNIST .gz 파일을 파싱해 train/test로 분할하고 [0, 1] 정규화까지 수행하는 `load_mnist` 함수를 구현한다.

- [x] `src/data/mnist.py`
- [x] `tests/stage2/test_mnist.py`
- [x] [[docs/stage2/phase2.1_mnist|phase2.1_mnist.md]] 문서 작성

### 3.2. Phase 2.2 Dataset 클래스 구현

`MnistDataset` 클래스를 구현하고, task에 따른 target 변환과 `__getitem__` 인터페이스를 갖추어 DataLoader와 연동 가능하게 한다.

- [x] `src/data/mnist.py` (`MnistDataset` 추가)
- [x] `tests/stage2/test_dataset.py`
- [x] [[docs/stage2/phase2.2_dataset|phase2.2_dataset.md]] 문서 작성

### 3.3. Phase 2.3 DataLoader 구현

배치 생성, shuffle, 반복 순회를 지원하는 `DataLoader`를 구현하고, `MnistDataset`과의 통합 동작을 검증한다.

- [x] `src/data/dataloader.py`
- [x] `tests/stage2/test_dataloader.py`
- [x] [[docs/stage2/phase2.3_dataloader|phase2.3_dataloader.md]] 문서 작성

### 3.4. Phase 2.4 Stage 2 노트북 작성

MNIST 데이터 로딩, Dataset, DataLoader를 실습하는 교육용 노트북을 작성한다.

- [x] `notebooks/stage2/stage2-1_mnist-loading.ipynb` 작성
- [x] `notebooks/stage2/stage2-2_dataset-and-dataloader.ipynb` 작성

## 4. Stage 3 NumPy nn 모듈 및 MLP 구현

### 4.1. Phase 3.1 activation 구현

sigmoid, softmax, identity, relu 네 가지 활성화 함수를 순수 함수로 구현하고, 수치 안정성과 배치 입출력 shape를 검증한다.

- [x] `src/nn/activations.py`
- [x] `tests/stage3/test_activations.py`
- [x] [[docs/stage3/phase3.1_activations|phase3.1_activations.md]] 문서 작성

### 4.2. Phase 3.2 layer module 구현

`Linear`, `Sigmoid`, `ReLU`, `Sequential` 레이어 클래스를 `Module` 기반으로 구현하고, forward/backward 동작과 parameter 추적을 검증한다.

- [x] `src/nn/layers.py`
- [x] `tests/stage3/test_layers.py`
- [x] [[docs/stage3/phase3.2_layers|phase3.2_layers.md]] 문서 작성

### 4.3. Phase 3.3 loss 및 gradient 구현

`cross_entropy`, `binary_cross_entropy`, `mse` 손실 함수와 각 task의 출력 gradient 함수를 `src/nn/losses.py`에 구현하고, 손실 값과 gradient shape를 검증한다.

- [x] `src/nn/losses.py`
- [x] `tests/stage3/test_losses.py`
- [x] [[docs/stage3/phase3.3_losses|phase3.3_losses.md]] 문서 작성

### 4.4. Phase 3.4 metric 구현

`accuracy`, `binary_accuracy`, `r2_score` 세 가지 평가 지표를 `src/nn/metrics.py`에 구현하고, task별 올바른 계산 결과를 검증한다.

- [x] `src/nn/metrics.py`
- [x] `tests/stage3/test_metrics.py`
- [x] [[docs/stage3/phase3.4_metrics|phase3.4_metrics.md]] 문서 작성

### 4.5. Phase 3.5 MLP model 구현

`src/nn` 모듈을 `Sequential`로 조립하고, forward/backward pass와 parameter 갱신까지 동작하는 MLP 모델을 `src/models/mlp.py`에 구현하고, 단계별 동작을 테스트로 검증한다.

- [x] `src/models/mlp.py`
- [x] `tests/stage3/test_mlp.py`
- [x] [[docs/stage3/phase3.5_mlp|phase3.5_mlp.md]] 문서 작성

### 4.6. Phase 3.6 Stage 3 노트북 작성

신경망 기초 모듈과 MLP를 실습하는 교육용 노트북을 작성한다.

- [x] `notebooks/stage3/stage3-1_activations.ipynb` 작성
- [x] `notebooks/stage3/stage3-2_layers.ipynb` 작성
- [x] `notebooks/stage3/stage3-3_losses-and-metrics.ipynb` 작성
- [x] `notebooks/stage3/stage3-4_mlp.ipynb` 작성

## 5. Stage 4 CuPy 기반 CNN 구현

### 5.1. Phase 4.0 CuPy environment 구성

CPU용 `numpy_py311`, GPU용 `cupy_py311_cuda118`, `cupy_py311_cuda121` 세 conda 환경을 생성하고 CuPy import와 CUDA 버전 연동을 확인한다.

- [x] README.md에 3개 conda environment 생성 기준 작성
- [x] `numpy_py311` CPU 기반 NumPy 실행 환경 확인
- [x] `cupy_py311_cuda118` CUDA 11.8 기준 CuPy 실행 환경 확인
- [x] `cupy_py311_cuda121` CUDA 12 계열 CuPy 실행 환경 확인
- [x] [[docs/stage4/phase4.0_cupy-setup|phase4.0_cupy-setup.md]] 문서 작성

### 5.2. Phase 4.1 CNN model 구현

`im2col`/`col2im`을 기반으로 `Conv2d`, `MaxPool2d`, `Flatten`, `Dropout`을 `src/nn/conv.py`에 구현하고, `src/models/cnn.py`로 CuPy/NumPy 양용 CNN 모델을 완성한다.

- [x] `src/nn/layers.py` (`Module` training/train/eval 추가)
- [x] `src/nn/conv.py` (`im2col`/`col2im` + `Conv2d`, `MaxPool2d`, `Flatten`, `Dropout`)
- [x] `src/models/cnn.py`
- [x] `tests/stage4/test_cnn.py`
- [x] [[docs/stage4/phase4.1_cnn|phase4.1_cnn.md]] 문서 작성

### 5.3. Phase 4.2 CNN-core integration 검증 및 CLI 확장

`Experiment`에 config `"model"` 분기를 추가하여 MLP/CNN을 선택하고, 4개 CLI 스크립트에 `--model` 플래그(mlp/cnn, 기본값 mlp)를 추가한다. core 통합 테스트와 stage6 CLI 테스트에 MLP/CNN 케이스를 추가하여 전 계층 동작을 검증한다.

- [x] `src/core/experiment.py` (`config["model"]` CNN 분기 추가)
- [x] `tests/stage4/test_experiment.py` (CNN 통합 케이스 추가)
- [x] `scripts/train.py` (`--model` 플래그 추가, choices: mlp, cnn, default: mlp)
- [x] `scripts/evaluate.py` (`--model` 플래그 추가)
- [x] `scripts/predict.py` (`--model` 플래그 추가)
- [x] `scripts/visualize.py` (`--model` 플래그 추가)
- [x] `tests/stage6/test_train.py` (`--model` MLP/CNN 테스트 케이스 추가)
- [x] `tests/stage6/test_evaluate.py` (`--model` MLP/CNN 테스트 케이스 추가)
- [x] `tests/stage6/test_predict.py` (`--model` MLP/CNN 테스트 케이스 추가)
- [x] `tests/stage6/test_visualize.py` (`--model` MLP/CNN 테스트 케이스 추가)
- [x] [[docs/stage4/phase4.2_cnn-integration|phase4.2_cnn-integration.md]] 문서 작성

### 5.4. Phase 4.3 Stage 4 노트북 작성

CNN 모델 구조, CuPy 환경, MLP 대비 파라미터 비교를 실습하는 교육용 노트북을 작성한다.

- [x] `notebooks/stage4/stage4-1_cnn-architecture.ipynb` 작성
- [x] `notebooks/stage4/stage4-2_cnn-training.ipynb` 작성

## 6. Stage 5 실행 객체 구현

### 6.1. Phase 5.1 optimizer 구현

`SGD`와 `Adam` optimizer를 구현하고, 각 parameter에 대한 update 규칙과 모멘텀/적응 학습률 동작을 테스트로 검증한다.

- [x] `src/core/optimizers.py`
- [x] `tests/stage5/test_optimizers.py`
- [x] [[docs/stage5/phase5.1_optimizers|phase5.1_optimizers.md]] 문서 작성

### 6.2. Phase 5.2 checkpoint 구현

모델 parameter를 `.npz` 파일로 저장·로드하는 checkpoint 기능을 구현하고, CuPy array도 NumPy로 변환하여 저장·복원됨을 검증한다.

- [x] `src/core/checkpoints.py`
- [x] `tests/stage5/test_checkpoints.py`
- [x] [[docs/stage5/phase5.2_checkpoints|phase5.2_checkpoints.md]] 문서 작성

### 6.3. Phase 5.3 Trainer 구현

`DataLoader`를 받아 epoch 단위 training loop를 실행하는 `Trainer.fit` 인터페이스를 구현하고, loss/metric 로그 반환을 검증한다.

- [x] `src/core/trainer.py`
- [x] `tests/stage5/test_trainer.py`
- [x] [[docs/stage5/phase5.3_trainer|phase5.3_trainer.md]] 문서 작성

### 6.4. Phase 5.4 Evaluator 구현

`DataLoader`를 받아 배치 단위 evaluation loop를 실행하는 `Evaluator.evaluate` 인터페이스를 구현하고, loss/metric 집계 결과를 검증한다.

- [x] `src/core/evaluator.py`
- [x] `tests/stage5/test_evaluator.py`
- [x] [[docs/stage5/phase5.4_evaluator|phase5.4_evaluator.md]] 문서 작성

### 6.5. Phase 5.5 Predictor 구현

`Predictor.predict` 인터페이스를 구현하고, task에 따라 argmax/threshold/round_clip 세 가지 후처리를 적용하여 예측 결과를 반환함을 검증한다.

- [x] `src/core/predictor.py`
- [x] `tests/stage5/test_predictor.py`
- [x] [[docs/stage5/phase5.5_predictor|phase5.5_predictor.md]] 문서 작성

### 6.6. Phase 5.6 Experiment 구현

`Experiment.run`으로 model, optimizer, trainer, evaluator, predictor를 조립하고 config dict를 통해 의존성을 주입하는 최상위 진입점을 구현하고, synthetic MNIST 기반 통합 테스트로 검증한다.

- [x] `src/core/experiment.py`
- [x] `tests/stage5/test_experiment.py`
- [x] [[docs/stage5/phase5.6_experiment|phase5.6_experiment.md]] 문서 작성

### 6.7. Phase 5.7 Visualizer 및 training plot helper 구현

예측 이미지를 grid로 저장하는 `Visualizer`와 학습 loss/metric 곡선을 PNG로 저장하는 `training_plots.py` helper를 분리 구현하고, 출력 파일 생성 여부를 테스트로 검증한다.

- [x] `src/core/visualizer.py`
- [x] `src/utils/training_plots.py`
- [x] `tests/stage5/test_visualizer.py`
- [x] `tests/stage1/test_training_plots.py`
- [x] [[docs/stage5/phase5.7_visualizer|phase5.7_visualizer.md]] 문서 작성

### 6.8. Phase 5.8 Stage 5 노트북 작성

학습 프레임워크(Experiment, Trainer, Evaluator, Predictor, Checkpoint)를 실습하는 교육용 노트북을 작성한다.

- [x] `notebooks/stage5/stage5-1_optimizers.ipynb` 작성
- [x] `notebooks/stage5/stage5-2_trainer-and-evaluator.ipynb` 작성
- [x] `notebooks/stage5/stage5-3_experiment.ipynb` 작성

## 7. Stage 6 클라이언트 코드 구현

### 7.1. Phase 6.1 training CLI 구현

커맨드라인 인수를 파싱하여 `Experiment`를 조립하고 `Trainer.fit`을 호출하는 `scripts/train.py` CLI를 구현한다.

- [x] `scripts/train.py`
- [x] `tests/stage6/test_train.py`
- [x] [[docs/stage6/phase6.1_train|phase6.1_train.md]] 문서 작성

### 7.2. Phase 6.2 evaluation CLI 구현

커맨드라인 인수를 파싱하여 checkpoint를 로드하고 `Evaluator.evaluate`를 호출하는 `scripts/evaluate.py` CLI를 구현한다.

- [x] `scripts/evaluate.py`
- [x] `tests/stage6/test_evaluate.py`
- [x] [[docs/stage6/phase6.2_evaluate|phase6.2_evaluate.md]] 문서 작성

### 7.3. Phase 6.3 prediction CLI 구현

커맨드라인 인수를 파싱하여 checkpoint를 로드하고 `Predictor.predict`를 호출하는 `scripts/predict.py` CLI를 구현한다.

- [x] `scripts/predict.py`
- [x] `tests/stage6/test_predict.py`
- [x] [[docs/stage6/phase6.3_predict|phase6.3_predict.md]] 문서 작성

### 7.4. Phase 6.4 visualization CLI 구현

커맨드라인 인수를 파싱하여 `Visualizer`와 `training_plots` helper를 호출하고 결과 이미지를 `outputs/`에 저장하는 `scripts/visualize.py` CLI를 구현한다.

- [x] `scripts/visualize.py`
- [x] `tests/stage6/test_visualize.py`
- [x] [[docs/stage6/phase6.4_visualize|phase6.4_visualize.md]] 문서 작성

### 7.5. Phase 6.5 Stage 6 노트북 작성

CLI 스크립트(train/evaluate/predict/visualize)를 Python에서 직접 호출하는 교육용 노트북을 작성한다.

- [x] `notebooks/stage6/stage6-1_cli-scripts.ipynb` 작성

## 8. Stage 7 documentation 및 verification

### 8.1. Phase 7.1 experiment 실행 및 result 수집

multiclass, binary, regression 각 task에 대해 MLP와 CNN을 학습하여 `training_log.png`, `predictions.png`, `model.npz`를 `outputs/`에 저장하고 결과를 문서화한다.

- [x] `outputs/multiclass/mlp/` (`training_log.png`, `predictions.png`, `model.npz`)
- [x] `outputs/multiclass/cnn/` (`training_log.png`, `predictions.png`, `model.npz`)
- [x] `outputs/binary/mlp/` (`training_log.png`, `predictions.png`, `model.npz`)
- [x] `outputs/binary/cnn/` (`training_log.png`, `predictions.png`, `model.npz`)
- [x] `outputs/regression/mlp/` (`training_log.png`, `predictions.png`, `model.npz`)
- [x] `outputs/regression/cnn/` (`training_log.png`, `predictions.png`, `model.npz`)
- [x] [[docs/stage7/phase7.1_results|phase7.1_results.md]] 문서 작성

### 8.2. Phase 7.2 Multiclass tutorial

10클래스 분류 task에서 MLP와 CNN의 학습 절차, CLI 실행 명령, 평가 결과를 각각 tutorial 문서로 작성한다.

- [x] [[docs/stage7/phase7.2_tutorial-mlp|phase7.2_tutorial-mlp.md]] 문서 작성
- [x] [[docs/stage7/phase7.2_tutorial-cnn|phase7.2_tutorial-cnn.md]] 문서 작성

### 8.3. Phase 7.3 Binary tutorial

이진 분류 task에서 MLP와 CNN의 학습 절차, sigmoid threshold 후처리, CLI 실행 명령, 평가 결과를 각각 tutorial 문서로 작성한다.

- [x] [[docs/stage7/phase7.3_tutorial-mlp|phase7.3_tutorial-mlp.md]] 문서 작성
- [x] [[docs/stage7/phase7.3_tutorial-cnn|phase7.3_tutorial-cnn.md]] 문서 작성

### 8.4. Phase 7.4 Regression tutorial

회귀 task에서 MLP와 CNN의 학습 절차, R² 평가 지표, round_clip 후처리, CLI 실행 명령, 평가 결과를 각각 tutorial 문서로 작성한다.

- [x] [[docs/stage7/phase7.4_tutorial-mlp|phase7.4_tutorial-mlp.md]] 문서 작성
- [x] [[docs/stage7/phase7.4_tutorial-cnn|phase7.4_tutorial-cnn.md]] 문서 작성

### 8.5. Phase 7.5 framework 연계 준비

본 프로젝트의 모듈·인터페이스 규약을 검토하고, PyTorch 마이그레이션 시 대응 항목을 checklist 문서로 정리하여 후속 프레임워크 프로젝트 시작 기준을 마련하고 문서로 확정한다.

- [x] [[docs/stage7/phase7.5_framework-checklist|phase7.5_framework-checklist.md]] 문서 작성

### 8.6. Phase 7.6 Stage 7 노트북 작성

3종 태스크 전체 실험(MLP + CNN 비교)을 실습하는 교육용 노트북 3개를 작성한다.

- [x] `notebooks/stage7/stage7-1_multiclass-experiment.ipynb` 작성
- [x] `notebooks/stage7/stage7-2_binary-experiment.ipynb` 작성
- [x] `notebooks/stage7/stage7-3_regression-experiment.ipynb` 작성
