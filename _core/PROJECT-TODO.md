---
tags: [project, docs]
created: 2026-06-08
updated: 2026-06-21 (Phase 5.1~5.4, Phase 6.1~6.2 완료 처리)
---

# PROJECT-TODO.md

이 프로젝트의 진행 현황을 관리한다.
Stage - Phase - Task 단위로 체크박스를 관리한다.
Stage 및 Phase 구성은 `PROJECT-SPEC.md`의 진행 단계를 기준으로 작성한다.
Stage - Phase 수준의 제목은 키워드 중심의 개조식 표현으로 작성한다.

## 1. Stage 0 환경 구성 및 계획 수립

### 1.0. Phase 0.0 레거시 코드 분석

레거시 `src/` 코드 6개 스크립트와 common 모듈을 읽고, manual/module 두 패턴을 비교하며 task별 차이를 도출한다.

- [x] `_core/legacy/src/` 레거시 코드 구조 파악 (task 스크립트 6개 + common 모듈 6개)
- [x] common 모듈별 제공 요소 정리 (functions, modules, optimizers, dataloader, trainer)
- [x] manual 및 module 두 가지 구현 패턴 비교 분석
- [x] task별 차이 도출 (target 변환, output dim, loss, metric, gradient, 후처리)
- [x] [[docs/stage0/phase0.0_legacy-analysis|phase0.0_legacy-analysis.md]] 문서 작성

### 1.1. Phase 0.1 개발 환경 구성

MLP 실행용 `numpy_py311`과 CNN 실행용 `cupy_py311_cuda118`, `cupy_py311_cuda121` conda 환경을 생성하고 정상 동작을 확인한다.

- [x] `numpy_py311` CPU 기반 NumPy 실행 환경 확인
- [x] `cupy_py311_cuda118` CUDA 11.8 기준 CuPy 실행 환경 확인
- [x] `cupy_py311_cuda121` CUDA 12 계열 CuPy 실행 환경 확인
- [x] README.md에 3개 conda environment 생성 기준 작성
- [x] [[docs/stage0/phase0.1_conda-setup|phase0.1_conda-setup.md]] 문서 작성

### 1.2. Phase 0.2 구현 계획 수립

레거시 common 모듈을 src 파일로 1:1 매핑하고, 패키지 구조와 Stage 1-6 구현 순서를 확정한다. 후속 프레임워크(PyTorch, TensorFlow, JAX)와 공유할 공통 인터페이스 규약을 문서화한다.

- [x] 레거시 common 모듈에서 `src` 파일로 1:1 매핑 확정
- [x] `src` 패키지 구조 확정 (`data/`, `models/`, `core/`, `utils/`)
- [x] 각 파일의 책임 범위 확정
- [x] Stage 1-6 구현 순서 및 Phase 단위 분할 확정
- [x] [[docs/stage0/phase0.2_implementation-plan|phase0.2_implementation-plan.md]] 문서 작성
- [x] [[docs/stage0/phase0.2_framework-interface|phase0.2_framework-interface.md]] 후속 프레임워크 공통 인터페이스 규약 문서 작성

### 1.3. Phase 0.3 테스트 계획 수립

tests/ 폴더 구조와 파일별 공개 인터페이스 규약을 정하고, synthetic array 기반 TDD 원칙과 pytest 실행 방법을 확정한다.

- [x] `tests` 폴더 구조 확정 (stage 단위, `__init__.py` 없음)
- [x] 파일별 공개 인터페이스 규약 확정 (진입점, 입력, 출력)
- [x] TDD 원칙 확정 (synthetic array 우선, tolerance 비교 등)
- [x] `pytest` 실행 명령 정리 (stage 단위, 단일 파일, 전체)
- [x] [[docs/stage0/phase0.3_test-plan|phase0.3_test-plan.md]] 문서 작성

## 2. Stage 1 공통 유틸리티

### 2.1. Phase 1.1 배치 및 난수 유틸리티

mini-batch 인덱스 생성과 shuffle을 담당하는 `batching.py`와 난수 시드를 고정하는 `random.py`를 구현한다.

- [x] `src/utils/batching.py` 구현
- [x] `tests/stage1/test_batching.py` 작성
- [x] `src/utils/random.py` 구현
- [x] `tests/stage1/test_random.py` 작성
- [x] [[docs/stage1/phase1.1_batching-and-random|phase1.1_batching-and-random.md]] 문서 작성

### 2.2. Phase 1.2 파일 입출력 유틸리티

파일 저장·로딩 보조 함수(`io.py`)와 모델 파라미터를 `.npz` 파일로 저장·복원하는 `checkpoints.py`를 구현한다.

- [x] `src/utils/io.py` 구현
- [x] `tests/stage1/test_io.py` 작성
- [x] `src/utils/checkpoints.py` 구현 (이동 예정: `src/core/checkpoints.py`)
- [x] `tests/stage1/test_checkpoints.py` 작성 (이동 예정: `tests/stage4/test_checkpoints.py`)
- [x] [[docs/stage1/phase1.2_io-and-checkpoints|phase1.2_io-and-checkpoints.md]] 문서 작성

### 2.3. Phase 1.3 시각화 유틸리티

학습 로그 loss/metric 곡선을 PNG 파일로 저장하는 `training_plots.py` helper를 구현한다.

- [x] `src/utils/training_plots.py` 구현
- [x] `tests/stage1/test_training_plots.py` 작성
- [x] [[docs/stage1/phase1.3_training-plots|phase1.3_training-plots.md]] 문서 작성

### 2.4. Phase 1.4 실습 노트북 작성

Stage 1에서 구현한 유틸리티(batching, random, io, checkpoints, training_plots)를 실습하는 교육용 노트북을 작성한다.

- [x] `notebooks/stage1/stage1-1_utils.ipynb` 작성

## 3. Stage 2 MNIST 데이터 로더 구현

### 3.1. Phase 2.1 MNIST 데이터 로딩

MNIST `.gz` 파일을 파싱해 train/test로 분할하고 원본 배열을 반환하는 `load_mnist` 함수를 구현한다.

- [x] `src/data/mnist.py` 구현
- [x] `tests/stage2/test_mnist.py` 작성
- [x] [[docs/stage2/phase2.1_mnist|phase2.1_mnist.md]] 문서 작성

### 3.2. Phase 2.2 Dataset 구현

`MnistDataset` 클래스를 구현하고, task에 따른 target 변환과 `__getitem__` 인터페이스를 갖추어 DataLoader와 연동 가능하게 한다.

- [x] `src/data/mnist.py` 구현 (`MnistDataset` 추가)
- [x] `tests/stage2/test_dataset.py` 작성
- [x] [[docs/stage2/phase2.2_dataset|phase2.2_dataset.md]] 문서 작성

### 3.3. Phase 2.3 DataLoader 구현

배치 생성, shuffle, 반복 순회를 지원하는 `DataLoader`를 구현하고, `MnistDataset`과의 통합 동작을 검증한다.

- [x] `src/data/dataloader.py` 구현
- [x] `tests/stage2/test_dataloader.py` 작성
- [x] [[docs/stage2/phase2.3_dataloader|phase2.3_dataloader.md]] 문서 작성

### 3.4. Phase 2.4 실습 노트북 작성

MNIST 데이터 로딩, Dataset, DataLoader를 실습하는 교육용 노트북을 작성한다.

- [x] `notebooks/stage2/stage2-1_mnist-loading.ipynb` 작성
- [x] `notebooks/stage2/stage2-2_dataset-and-dataloader.ipynb` 작성

## 4. Stage 3 nn 모듈 구현

### 4.1. Phase 3.1 activation 함수 구현

`sigmoid`, `softmax`, `identity`, `relu` 활성화 함수를 구현한다.

- [x] `src/nn/activations.py` 구현
- [x] `tests/stage3/test_activations.py` 작성
- [x] [[docs/stage3/phase3.1_activations|phase3.1_activations.md]] 문서 작성

### 4.2. Phase 3.2 loss 함수 구현

`cross_entropy`, `binary_cross_entropy`, `mse` 손실 함수와 gradient 함수를 구현한다.

- [x] `src/nn/losses.py` 구현
- [x] `tests/stage3/test_losses.py` 작성
- [x] [[docs/stage3/phase3.2_losses|phase3.2_losses.md]] 문서 작성

### 4.3. Phase 3.3 metric 함수 구현

`accuracy`, `binary_accuracy`, `r2_score` 평가 지표를 구현한다.

- [x] `src/nn/metrics.py` 구현
- [x] `tests/stage3/test_metrics.py` 작성
- [x] [[docs/stage3/phase3.3_metrics|phase3.3_metrics.md]] 문서 작성

### 4.4. Phase 3.4 MLP 레이어 구현

`Module` 기반 `Linear`, `Sigmoid`, `ReLU`, `Sequential` 레이어(training/train/eval 포함)를 구현한다.

- [x] `src/nn/layers.py` 구현 (`Module` + `Linear`, `Sigmoid`, `ReLU`, `Sequential`, training/train/eval 포함)
- [x] `tests/stage3/test_layers.py` 작성
- [x] [[docs/stage3/phase3.4_mlp-layers|phase3.4_mlp-layers.md]] 문서 작성

### 4.5. Phase 3.5 CNN 레이어 구현

`im2col`/`col2im` 기반 `Conv2d`, `MaxPool2d`, `Flatten`, `Dropout`을 구현한다.

- [x] `src/nn/conv.py` 구현 (`im2col`/`col2im` + `Conv2d`, `MaxPool2d`, `Flatten`, `Dropout`)
- [x] `tests/stage3/test_conv.py` 작성 (test_cnn.py에 통합)
- [x] [[docs/stage3/phase3.5_cnn-layers|phase3.5_cnn-layers.md]] 문서 작성

### 4.6. Phase 3.6 실습 노트북 작성

nn 모듈(activation, loss, metric, layer, conv)을 실습하는 교육용 노트북을 작성한다.

- [ ] `notebooks/stage3/stage3-1_activations.ipynb` 작성
- [ ] `notebooks/stage3/stage3-2_losses-and-metrics.ipynb` 작성
- [ ] `notebooks/stage3/stage3-3_layers.ipynb` 작성
- [ ] `notebooks/stage3/stage3-4_conv-architecture.ipynb` 작성

## 5. Stage 4 모델 구현

### 5.1. Phase 4.1 MLP 모델 구현

`src/nn/` 모듈을 조립하여 NumPy 기반 MLP를 구현한다.

- [x] `src/models/mlp.py` 구현
- [x] `tests/stage4/test_mlp.py` 작성 (tests/stage3/test_mlp.py에 위치)
- [x] [[docs/stage4/phase4.1_mlp|phase4.1_mlp.md]] 문서 작성 (개념 섹션 상세화)

### 5.2. Phase 4.2 CNN 모델 구현

CuPy 기반 CNN 모델을 구현한다.

- [x] `src/models/cnn.py` 구현
- [x] `tests/stage4/test_cnn.py` 작성 (tests/stage3/test_cnn.py에 통합)
- [x] [[docs/stage4/phase4.2_cnn|phase4.2_cnn.md]] 문서 작성 (개념 섹션 상세화)

### 5.3. Phase 4.3 실습 노트북 작성

MLP와 CNN 모델을 실습하는 교육용 노트북을 작성한다.

- [ ] `notebooks/stage4/stage4-1_mlp.ipynb` 작성 (이동 예정: `notebooks/stage3/stage3-4_mlp.ipynb`)
- [ ] `notebooks/stage4/stage4-2_cnn-model.ipynb` 작성 (이동 예정: `notebooks/stage3/stage3-6_cnn-training.ipynb`)

## 6. Stage 5 실행 객체 구현

### 6.1. Phase 5.1 optimizer 구현

`SGD`와 `Adam` optimizer를 구현하고, 각 parameter에 대한 update 규칙과 모멘텀/적응 학습률 동작을 테스트로 검증한다.

- [x] `src/core/optimizers.py` 구현
- [x] `tests/stage5/test_optimizers.py` 작성
- [x] [[docs/stage5/phase5.1_optimizers|phase5.1_optimizers.md]] 문서 작성

### 6.2. Phase 5.2 Trainer 및 Evaluator 구현

`DataLoader`를 받아 epoch 단위 학습 루프를 실행하는 `Trainer.fit`과 배치 단위 평가 루프를 실행하는 `Evaluator.evaluate` 인터페이스를 구현한다.

- [x] `src/core/trainer.py` 구현
- [x] `tests/stage5/test_train.py` 작성
- [x] `src/core/evaluator.py` 구현
- [x] `tests/stage5/test_evaluate.py` 작성
- [x] [[docs/stage5/phase5.2_trainer-evaluator|phase5.2_trainer-evaluator.md]] 문서 작성

### 6.3. Phase 5.3 Predictor 및 Visualizer 구현

task에 따라 argmax/threshold/round_clip 후처리를 적용하는 `Predictor.predict`와 예측 이미지를 grid로 저장하는 `Visualizer`를 구현한다.

- [x] `src/core/predictor.py` 구현
- [x] `tests/stage5/test_predict.py` 작성
- [x] `src/core/visualizer.py` 구현
- [x] `tests/stage5/test_visualize.py` 작성
- [x] [[docs/stage5/phase5.3_predictor-visualizer|phase5.3_predictor-visualizer.md]] 문서 작성

### 6.4. Phase 5.4 Logger 구현

epoch별 loss/metric 로그를 기록하고 CSV 또는 dict 형태로 반환하는 `Logger`를 구현한다.

- [x] `src/core/logger.py` 구현
- [x] `tests/stage5/test_logger.py` 작성
- [x] [[docs/stage5/phase5.4_logger|phase5.4_logger.md]] 문서 작성

### 6.5. Phase 5.5 실습 노트북 작성

optimizer, Trainer, Evaluator, Predictor, Visualizer, Logger를 실습하는 교육용 노트북을 작성한다.

- [ ] `notebooks/stage5/stage5-1_optimizers.ipynb` 작성 (이동 예정: `notebooks/stage4/stage4-1_optimizers.ipynb`)
- [ ] `notebooks/stage5/stage5-2_trainer-and-evaluator.ipynb` 작성 (이동 예정: `notebooks/stage4/stage4-2_trainer-and-evaluator.ipynb`)
- [ ] `notebooks/stage5/stage5-3_predictor-and-visualizer.ipynb` 작성

## 7. Stage 6 클라이언트 코드 구현

### 7.1. Phase 6.1 학습 및 평가 스크립트 작성

학습을 실행하는 `scripts/train.py`와 checkpoint를 로드하여 평가를 실행하는 `scripts/evaluate.py`를 작성한다.

- [x] `scripts/train.py` 작성
- [x] `tests/stage6/test_train.py` 작성
- [x] `scripts/evaluate.py` 작성
- [x] `tests/stage6/test_evaluate.py` 작성
- [x] [[docs/stage6/phase6.1_train-evaluate|phase6.1_train-evaluate.md]] 문서 작성

### 7.2. Phase 6.2 예측 및 시각화 스크립트 작성

checkpoint를 로드하여 예측을 실행하는 `scripts/predict.py`와 결과 이미지를 저장하는 `scripts/visualize.py`를 작성한다.

- [x] `scripts/predict.py` 작성
- [x] `tests/stage6/test_predict.py` 작성
- [x] `scripts/visualize.py` 작성
- [x] `tests/stage6/test_visualize.py` 작성
- [x] [[docs/stage6/phase6.2_predict-visualize|phase6.2_predict-visualize.md]] 문서 작성

### 7.3. Phase 6.3 실험 배치 스크립트 작성

모든 task와 모델 조합을 순차 실행하는 `experiments/run_all.py` 배치 스크립트를 작성하고, 결과를 `outputs/{exp_name}/`에 저장하는 흐름을 검증한다.

- [x] `experiments/run_all.py` 작성
- [x] [[docs/stage6/phase6.3_run-all|phase6.3_run-all.md]] 문서 작성

### 7.4. Phase 6.4 실습 노트북 작성

스크립트(train/evaluate/predict/visualize)와 task별 MLP-CNN 비교 실험을 실습하는 교육용 노트북을 작성한다.

- [ ] `notebooks/stage6/stage6-1_cli-and-experiments.ipynb` 작성
- [ ] `notebooks/stage6/stage6-2_multiclass-experiment.ipynb` 작성
- [ ] `notebooks/stage6/stage6-3_binary-experiment.ipynb` 작성
- [ ] `notebooks/stage6/stage6-4_regression-experiment.ipynb` 작성
- [x] [[docs/stage6/phase6.4_experiments-and-results|phase6.4_experiments-and-results.md]] 문서 작성

