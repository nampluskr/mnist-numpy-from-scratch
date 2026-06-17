---
tags: [project, docs]
created: 2026-06-08
updated: 2026-06-18
---

# PROJECT-TODO.md

이 프로젝트의 진행 현황을 관리한다.
Stage - Phase - Task 단위로 체크박스를 관리한다.
Stage 및 Phase 구성은 `PROJECT-SPEC.md`의 진행 단계를 기준으로 작성한다.

## 1. Stage 0 레거시 코드 분석 및 계획 수립

### 1.1. Phase 0.1 레거시 코드 분석

코드 구조, 패턴, task별 차이를 분석하기 위한 작업은 다음과 같다.

- [x] `_core/legacy/src/` 레거시 코드 구조 파악 (task 스크립트 6개 + common 모듈 6개)
- [x] common 모듈별 제공 요소 정리 (functions, modules, optimizers, dataloader, trainer)
- [x] manual 및 module 두 가지 구현 패턴 비교 분석
- [x] task별 차이 도출 (target 변환, output dim, loss, metric, gradient, 후처리)
- [x] [[docs/stage0/phase0.1_legacy-analysis|phase0.1_legacy-analysis.md]] 문서 작성

### 1.2. Phase 0.2 구현 계획 수립

레거시-src 매핑, 패키지 구조, 구현 순서를 확정하기 위한 작업은 다음과 같다.

- [x] 레거시 common 모듈에서 `src` 파일로 1:1 매핑 확정
- [x] `src` 패키지 구조 확정 (`data/`, `models/`, `core/`, `utils/`)
- [x] 각 파일의 책임 범위 확정
- [x] Stage 1-7 구현 순서 및 Phase 단위 분할 확정
- [x] [[docs/stage0/phase0.2_implementation-plan|phase0.2_implementation-plan.md]] 문서 작성

### 1.3. Phase 0.3 테스트 계획 수립

tests 구조, 인터페이스 규약, TDD 원칙을 수립하기 위한 작업은 다음과 같다.

- [x] `tests` 폴더 구조 확정 (stage 단위, `__init__.py` 없음)
- [x] 파일별 공개 인터페이스 규약 확정 (진입점, 입력, 출력)
- [x] TDD 원칙 확정 (synthetic array 우선, tolerance 비교 등)
- [x] `pytest` 실행 명령 정리 (stage 단위, 단일 파일, 전체)
- [x] [[docs/stage0/phase0.3_test-plan|phase0.3_test-plan.md]] 문서 작성

## 2. Stage 1 config 및 task 규약

### 2.1. Phase 1.1 config 구성

기본 경로, 실행 기본값, hyperparameter를 config로 구성하기 위한 작업은 다음과 같다.

- [x] `requirements.txt`
- [x] `src/config.py`
- [x] `tests/stage1/test_config.py`
- [x] [[docs/stage1/phase1.1_config|phase1.1_config.md]] 문서 작성

### 2.2. Phase 1.2 과제 규약 정의

target 변환, loss 함수, metric 계산 규약을 정의하기 위한 작업은 다음과 같다.

- [x] `src/task.py`
- [x] `tests/stage1/test_task.py`
- [x] [[docs/stage1/phase1.2_task|phase1.2_task.md]] 문서 작성

### 2.3. Phase 1.3 utility 구현

batch 처리, random seed, 파일 input/output utility를 구현하기 위한 작업은 다음과 같다.

- [x] `src/utils/batching.py`
- [x] `tests/stage1/test_batching.py`
- [x] `src/utils/random.py`
- [x] `tests/stage1/test_random.py`
- [x] `src/utils/io.py`
- [x] `tests/stage1/test_io.py`
- [x] [[docs/stage1/phase1.3_utils|phase1.3_utils.md]] 문서 작성

## 3. Stage 2 MNIST DataLoader

### 3.1. Phase 2.1 MNIST raw data loading

gz parsing, train/test split, normalization을 포함한 MNIST raw data loading 작업은 다음과 같다.

- [x] `src/data/mnist.py`
- [x] `tests/stage2/test_mnist.py`
- [x] [[docs/stage2/phase2.1_mnist|phase2.1_mnist.md]] 문서 작성

### 3.2. Phase 2.2 Dataset 클래스 구현

MnistDataset, task 변환, __getitem__을 구현하기 위한 작업은 다음과 같다.

- [x] `src/data/mnist.py` (`MnistDataset` 추가)
- [x] `tests/stage2/test_dataset.py`
- [x] [[docs/stage2/phase2.2_dataset|phase2.2_dataset.md]] 문서 작성

### 3.3. Phase 2.3 DataLoader 구현

batch 생성, shuffle, iteration을 지원하는 DataLoader 구현 작업은 다음과 같다.

- [x] `src/data/dataloader.py`
- [x] `tests/stage2/test_dataloader.py`
- [x] [[docs/stage2/phase2.3_dataloader|phase2.3_dataloader.md]] 문서 작성

## 4. Stage 3 NumPy nn 모듈 및 MLP

### 4.1. Phase 3.1 activation 구현

sigmoid, softmax, identity, relu activation을 구현하기 위한 작업은 다음과 같다.

- [x] `src/nn/activations.py`
- [x] `tests/stage3/test_activations.py`
- [x] [[docs/stage3/phase3.1_activations|phase3.1_activations.md]] 문서 작성

### 4.2. Phase 3.2 layer module 구현

Linear, Sigmoid, ReLU, Sequential layer module을 구현하기 위한 작업은 다음과 같다.

- [x] `src/nn/layers.py`
- [x] `tests/stage3/test_layers.py`
- [x] [[docs/stage3/phase3.2_layers|phase3.2_layers.md]] 문서 작성

### 4.3. Phase 3.3 loss 및 metric 구현

cross_entropy, binary_cross_entropy, mse, accuracy 계열 loss 및 metric을 구현하기 위한 작업은 다음과 같다.

- [x] `src/nn/losses.py`
- [x] `tests/stage3/test_losses.py`
- [x] [[docs/stage3/phase3.3_losses|phase3.3_losses.md]] 문서 작성

### 4.4. Phase 3.4 MLP model 구현

src/nn module 조립, forward, backward, parameter 갱신을 포함한 MLP model 구현 작업은 다음과 같다.

- [x] `src/models/mlp.py`
- [x] `tests/stage3/test_mlp.py`
- [x] [[docs/stage3/phase3.4_mlp|phase3.4_mlp.md]] 문서 작성

## 5. Stage 4 실행 객체

### 5.1. Phase 4.1 optimizer 구현

SGD, Adam, parameter update 규칙을 포함한 optimizer 구현 작업은 다음과 같다.

- [x] `src/core/optimizers.py`
- [x] `tests/stage4/test_optimizers.py`
- [x] [[docs/stage4/phase4.1_optimizers|phase4.1_optimizers.md]] 문서 작성

### 5.2. Phase 4.2 checkpoint 구현

parameter 저장, loading, 경로 관리를 포함한 checkpoint 구현 작업은 다음과 같다.

- [x] `src/core/checkpoints.py`
- [x] `tests/stage4/test_checkpoints.py`
- [x] [[docs/stage4/phase4.2_checkpoints|phase4.2_checkpoints.md]] 문서 작성

### 5.3. Phase 4.3 Trainer 구현

training loop, DataLoader 수신, fit interface를 구현하기 위한 작업은 다음과 같다.

- [x] `src/core/trainer.py`
- [x] `tests/stage4/test_trainer.py`
- [x] [[docs/stage4/phase4.3_trainer|phase4.3_trainer.md]] 문서 작성

### 5.4. Phase 4.4 Evaluator 구현

evaluation loop, DataLoader 수신, evaluate interface를 구현하기 위한 작업은 다음과 같다.

- [x] `src/core/evaluator.py`
- [x] `tests/stage4/test_evaluator.py`
- [x] [[docs/stage4/phase4.4_evaluator|phase4.4_evaluator.md]] 문서 작성

### 5.5. Phase 4.5 Predictor 구현

prediction 실행, task별 post-processing, predict interface를 구현하기 위한 작업은 다음과 같다.

- [x] `src/core/predictor.py`
- [x] `tests/stage4/test_predictor.py`
- [x] [[docs/stage4/phase4.5_predictor|phase4.5_predictor.md]] 문서 작성

### 5.6. Phase 4.6 Experiment 구현

실행 객체 조립, dependency injection, 최상위 진입점을 구현하기 위한 작업은 다음과 같다.

- [x] `src/core/experiment.py`
- [x] `tests/stage4/test_experiment.py`
- [x] [[docs/stage4/phase4.6_experiment|phase4.6_experiment.md]] 문서 작성

### 5.7. Phase 4.7 Visualizer 구현

training log, prediction 결과, visualization 저장을 구현하기 위한 작업은 다음과 같다.

- [x] `src/core/visualizer.py`
- [x] `tests/stage4/test_visualizer.py`
- [x] [[docs/stage4/phase4.7_visualizer|phase4.7_visualizer.md]] 문서 작성

## 6. Stage 5 클라이언트 코드

### 6.1. Phase 5.1 training CLI 구현

argument parsing, experiment 조립, trainer 호출을 포함한 training CLI 구현 작업은 다음과 같다.

- [x] `scripts/train.py`
- [x] `tests/stage5/test_train.py`
- [x] [[docs/stage5/phase5.1_train|phase5.1_train.md]] 문서 작성

### 6.2. Phase 5.2 evaluation CLI 구현

argument parsing, experiment 조립, evaluator 호출을 포함한 evaluation CLI 구현 작업은 다음과 같다.

- [x] `scripts/evaluate.py`
- [x] `tests/stage5/test_evaluate.py`
- [x] [[docs/stage5/phase5.2_evaluate|phase5.2_evaluate.md]] 문서 작성

### 6.3. Phase 5.3 prediction CLI 구현

argument parsing, experiment 조립, predictor 호출을 포함한 prediction CLI 구현 작업은 다음과 같다.

- [x] `scripts/predict.py`
- [x] `tests/stage5/test_predict.py`
- [x] [[docs/stage5/phase5.3_predict|phase5.3_predict.md]] 문서 작성

### 6.4. Phase 5.4 visualization CLI 구현

argument parsing, experiment 조립, visualizer 호출을 포함한 visualization CLI 구현 작업은 다음과 같다.

- [x] `scripts/visualize.py`
- [x] `tests/stage5/test_visualize.py`
- [x] [[docs/stage5/phase5.4_visualize|phase5.4_visualize.md]] 문서 작성

## 7. Stage 6 CuPy CNN

### 7.1. Phase 6.0 CuPy environment 구성

conda numpy_env, cupy-cuda118 설치, environment 검증을 위한 작업은 다음과 같다.

- [x] `requirements.txt`에 `cupy-cuda11x` 추가
- [x] `conda run -n numpy_env pip install cupy-cuda11x` 설치 확인
- [x] `python -c "import cupy; print(cupy.__version__)"` 환경 검증
- [x] [[docs/stage6/phase6.0_cupy-setup|phase6.0_cupy-setup.md]] 문서 작성

### 7.2. Phase 6.1 CNN model 구현

CuPy 기반 forward, backward, parameter 갱신을 포함한 CNN model 구현 작업은 다음과 같다.

- [x] `src/nn/layers.py` (`Module` training/train/eval 추가)
- [x] `src/nn/conv.py` (`im2col`/`col2im` + `Conv2d`, `MaxPool2d`, `Flatten`, `Dropout`)
- [x] `src/models/cnn.py`
- [x] `tests/stage6/test_cnn.py`
- [x] [[docs/stage6/phase6.1_cnn|phase6.1_cnn.md]] 문서 작성

### 7.3. Phase 6.2 CNN-core integration 검증

core interface 호환성과 integration test를 검증하기 위한 CNN-core integration 작업은 다음과 같다.

- [x] `src/core/experiment.py` (`config["model"]` CNN 분기 추가)
- [x] `tests/stage6/test_experiment.py` (CNN 통합 케이스 추가)
- [x] [[docs/stage6/phase6.2_cnn-integration|phase6.2_cnn-integration.md]] 문서 작성

## 8. Stage 7 documentation 및 verification

### 8.1. Phase 7.1 CLI 확장

scripts --model flag 추가와 stage5 test update를 위한 CLI 확장 작업은 다음과 같다.

- [x] `scripts/train.py` - `--model` 플래그 추가 (choices: mlp, cnn, default: mlp)
- [x] `scripts/evaluate.py` - `--model` 플래그 추가
- [x] `scripts/predict.py` - `--model` 플래그 추가
- [x] `scripts/visualize.py` - `--model` 플래그 추가
- [x] `tests/stage5/test_train.py` - `--model` 테스트 케이스 추가
- [x] `tests/stage5/test_evaluate.py` - `--model` 테스트 케이스 추가
- [x] `tests/stage5/test_predict.py` - `--model` 테스트 케이스 추가
- [x] `tests/stage5/test_visualize.py` - `--model` 테스트 케이스 추가
- [x] [[docs/stage7/phase7.1_cli-extension|phase7.1_cli-extension.md]] 문서 작성

### 8.2. Phase 7.2 experiment 실행 및 result 수집

6종 experiment 실행, outputs 저장, results 문서 작성을 위한 작업은 다음과 같다.

- [x] `outputs/multiclass/mlp/` (`training_log.png`, `predictions.png`, `model.npz`)
- [ ] `outputs/multiclass/cnn/` (`training_log.png`, `predictions.png`, `model.npz`)
- [x] `outputs/binary/mlp/` (`training_log.png`, `predictions.png`, `model.npz`)
- [ ] `outputs/binary/cnn/` (`training_log.png`, `predictions.png`, `model.npz`)
- [x] `outputs/regression/mlp/` (`training_log.png`, `predictions.png`, `model.npz`)
- [ ] `outputs/regression/cnn/` (`training_log.png`, `predictions.png`, `model.npz`)
- [ ] [[docs/stage7/phase7.2_results|phase7.2_results.md]] 문서 작성

### 8.3. Phase 7.3 Multiclass tutorial

Multiclass MLP 및 CNN tutorial 작성을 위한 작업은 다음과 같다.

- [x] [[docs/stage7/phase7.3_tutorial-mlp|phase7.3_tutorial-mlp.md]] 문서 작성
- [ ] [[docs/stage7/phase7.3_tutorial-cnn|phase7.3_tutorial-cnn.md]] 문서 작성

### 8.4. Phase 7.4 Binary tutorial

Binary MLP 및 CNN tutorial 작성을 위한 작업은 다음과 같다.

- [x] [[docs/stage7/phase7.4_tutorial-mlp|phase7.4_tutorial-mlp.md]] 문서 작성
- [ ] [[docs/stage7/phase7.4_tutorial-cnn|phase7.4_tutorial-cnn.md]] 문서 작성

### 8.5. Phase 7.5 Regression tutorial

Regression MLP 및 CNN tutorial 작성을 위한 작업은 다음과 같다.

- [x] [[docs/stage7/phase7.5_tutorial-mlp|phase7.5_tutorial-mlp.md]] 문서 작성
- [ ] [[docs/stage7/phase7.5_tutorial-cnn|phase7.5_tutorial-cnn.md]] 문서 작성

### 8.6. Phase 7.6 framework 연계 준비

interface 규약 검토와 PyTorch migration checklist 작성을 위한 framework 연계 작업은 다음과 같다.

- [x] [[docs/stage7/phase7.6_framework-checklist|phase7.6_framework-checklist.md]] 문서 작성
