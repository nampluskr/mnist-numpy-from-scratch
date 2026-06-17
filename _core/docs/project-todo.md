---
tags: [project, docs]
created: 2026-06-08
updated: 2026-06-17
---

# project-todo.md

이 프로젝트의 진행 현황을 관리한다.
Stage - Phase - Task 단위로 체크박스를 관리한다.
Stage 및 Phase 구성은 project-spec.md 의 진행 단계를 기준으로 작성한다.

## Stage 0 레거시 코드 분석 및 계획 수립

### Phase 0.1 레거시 코드 분석: 코드 구조, 패턴, task별 차이

- [x] _core/legacy/src/ 레거시 코드 구조 파악 (task 스크립트 6개 + common 모듈 6개)
- [x] common 모듈별 제공 요소 정리 (functions, modules, optimizers, dataloader, trainer)
- [x] manual · module 두 가지 구현 패턴 비교 분석
- [x] task별 차이 도출 (target 변환, output dim, loss, metric, gradient, 후처리)
- [x] docs/stage0/phase0.1_legacy-analysis.md

### Phase 0.2 구현 계획 수립: 레거시-src 매핑, 패키지 구조, 구현 순서

- [x] 레거시 common 모듈 → src 파일 1:1 매핑 확정
- [x] src 패키지 구조 확정 (data/, models/, core/, utils/)
- [x] 각 파일의 책임 범위 확정
- [x] Stage 1~7 구현 순서 및 Phase 단위 분할 확정
- [x] docs/stage0/phase0.2_implementation-plan.md

### Phase 0.3 테스트 계획 수립: tests 구조, 인터페이스 규약, TDD 원칙

- [x] tests 폴더 구조 확정 (stage 단위, __init__.py 없음)
- [x] 파일별 공개 인터페이스 규약 확정 (진입점, 입력, 출력)
- [x] TDD 원칙 확정 (synthetic array 우선, tolerance 비교 등)
- [x] pytest 실행 명령 정리 (stage 단위, 단일 파일, 전체)
- [x] docs/stage0/phase0.3_test-plan.md

## Stage 1 기본 설정 · 과제 규약

### Phase 1.1 환경 설정 구성: 기본 경로, 실행 기본값, 하이퍼파라미터

- [x] requirements.txt
- [x] src/config.py
- [x] tests/stage1/test_config.py
- [x] docs/stage1/phase1.1_config.md

### Phase 1.2 과제 규약 정의: target 변환, loss 함수, metric 계산

- [x] src/task.py
- [x] tests/stage1/test_task.py
- [x] docs/stage1/phase1.2_task.md

### Phase 1.3 유틸리티 구현: 배치 처리, 난수 시드, 파일 I/O

- [x] src/utils/batching.py
- [x] tests/stage1/test_batching.py
- [x] src/utils/random.py
- [x] tests/stage1/test_random.py
- [x] src/utils/io.py
- [x] tests/stage1/test_io.py
- [x] docs/stage1/phase1.3_utils.md

## Stage 2 MNIST 데이터 로더

### Phase 2.1 MNIST 원시 데이터 로딩: gz 파싱, train/test split, 정규화

- [x] src/data/mnist.py
- [x] tests/stage2/test_mnist.py
- [x] docs/stage2/phase2.1_mnist.md

### Phase 2.2 Dataset 클래스 구현: MnistDataset, task 변환, __getitem__

- [x] src/data/mnist.py (MnistDataset 추가)
- [x] tests/stage2/test_dataset.py
- [x] docs/stage2/phase2.2_dataset.md

### Phase 2.3 DataLoader 구현: 배치 생성, shuffle, 반복 순회

- [x] src/data/dataloader.py
- [x] tests/stage2/test_dataloader.py
- [x] docs/stage2/phase2.3_dataloader.md

## Stage 3 NumPy MLP

### Phase 3.1 MLP 모델 구현: forward, backward, 파라미터 갱신

- [x] src/models/mlp.py
- [x] tests/stage3/test_mlp.py
- [x] docs/stage3/phase3.1_mlp.md

### Phase 3.2 신경망 구성요소 구현: layers, activations, losses

- [ ] src/models/layers.py
- [ ] tests/stage3/test_layers.py
- [ ] src/models/activations.py
- [ ] tests/stage3/test_activations.py
- [ ] src/models/losses.py
- [ ] tests/stage3/test_losses.py
- [ ] docs/stage3/phase3.2_nn.md

## Stage 4 실행 객체

### Phase 4.1 옵티마이저 구현: SGD, Adam, 파라미터 업데이트 규칙

- [ ] src/core/optimizers.py
- [ ] tests/stage4/test_optimizers.py
- [ ] docs/stage4/phase4.1_optimizers.md

### Phase 4.2 체크포인트 구현: 파라미터 저장, 로딩, 경로 관리

- [ ] src/core/checkpoints.py
- [ ] tests/stage4/test_checkpoints.py
- [ ] docs/stage4/phase4.2_checkpoints.md

### Phase 4.3 Trainer 구현: 학습 루프, DataLoader 수신, fit 인터페이스

- [ ] src/core/trainer.py
- [ ] tests/stage4/test_trainer.py
- [ ] docs/stage4/phase4.3_trainer.md

### Phase 4.4 Evaluator 구현: 평가 루프, DataLoader 수신, evaluate 인터페이스

- [ ] src/core/evaluator.py
- [ ] tests/stage4/test_evaluator.py
- [ ] docs/stage4/phase4.4_evaluator.md

### Phase 4.5 Predictor 구현: 예측 실행, task별 후처리, predict 인터페이스

- [ ] src/core/predictor.py
- [ ] tests/stage4/test_predictor.py
- [ ] docs/stage4/phase4.5_predictor.md

### Phase 4.6 Experiment 구현: 실행 객체 조립, 의존성 주입, 최상위 진입점

- [ ] src/core/experiment.py
- [ ] tests/stage4/test_experiment.py
- [ ] docs/stage4/phase4.6_experiment.md

### Phase 4.7 Visualizer 구현: 학습 로그, 예측 결과, 시각화 저장

- [ ] src/core/visualizer.py
- [ ] tests/stage4/test_visualizer.py
- [ ] docs/stage4/phase4.7_visualizer.md

## Stage 5 클라이언트 코드

### Phase 5.1 학습 CLI 구현: 인자 파싱, experiment 조립, trainer 호출

- [ ] scripts/train.py
- [ ] tests/stage5/test_train.py
- [ ] docs/stage5/phase5.1_train.md

### Phase 5.2 평가 CLI 구현: 인자 파싱, experiment 조립, evaluator 호출

- [ ] scripts/evaluate.py
- [ ] tests/stage5/test_evaluate.py
- [ ] docs/stage5/phase5.2_evaluate.md

### Phase 5.3 예측 CLI 구현: 인자 파싱, experiment 조립, predictor 호출

- [ ] scripts/predict.py
- [ ] tests/stage5/test_predict.py
- [ ] docs/stage5/phase5.3_predict.md

### Phase 5.4 시각화 CLI 구현: 인자 파싱, experiment 조립, visualizer 호출

- [ ] scripts/visualize.py
- [ ] tests/stage5/test_visualize.py
- [ ] docs/stage5/phase5.4_visualize.md

## Stage 6 CuPy CNN

### Phase 6.1 CNN 모델 구현: CuPy 기반, forward, backward, 파라미터 갱신

- [ ] src/models/cnn.py
- [ ] tests/stage6/test_cnn.py
- [ ] docs/stage6/phase6.1_cnn.md

### Phase 6.2 CNN-core 통합 검증: core 인터페이스 호환, 통합 테스트

- [ ] tests/stage6/test_experiment.py (CNN 통합 케이스 추가)
- [ ] docs/stage6/phase6.2_cnn-integration.md

## Stage 7 문서화 · 검증

### Phase 7.1 튜토리얼 문서화: Jupyter Book 챕터, 코드 설명, 실습 예제

- [ ] docs/stage7/ 챕터 작성

### Phase 7.2 실행 결과 정리: task별 로그, 평가 결과, 시각화 출력

- [ ] outputs/ 결과 정리
- [ ] docs/stage7/phase7.2_results.md

### Phase 7.3 프레임워크 연계 준비: 인터페이스 규약 검토, 후속 프로젝트 체크리스트

- [ ] docs/stage7/phase7.3_framework-checklist.md
