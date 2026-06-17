---
tags: [project, docs]
created: 2026-06-08
updated: 2026-06-17
---

# project-todo.md

이 프로젝트의 진행 현황을 관리한다.
Stage - Phase - Task 단위로 체크박스를 관리한다.
Stage 및 Phase 구성은 project-spec.md 의 진행 단계를 기준으로 작성한다.

## Stage 0 설계 재검토

### Phase 0.1 레거시 재검토 — 레거시 분석, 구현 매핑

- [x] _core/legacy/ 기준 Stage-Phase-Task 구성 검토
- [x] _core/legacy/src/ 레거시 코드 3개 파일 확인
- [x] 레거시 코드 흐름 → 파일 단위 구현 계획 매핑
- [x] docs/stage0/phase0.1_legacy-review.md

### Phase 0.2 src · tests 구조 확정 — 공통 구조, 폴더 원칙

- [x] 4개 프레임워크 공유 src 최상위 구조 확정
- [x] core/ 실행 객체 배치 원칙 확정
- [x] models/ 하위 구현 흡수 원칙 확정
- [x] tests 폴더 구조 확정
- [x] docs/stage0/phase0.2_structure.md

### Phase 0.3 구현 순서 확정 — Task 단위, 규약, TDD 원칙

- [x] Stage 1 이후 Task 단위 재정의
- [x] 공통 함수명과 입출력 규약 정의
- [x] 실패 테스트 원칙과 pytest 실행 명령 정리
- [x] docs/stage0/phase0.3_implementation-order.md

## Stage 1 기본 설정 · 과제 규약

### Phase 1.1 config — 기본 경로, 실행 기본값

- [x] requirements.txt
- [x] src/config.py
- [x] tests/stage1/test_config.py
- [x] docs/stage1/phase1.1_config.md

### Phase 1.2 task — 과제 규약, target 변환, loss, metric

- [x] src/task.py
- [x] tests/stage1/test_task.py
- [x] docs/stage1/phase1.2_task.md

### Phase 1.3 utils — 배치 처리, 난수 시드, 파일 I/O

- [x] src/utils/batching.py
- [x] tests/stage1/test_batching.py
- [x] src/utils/random.py
- [x] tests/stage1/test_random.py
- [x] src/utils/io.py
- [x] tests/stage1/test_io.py
- [x] docs/stage1/phase1.3_utils.md

## Stage 2 MNIST 데이터 로더

### Phase 2.1 mnist — gz 로딩, split 처리, task 변환 통합

- [x] src/data/mnist.py
- [x] tests/stage2/test_mnist.py
- [x] docs/stage2/phase2.1_mnist.md

## Stage 3 NumPy MLP

### Phase 3.1 mlp — forward, backward, parameter update

- [ ] src/models/mlp.py
- [ ] tests/stage3/test_mlp.py
- [ ] docs/stage3/phase3.1_mlp.md

### Phase 3.2 layers · activations · losses — from-scratch 구성요소

- [ ] src/models/layers.py
- [ ] tests/stage3/test_layers.py
- [ ] src/models/activations.py
- [ ] tests/stage3/test_activations.py
- [ ] src/models/losses.py
- [ ] tests/stage3/test_losses.py
- [ ] docs/stage3/phase3.2_nn.md

## Stage 4 실행 객체

### Phase 4.1 checkpoints — 파라미터 저장, 로딩

- [ ] src/core/checkpoints.py
- [ ] tests/stage4/test_checkpoints.py
- [ ] docs/stage4/phase4.1_checkpoints.md

### Phase 4.2 trainer — 학습 루프, 배치 처리

- [ ] src/core/trainer.py
- [ ] tests/stage4/test_trainer.py
- [ ] docs/stage4/phase4.2_trainer.md

### Phase 4.3 evaluator — 평가 루프, 지표 집계

- [ ] src/core/evaluator.py
- [ ] tests/stage4/test_evaluator.py
- [ ] docs/stage4/phase4.3_evaluator.md

### Phase 4.4 predictor — 예측, task별 후처리

- [ ] src/core/predictor.py
- [ ] tests/stage4/test_predictor.py
- [ ] docs/stage4/phase4.4_predictor.md

### Phase 4.5 experiment — 실행 객체 조립, 최상위 진입점

- [ ] src/core/experiment.py
- [ ] tests/stage4/test_experiment.py
- [ ] docs/stage4/phase4.5_experiment.md

### Phase 4.6 visualizer — 학습 로그, 예측 결과 시각화

- [ ] src/core/visualizer.py
- [ ] tests/stage4/test_visualizer.py
- [ ] docs/stage4/phase4.6_visualizer.md

## Stage 5 클라이언트 코드

### Phase 5.1 train — 학습 CLI, experiment · trainer 호출

- [ ] scripts/train.py
- [ ] tests/stage5/test_train.py
- [ ] docs/stage5/phase5.1_train.md

### Phase 5.2 evaluate — 평가 CLI, experiment · evaluator 호출

- [ ] scripts/evaluate.py
- [ ] tests/stage5/test_evaluate.py
- [ ] docs/stage5/phase5.2_evaluate.md

### Phase 5.3 predict — 예측 CLI, experiment · predictor 호출

- [ ] scripts/predict.py
- [ ] tests/stage5/test_predict.py
- [ ] docs/stage5/phase5.3_predict.md

### Phase 5.4 visualize — 시각화 CLI, experiment · visualizer 호출

- [ ] scripts/visualize.py
- [ ] tests/stage5/test_visualize.py
- [ ] docs/stage5/phase5.4_visualize.md

## Stage 6 CuPy CNN

### Phase 6.1 cnn — forward, backward (CuPy 기반)

- [ ] src/models/cnn.py
- [ ] tests/stage6/test_cnn.py
- [ ] docs/stage6/phase6.1_cnn.md

### Phase 6.2 CNN-core 연동 — core 인터페이스 통합 검증

- [ ] tests/stage6/test_experiment.py (CNN 통합 케이스 추가)
- [ ] docs/stage6/phase6.2_cnn-integration.md

## Stage 7 문서화 · 검증

### Phase 7.1 튜토리얼 — Jupyter Book 챕터 작성

- [ ] docs/stage7/ 챕터 작성

### Phase 7.2 실행 예제 — task별 로그, 평가 결과, 시각화 정리

- [ ] outputs/ 결과 정리
- [ ] docs/stage7/phase7.2_results.md

### Phase 7.3 프레임워크 연계 — 후속 프로젝트 초기화 체크리스트

- [ ] docs/stage7/phase7.3_framework-checklist.md
