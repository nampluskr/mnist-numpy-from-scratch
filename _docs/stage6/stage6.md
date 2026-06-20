---
tags: [docs, stage6, overview]
created: 2026-06-19
updated: 2026-06-20
---

# Stage 6 문서화 및 검증

## 1. 개요

Stage 6은 Stage 1~5에서 구현한 전체 파이프라인을 실제로 실행하고 결과를 수집한 뒤, 과제별 tutorial 문서를 작성하고 후속 PyTorch 프로젝트 연계를 준비하는 단계이다.
CLI 스크립트에 `--model` 플래그를 추가하여 MLP와 CNN 모두 동일한 진입점에서 실행할 수 있도록 확장하고, multiclass, binary, regression 각각에 대해 MLP와 CNN 두 가지 모델의 실행 결과를 수집한다.
이 Stage가 완료되면 6종의 experiment 결과물이 `outputs/`에 저장되고, 과제별 tutorial과 framework 연계 checklist가 작성된다.

## 2. Phase 구성

### 2.1. Phase 6.1 experiment 실행 및 result 수집

multiclass, binary, regression 각 과제에 대해 MLP와 CNN 두 모델을 실행하여 총 6종의 experiment 결과를 수집한다.
각 experiment의 `training_log.png`, `predictions.png`, `model.npz`를 `outputs/{task}/{model}/` 경로에 저장하고, 결과 요약을 문서화한다.

- [[phase6.1_results|Phase 6.1 experiment 실행 및 result 수집]]

### 2.2. Phase 6.2 Multiclass tutorial

MNIST 10-class 분류 과제를 대상으로 MLP와 CNN 각각의 실행 절차, 주요 파라미터, 평가 결과를 tutorial 문서로 작성한다.

- [[phase6.2_tutorial-mlp|Phase 6.2 Multiclass tutorial (MLP)]]
- [[phase6.2_tutorial-cnn|Phase 6.2 Multiclass tutorial (CNN)]]

### 2.3. Phase 6.3 Binary tutorial

홀수/짝수 이진 분류 과제를 대상으로 MLP와 CNN 각각의 실행 절차, 주요 파라미터, 평가 결과를 tutorial 문서로 작성한다.

- [[phase6.3_tutorial-mlp|Phase 6.3 Binary tutorial (MLP)]]
- [[phase6.3_tutorial-cnn|Phase 6.3 Binary tutorial (CNN)]]

### 2.4. Phase 6.4 Regression tutorial

레이블 값을 0~1로 정규화하는 회귀 과제를 대상으로 MLP와 CNN 각각의 실행 절차, 주요 파라미터, 평가 결과를 tutorial 문서로 작성한다.

- [[phase6.4_tutorial-mlp|Phase 6.4 Regression tutorial (MLP)]]
- [[phase6.4_tutorial-cnn|Phase 6.4 Regression tutorial (CNN)]]

### 2.5. Phase 6.5 framework 연계 준비

이 프로젝트에서 확정된 모듈명, 함수명, 인터페이스 규약을 정리하고, 후속 PyTorch 프로젝트로 이전할 때 변경이 필요한 항목과 유지할 항목을 checklist 형태로 작성한다.

- [[phase6.5_framework-checklist|Phase 6.5 framework 연계 준비]]

## 3. 주요 산출물

| 산출물 | 내용 |
|---|---|
| `outputs/multiclass/mlp/` | training_log.png, predictions.png, model.npz |
| `outputs/multiclass/cnn/` | training_log.png, predictions.png, model.npz |
| `outputs/binary/mlp/` | training_log.png, predictions.png, model.npz |
| `outputs/binary/cnn/` | training_log.png, predictions.png, model.npz |
| `outputs/regression/mlp/` | training_log.png, predictions.png, model.npz |
| `outputs/regression/cnn/` | training_log.png, predictions.png, model.npz |
| `docs/stage6/phase6.1_results.md` | 6종 experiment 평가 결과 요약 |
| `docs/stage6/phase6.2~6.4_tutorial-*.md` | 과제별 MLP/CNN tutorial 문서 6개 |
| `docs/stage6/phase6.5_framework-checklist.md` | PyTorch 이전 checklist |
