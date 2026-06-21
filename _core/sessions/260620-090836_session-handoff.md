---
tags: [session, handoff]
created: 2026-06-20
---

# 세션 핸드오프 - 260620-090836

## 1. 세션 요약

이번 세션에서는 `_core/PROJECT-BOOK-PLAN.md`를 검토하여 프레임워크 공통 관점의 새 Stage-Phase 목차를 설계했다.
코드 변경은 없었고 목차 설계 확정이 이번 세션의 전부다.

## 2. 확정된 설계

### 2.1. 핵심 원칙

- **src/ 하위 폴더 1개 = Stage 1개** (Chapter 단위)
- **Stage 3 (nn module)**: numpy 책에서는 직접 구현, 후속 프레임워크 책(PyTorch 등)에서는 번호·제목 유지하고 내용만 교체
- **목표 독자**: Python + NumPy 사용자, 딥러닝 초보자
- **상호 참조**: 모든 프레임워크 책이 동일한 Stage 번호·제목 공유

### 2.2. 목표 Stage-Phase 목차

```
Stage 0  환경 구성과 프로젝트 설계       (notebook 없음)
  0.1  실행 환경 구성
  0.2  레거시 코드 분석                  (numpy 전용 Phase)
  0.3  package 구조와 구현 계획
  0.4  테스트 전략

Stage 1  config, task 규약, utility      src/config.py + src/task.py + src/utils/
  1.1  config 구성
  1.2  task 규약과 target 변환
  1.3  공통 utility
  1.4  Chapter 1 실습

Stage 2  MNIST data pipeline             src/data/
  2.1  MNIST raw data loading
  2.2  Dataset과 target 변환
  2.3  DataLoader와 mini-batch
  2.4  Chapter 2 실습

Stage 3  nn module  (numpy 전용)         src/nn/
  3.1  activation 함수
  3.2  loss와 gradient
  3.3  metric 함수
  3.4  layer와 Module interface
  3.5  convolution과 pooling layer
  3.6  Chapter 3 실습

Stage 4  model  [신규]                   src/models/
  4.1  MLP model
  4.2  CNN model
  4.3  Chapter 4 실습

Stage 5  학습 및 실행 framework          src/core/
  5.1  optimizer
  5.2  checkpoint
  5.3  Trainer
  5.4  Evaluator
  5.5  Predictor
  5.6  Visualizer
  5.7  Chapter 5 실습

Stage 6  CLI와 실험 자동화              scripts/ + experiments/
  6.1  training CLI
  6.2  evaluation CLI
  6.3  prediction CLI
  6.4  visualization CLI
  6.5  batch experiment 실행
  6.6  Chapter 6 실습

Stage 7  실험 결과와 framework 연계     outputs/
  7.1  실험 조건과 결과 개요
  7.2  multiclass MLP-CNN 비교
  7.3  binary MLP-CNN 비교
  7.4  regression MLP-CNN 비교
  7.5  framework interface와 migration
  7.6  Chapter 7 실습
```

### 2.3. 현재 구조 대비 주요 변경

| 변경 | 내용 |
|---|---|
| Stage 0 Phase 번호 | 0.0~0.3 → 0.1~0.4 |
| Stage 3 | nn만 (MLP/CNN → Stage 4 분리), 7 Phase → 6 Phase |
| Stage 4 신규 | models/ 전담 |
| Stage 4→5, 5→6, 6→7 | 번호 이동 |
| Stage 5에서 experiment Phase 제거 | experiment.py 삭제 예정 |
| Stage 6 Phase 6.5 신규 | experiments/run_*.py batch 설명 |
| Stage 7 Phase 7.2~7.4 | MLP/CNN 각각 → task별 비교 문서 1개 통합 |

## 3. 현재 상태

- 브랜치: `refactor/book-notebook-restructure`
- 커밋: `14d81b6` (직전 세션과 동일, 이번 세션 코드 변경 없음)
- 설계 확정만 완료. 실행은 다음 세션부터.

## 4. 다음 세션 최우선 작업

### 우선 1: PROJECT-SPEC.md 재작성

`## 5. 진행 단계`를 위 Stage 0~7 목차로 전면 교체.
`## 6. 확정 구조`의 src 패키지 구조 갱신 (config.py, task.py 삭제 / experiment.py 삭제 반영).
tests/, notebooks/ 구조 설명도 새 Stage 번호에 맞게 갱신.

### 우선 2: PROJECT-TODO.md 재편

새 Stage 0~7 구성으로 전면 재작성. 완료 항목([x])은 새 Phase 위치에 재배치.

### 우선 3: src/ 코드 변경

1. `src/config.py`, `src/task.py`, `src/core/experiment.py` 삭제
2. 영향 파일 수정 ( MNISTDataset 내부에 task 변환 로직 흡수, scripts/*.py 직접 조립 패턴)
3. `src/core/logger.py` 신규 구현
4. `experiments/run_*.py` CONFIGS에 task 규약 key 추가

### 우선 4: 파일 이동 (git mv)

tests/, docs/, notebooks/ 폴더 번호 재조정 (Stage 3→4→5→6→7 이동 포함).

### 우선 5: pytest 전체 통과 확인

```bash
conda run -n numpy_py311 pytest tests/ -q
```

### 우선 6: docs/ 파일 이동 및 번호 재조정

Stage 문서 파일명 번호 변경 + Phase 6.2~6.4 MLP/CNN 문서 통합 재작성.

### 우선 7: notebooks/ 재작성

새 template 기준 Stage별 재작성.

## 5. 다음 세션 시작 지시문

"session-start 실행 후 PROJECT-SPEC.md를 새 Stage 0~7 목차로 재작성하는 것부터 시작해 주세요. 핸드오프 260620-090836의 목차를 기준으로 하며, PROJECT-BOOK-PLAN.md § 4의 내용도 함께 참조해 주세요."
