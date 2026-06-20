---
tags: [docs, stage1, overview]
created: 2026-06-19
updated: 2026-06-19
---

# Stage 1 config 및 task 규약

## 1. 개요

Stage 1은 프로젝트 전체에서 공유하는 기본 설정과 과제별 규약을 코드로 확정하는 단계이다.
Stage 0에서 수립한 구현 계획을 바탕으로 `src/config.py`, `src/task.py`, `src/utils/` 세 영역을 구현하고 각각에 대응하는 테스트를 작성한다.
이 Stage가 완료되면 이후 Stage에서 사용하는 기본 경로, 실행 기본값, 과제별 target/loss/metric 규약, 공통 utility가 모두 확정된다.

## 2. Phase 구성

### 2.1. Phase 1.1 config 구성

`src/config.py`에 데이터셋 기본 경로(`DATASET_DIR`), 기본 split, 기본 task, batch size, epoch, learning rate, seed 등 프로젝트 전반에서 공유하는 기본값을 정의한다.
`get_default_config()` 함수가 dict를 반환하며, 이후 Experiment 조립 시 기본값으로 사용한다.

- [[phase1.1_config|Phase 1.1 config 구성]]

### 2.2. Phase 1.2 과제 규약 정의

`src/task.py`에 multiclass, binary, regression 세 가지 과제의 output_dim, activation, loss 함수, metric 함수, prediction 후처리 규약을 단일 진입점으로 관리한다.
`get_task_spec(task)` 함수가 규약 dict를 반환하고, `transform_targets(labels, task)` 함수가 target 배열 변환을 담당한다.
이 파일 하나로 과제별 차이를 흡수하여 Trainer, Evaluator, Predictor가 task를 직접 분기하지 않아도 된다.

- [[phase1.2_task|Phase 1.2 과제 규약 정의]]

### 2.3. Phase 1.3 utility 구현

`src/utils/` 하위에 mini-batch 처리, 난수 시드 고정, 파일 I/O 세 가지 utility를 구현한다.
`batching.py`는 epoch당 mini-batch 인덱스 생성과 shuffle을 담당하고, `random.py`는 NumPy seed 고정을 제공하며, `io.py`는 결과 파일 저장과 로딩 보조 함수를 제공한다.

- [[phase1.3_utils|Phase 1.3 utility 구현]]

## 3. 주요 산출물

| 산출물 | 내용 |
|---|---|
| `src/config.py` | 기본 경로, 실행 기본값, hyperparameter 정의 |
| `src/task.py` | 과제별 output_dim, loss, metric, target 변환 규약 |
| `src/utils/batching.py` | mini-batch 인덱스 생성 및 shuffle |
| `src/utils/random.py` | NumPy 난수 시드 고정 |
| `src/utils/io.py` | 파일 저장/로딩 보조 함수 |
| `tests/stage1/` | config, task, batching, random, io 테스트 파일 5개 |
