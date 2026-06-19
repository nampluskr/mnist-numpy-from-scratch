---
tags: [docs, stage0, overview]
created: 2026-06-19
updated: 2026-06-19
---

# Stage 0 레거시 코드 분석 및 계획 수립

## 1. 개요

Stage 0은 사용자가 제공한 레거시 코드를 분석하여 `src/` 구현 방향과 테스트 전략을 수립하는 단계이다.
레거시 코드는 `_core/legacy/src/`에 보관된 task 스크립트 6개(multiclass, binary, regression 각 manual, module 2종)와 공통 모듈 6개(mnist, functions, modules, optimizers, dataloader, trainer)로 구성된다.
이 Stage가 완료되면 레거시와 신규 구현 간의 매핑, Stage 1 이후 파일별 책임 범위, TDD 원칙이 모두 확정된다.

## 2. Phase 구성

### 2.1. Phase 0.1 레거시 코드 분석

레거시 코드 6개 task 스크립트와 6개 common 모듈의 전체 구조를 파악한다.
manual 방식(직접 행렬 연산)과 module 방식(클래스 기반 레이어)의 두 가지 구현 패턴을 비교하고, multiclass, binary, regression 과제별로 다르게 처리되는 항목(target 변환, output dim, loss, metric, gradient, 후처리)을 도출한다.

- [[phase0.1_legacy-analysis|Phase 0.1 레거시 코드 분석]]

### 2.2. Phase 0.2 구현 계획 수립

레거시 common 모듈에서 `src/` 파일로의 1:1 매핑을 확정하고, `src/` 패키지 구조(`data/`, `nn/`, `models/`, `core/`, `utils/`)와 각 파일의 책임 범위를 결정한다.
Stage 1~7 구현 순서와 Phase 단위 분할도 이 단계에서 확정한다.

- [[phase0.2_implementation-plan|Phase 0.2 구현 계획 수립]]

### 2.3. Phase 0.3 테스트 계획 수립

`tests/` 폴더의 stage 단위 구조와 `__init__.py` 없는 규칙을 확정한다.
파일별 공개 인터페이스 규약(진입점, 입력, 출력)과 TDD 원칙(synthetic array 우선, tolerance 비교 등)을 수립하고, pytest 실행 명령을 stage 단위, 단일 파일, 전체 세 가지로 정리한다.

- [[phase0.3_test-plan|Phase 0.3 테스트 계획 수립]]

## 3. 주요 산출물

| 산출물 | 내용 |
|---|---|
| `docs/stage0/phase0.1_legacy-analysis.md` | 레거시 코드 구조, 두 가지 구현 패턴, task별 차이 정리 |
| `docs/stage0/phase0.2_implementation-plan.md` | 레거시-src 매핑, 패키지 구조, Stage 1~7 구현 순서 |
| `docs/stage0/phase0.3_test-plan.md` | tests 폴더 구조, 인터페이스 규약, TDD 원칙, pytest 실행 기준 |
