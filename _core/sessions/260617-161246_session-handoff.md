---
tags: [project, session]
created: 2026-06-17
updated: 2026-06-17
---

# 세션 핸드오프 — 260617-161246

## 1. 이번 세션 완료 항목

레거시 코드 전체 구조 추가 및 검증, Stage 0 문서 전면 재작성, 운영 문서 업데이트를 완료했다.

| 항목 | 내용 |
|---|---|
| 레거시 코드 구조화 | `binary/`, `multiclass/`, `regression/` 각 task별 manual·module 스크립트 + `common/` 6개 모듈 추가 |
| `common/mnist.py` 신규 작성 | `load_images`, `load_labels`, `one_hot` 구현, numpy_env 실행 검증 완료 |
| 레거시 실행 검증 | 3개 task × 2 패턴(manual·module) = 6개 스크립트 모두 동일 결과 확인 |
| `docs/stage0/phase0.1_legacy-review.md` | 레거시 구성(6+6), common 모듈 상세, 두 가지 패턴, task별 차이, 구현 매핑 전면 재작성 |
| `docs/stage0/phase0.2_structure.md` | src 구조(optimizers 추가), tests 구조(stage 단위), scripts-core 관계 재작성 |
| `docs/stage0/phase0.3_implementation-order.md` | Stage별 순서, 인터페이스 규약(optimizer 포함), TDD 원칙, pytest 명령 재작성 |
| `_core/PROJECT-SPEC.md` | src 구조(layers·activations·losses·optimizers 추가), tests stage 단위 구조, 인터페이스 규약 전면 업데이트 |
| `_core/PROJECT-TODO.md` | Stage 0 미완료 초기화(다음 세션 재진행), Stage 4 Phase 4.1 optimizers 추가 및 4.2~4.7 재번호 |

## 2. 주요 결정사항

| 항목 | 결정 내용 |
|---|---|
| Stage 0 재진행 | "설계 재검토" 프레임 제거, 레거시 분석 → 구현 계획 → 테스트 계획 수립 순서로 재편 |
| 새 Phase 파일명 | `phase0.1_legacy-analysis.md`, `phase0.2_implementation-plan.md`, `phase0.3_test-plan.md` |
| `src/core/optimizers.py` 신설 | 레거시 `common/optimizers.py` 대응, Stage 4 Phase 4.1로 배치 |
| Stage 4 Phase 번호 | 4.1 optimizers → 4.2 checkpoints → 4.3 trainer → 4.4 evaluator → 4.5 predictor → 4.6 experiment → 4.7 visualizer |
| tests 구조 | stage 단위(`stage1/`~`stage5/`), `__init__.py` 없음 |

## 3. 현재 진행 상태

| Stage | 상태 |
|---|---|
| Stage 0 | 미완료 (다음 세션에서 phase0.1~0.3 재작성 예정) |
| Stage 1 | 완료 (config, task, utils) |
| Stage 2 | 완료 (mnist, dataset, dataloader) |
| Stage 3 | Phase 3.1 완료 (mlp), Phase 3.2 미완료 (layers, activations, losses) |
| Stage 4~7 | 미착수 |

## 4. 다음 세션 시작 지시문

"session-start 실행 후 Stage 0 레거시 코드 분석 및 계획 수립 (Phase 0.1~0.3) 재작성을 이어서 진행해 주세요."

### 4.1. 다음 세션 작업 목록

다음 세션에서 진행할 Stage 0 Phase 별 작업은 다음과 같다.

**Phase 0.1 — 레거시 코드 분석** (`docs/stage0/phase0.1_legacy-analysis.md`)
- 레거시 코드 구조 (task 스크립트 6개 + common 모듈 6개)
- common 모듈별 제공 요소 정리
- manual · module 두 가지 구현 패턴 비교
- task별 차이 (target 변환, output dim, loss, metric, gradient, 후처리)

**Phase 0.2 — 구현 계획 수립** (`docs/stage0/phase0.2_implementation-plan.md`)
- 레거시 common 모듈 → src 파일 1:1 매핑
- src 패키지 구조 및 각 파일 책임 범위
- Stage 1~7 구현 순서 및 Phase 단위 분할

**Phase 0.3 — 테스트 계획 수립** (`docs/stage0/phase0.3_test-plan.md`)
- tests 폴더 구조 (stage 단위, `__init__.py` 없음)
- 파일별 공개 인터페이스 규약 (진입점, 입력, 출력)
- TDD 원칙 및 pytest 실행 명령

### 4.2. 참고 사항

- 기존 `phase0.1_legacy-review.md`, `phase0.2_structure.md`, `phase0.3_implementation-order.md` 는 새 파일로 대체한다.
- `src/models/`, `tests/stage3/`, `docs/stage3/` 는 미커밋 상태 (Phase 3.1 산출물)로 이번 세션 커밋에 포함된다.
