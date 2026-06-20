---
tags: [session, handoff]
created: 2026-06-20
---

# Stage-Phase 목차 재설계 및 SPEC 반영 세션 핸드오프

> 작성일시: 260620-095629
> 세션 목적: 신규 Stage 0~7 목차 확정 및 PROJECT-SPEC.md 반영
> 이전 핸드오프: 260620-090836_session-handoff.md

## 1. 세션 핵심 요약

이번 세션에서 신규 Stage 0~7 목차를 확정하고 PROJECT-SPEC.md를 전면 재작성했다.
핵심 변경 사항은 Stage 1(config/task 규약) 제거, `src/utils/` 독립 Stage 1 신설, Stage 7까지 번호 재조정이다.
PROJECT-BOOK-PLAN.md(삭제 예정)에서 누락된 원칙을 docs-rules.md, template-rules.md, PROJECT-SPEC.md에 반영했다.
PROJECT-TODO.md는 다음 세션 작업으로 남겨두었다.

## 2. 사용자 요청 및 의도

| 요청 내용 | 배경 목적 |
|---|---|
| 구 Stage 1(config/task) 제거 | config.py, task.py 삭제 방침과 일관성 유지 |
| utils 독립 Stage로 분리 | src/ 폴더 1개 = Stage 1개 원칙 일관 적용 |
| 레거시 코드 분석을 Phase 0.0으로 이동 | numpy 전용 Phase임을 명시, 후속 프레임워크에서 생략 가능 |
| 마지막 노트북 Phase 공통 네이밍 | Stage 번호 미표기, "실습 노트북 작성"으로 통일 |
| PROJECT-BOOK-PLAN.md 보존 원칙 확인 | 삭제 전 유지할 지침을 공식 문서에 반영 |
| PROJECT-SPEC.md 신규 목차 반영 | Stage 구조와 § 6 확정 구조를 동기화 |

## 3. 확정된 결정사항

| 항목 | 확정 내용 | 비고 |
|---|---|---|
| Stage 구조 | Stage 0~7 (8개 Stage) | 폴더 1개 = Stage 1개 |
| Stage 1 | 공통 유틸리티 (`src/utils/`) | 구 Stage 1(config/task) 대체 |
| Phase 0.0 | 레거시 코드 분석 (numpy 전용) | 후속 프레임워크 책에서 생략 |
| 마지막 Phase 명 | "실습 노트북 작성" | Stage 번호 미포함 |
| utils 폴더 | `src/utils/` 유지 | `checkpoints.py` → `src/utils/`로 이동 |
| checkpoints.py | `src/core/` → `src/utils/` 이동 | 프레임워크 무관 공통 기능 |
| Stage 3 (nn) | numpy 전용 직접 구현, 후속 프레임워크는 번호·제목 유지 내용 교체 | |
| notebooks/ | stage1~7 신규 번호 체계, 16개 유지 | stage7 신규 추가 |
| tests/ | stage1~6 신규 번호 체계 | |
| PROJECT-BOOK-PLAN.md | 삭제 예정 (원칙 반영 완료) | |

## 4. 미결 사항

| # | 항목 | 현재 상태 | 결정 필요 내용 |
|---|---|---|---|
| 1 | PROJECT-BOOK-PLAN.md 삭제 | 보류 | 사용자 최종 확인 후 삭제 |
| 2 | src/ 코드 실제 변경 | 미착수 | config.py, task.py, experiment.py 삭제 및 영향 파일 수정 |
| 3 | tests/ 폴더 실제 이동 | 미착수 | git mv로 stage 번호 재조정 |
| 4 | notebooks/ 파일 실제 이동 | 미착수 | stage 번호 재조정 및 파일명 변경 |
| 5 | docs/ 파일 이동 및 번호 재조정 | 미착수 | Phase 번호 변경 + MLP/CNN 통합 재작성 |

## 5. 다음 작업 목록

| 우선순위 | 작업 | 관련 파일 |
|---|---|---|
| 1 | **PROJECT-TODO.md 재작성** | `_core/PROJECT-TODO.md` |
| 2 | src/ 코드 변경 (삭제 및 이동) | `src/config.py`, `src/task.py`, `src/core/experiment.py`, `src/core/checkpoints.py` |
| 3 | tests/ 폴더 번호 재조정 (git mv) | `tests/stage1~5/` |
| 4 | pytest 전체 통과 확인 | `conda run -n numpy_py311 pytest tests/ -q` |
| 5 | docs/ 파일 이동 및 번호 재조정 | `docs/stage1~6/` |
| 6 | notebooks/ 파일 이동 및 번호 재조정 | `notebooks/stage1~6/` |
| 7 | PROJECT-BOOK-PLAN.md 삭제 | `_core/PROJECT-BOOK-PLAN.md` |

## 6. 다음 세션 시작 지시문

아래는 이전 세션의 컨텍스트입니다.
이 내용을 기반으로 PROJECT-TODO.md 재작성을 진행해 주세요.

- 신규 Stage 0~7 구조로 전면 재작성
- 완료 항목([x])은 새 Phase 위치에 재배치
- SPEC § 5의 Stage-Phase 제목과 동기화
- 기존 `## 2. Stage 1` 항목(config/task)은 Stage 1 공통 유틸리티로 대체
- 기존 `## 3. Stage 2~7`은 각각 `## 3. Stage 2` ~ `## 8. Stage 7`로 번호 이동

참고 파일:
- 핸드오프: `_core/sessions/260620-095629_session-handoff.md`
- SPEC: `_core/PROJECT-SPEC.md` (신규 Stage 0~7 반영 완료)
- TODO: `_core/PROJECT-TODO.md` (재작성 필요)
