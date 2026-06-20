---
tags: [session, handoff]
created: 2026-06-20
---

# PROJECT-TODO.md 신규 Stage 0~7 구조 재작성 세션 핸드오프

> 작성일시: 260620-103203
> 세션 목적: PROJECT-TODO.md 신규 Stage 0~7 구조 전면 재작성 및 SPEC 정비
> 이전 핸드오프: 260620-095629_session-handoff.md

## 1. 세션 핵심 요약

PROJECT-TODO.md를 신규 Stage 0~7 구조로 전면 재작성하고 모든 체크박스를 미완료로 초기화했다.
PROJECT-SPEC.md에서 Stage 7을 삭제하고 실험 노트북을 Stage 6 Phase 6.6으로 통합했다.
PROJECT-BOOK-PLAN.md를 삭제했다 (원칙은 이미 공식 문서에 반영 완료).
Phase 명칭 일부를 정비했다 (개발 환경 구성, MNIST 데이터 로딩, MLP/CNN 레이어 구현).

## 2. 확정된 결정사항

| 항목 | 확정 내용 | 비고 |
|---|---|---|
| Stage 수 | Stage 0~6 (7개) | Stage 7 삭제 |
| Stage 6 Phase 6.6 | 노트북 4개 (cli + multiclass/binary/regression) | 실험 비교는 노트북으로만 |
| framework 연계 문서 | Phase 0.2 마지막 task | docs/stage0/phase0.2_framework-interface.md |
| PROJECT-BOOK-PLAN.md | 삭제 완료 | |
| 체크박스 상태 | 전체 미완료 | 문서·노트북 템플릿 기준 재작성 예정 |
| Phase 0.1 | 개발 환경 구성 (conda → 개발) | |
| Phase 2.1 | MNIST 데이터 로딩 (raw data loading → 한글) | |
| Phase 3.4 / 3.5 | MLP 레이어 구현 / CNN 레이어 구현 | |

## 3. 미결 사항

| # | 항목 | 현재 상태 | 결정 필요 내용 |
|---|---|---|---|
| 1 | src/ 코드 변경 | 미착수 | config.py, task.py, experiment.py 삭제 및 영향 파일 수정 |
| 2 | checkpoints.py 이동 | 미착수 | src/core/ → src/utils/ git mv |
| 3 | tests/ 폴더 번호 재조정 | 미착수 | stage1~6 신규 번호 체계로 git mv |
| 4 | notebooks/ 파일 번호 재조정 | 미착수 | stage1~6 신규 번호 체계로 git mv |
| 5 | docs/ 파일 번호 재조정 | 미착수 | stage1~6 신규 번호 체계 + 파일명 변경 |
| 6 | pytest 전체 통과 확인 | 미착수 | 코드 변경 후 실행 |

## 4. 다음 작업 목록

| 우선순위 | 작업 | 관련 파일 |
|---|---|---|
| 1 | src/ 코드 변경 (삭제 및 이동) | `src/config.py`, `src/task.py`, `src/core/experiment.py`, `src/core/checkpoints.py` |
| 2 | tests/ 폴더 번호 재조정 (git mv) | `tests/stage1~5/` |
| 3 | pytest 전체 통과 확인 | `conda run -n numpy_py311 pytest tests/ -q` |
| 4 | notebooks/ 파일 번호 재조정 (git mv) | `notebooks/stage1~6/` |
| 5 | docs/ 파일 번호 재조정 | `docs/stage1~6/` |

## 5. 다음 세션 시작 지시문

아래는 이전 세션의 컨텍스트입니다.
이 내용을 기반으로 src/ 코드 변경 작업을 진행해 주세요.

- 신규 Stage 0~6 구조 확정 완료 (SPEC, TODO 동기화 완료)
- 다음 작업: src/ 코드 변경 (config.py, task.py, experiment.py 삭제, checkpoints.py 이동)
- 이후: tests/ → notebooks/ → docs/ 순서로 파일 번호 재조정

참고 파일:
- SPEC: `_core/PROJECT-SPEC.md`
- TODO: `_core/PROJECT-TODO.md`
- 핸드오프: `_core/sessions/260620-103203_session-handoff.md`
