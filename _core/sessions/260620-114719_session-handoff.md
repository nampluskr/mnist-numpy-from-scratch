---
tags: [session, handoff]
created: 2026-06-20
---

# Stage 0 docs/ 전체 작성 세션 핸드오프

> 작성일시: 260620-114719
> 세션 목적: Stage 0 Phase 0.0~0.3 docs/ 문서 전체 작성
> 이전 핸드오프: 260620-112500_session-handoff.md

## 1. 세션 핵심 요약

Stage 0의 Phase 0.0~0.3 문서 5개를 `docs/stage0/`에 새로 작성했다.
기존 `_docs/stage0/` 문서를 참고하되, 현재 프로젝트 구조(config/task/experiment 삭제, checkpoints 이동, Stage 0-6 체계)를 반영하여 전면 재작성했다.
README.md도 현재 src/ 구조, conda run -n 형식, Stage 0-6 로드맵으로 갱신했다.
PROJECT-TODO.md Stage 0 전체 완료 처리 완료.

## 2. 확정된 결정사항

| 항목 | 확정 내용 | 비고 |
|---|---|---|
| 확인된 환경 버전 | numpy_py311: NumPy 2.4.6 / cupy_py311_cuda118: CuPy 13.6.0 / cupy_py311_cuda121: CuPy 14.1.1 | phase0.1 문서에 표로 정리 |
| 실행 명령 형식 | `conda run -n {환경명}` 전 파일 통일 | README.md 갱신 완료 |
| src/ 패키지 구조 | nn/data/models/core/utils 5개 하위 패키지 | phase0.2 문서에 확정 |
| task 규약 관리 위치 | `src/data/mnist.py`의 `get_task_spec()` | phase0.2_framework-interface 문서에 정의 |

## 3. 미결 사항

| # | 항목 | 현재 상태 | 내용 |
|---|---|---|---|
| 1 | tests/ 폴더 번호 재조정 | 미착수 | stage1~6 신규 번호 체계로 git mv |
| 2 | notebooks/ 파일 번호 재조정 | 미착수 | stage1~6 신규 번호 체계로 git mv |
| 3 | docs/ 파일 번호 재조정 | 미착수 | stage1~6 신규 번호 체계 + 파일명 변경 |

## 4. 다음 작업 목록

| 우선순위 | 작업 | 관련 파일 |
|---|---|---|
| 1 | tests/ 폴더 번호 재조정 (git mv) | `tests/stage1~5/` |
| 2 | pytest 전체 통과 확인 | `conda run -n numpy_py311 pytest tests/ -q` |
| 3 | notebooks/ 파일 번호 재조정 (git mv) | `notebooks/stage1~6/` |
| 4 | docs/ 파일 번호 재조정 | `docs/stage1~6/` |

## 5. 다음 세션 시작 지시문

아래는 이전 세션의 컨텍스트입니다.
이 내용을 기반으로 tests/ 폴더 번호 재조정 작업을 진행해 주세요.

- Stage 0 docs/ 전체 작성 완료 (phase0.0~0.3 신규 5개, README.md 갱신)
- 다음 작업: tests/ 폴더 번호 재조정 (stage1~5 신규 번호 체계로 git mv)
- 이후: notebooks/ -> docs/ 순서로 파일 번호 재조정

참고 파일:
- SPEC: `_core/PROJECT-SPEC.md`
- TODO: `_core/PROJECT-TODO.md`
- 핸드오프: `_core/sessions/260620-114719_session-handoff.md`
