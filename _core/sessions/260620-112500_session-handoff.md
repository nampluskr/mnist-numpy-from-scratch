---
tags: [session, handoff]
created: 2026-06-20
---

# src/ 코드 변경 세션 핸드오프

> 작성일시: 260620-112500
> 세션 목적: 미결 사항 1번 - src/ 코드 변경 (config/task/experiment 삭제, checkpoints 이동)
> 이전 핸드오프: 260620-110844_session-handoff.md

## 1. 세션 핵심 요약

이전 세션 핸드오프의 미결 사항 1번, 2번, 6번을 완료했다.
`src/config.py`, `src/task.py`, `src/core/experiment.py`를 삭제하고,
`src/task.py`의 `get_task_spec()` / `transform_targets()` 함수를 `src/data/mnist.py`에 흡수했다.
`src/core/checkpoints.py`를 `src/utils/checkpoints.py`로 이동(git mv)했다.
scripts/ 4개를 `Experiment` 없이 구성 요소를 직접 조립하는 방식으로 재작성했다.
pytest 351 passed, 8 skipped 전체 통과 확인 완료.

## 2. 확정된 결정사항

| 항목 | 확정 내용 | 비고 |
|---|---|---|
| `get_task_spec()` 위치 | `src/data/mnist.py` | task 규약은 dataset과 함께 관리 |
| `transform_targets()` 위치 | `src/data/mnist.py` | 동일 파일 내 흡수 |
| scripts/ 기본값 | 각 스크립트 `_DEFAULTS` 상수 | `get_default_config()` 대체 |
| `Experiment` 대체 | 각 스크립트 직접 조립 | PROJECT-SPEC §6.3 기준 |
| `checkpoints` 위치 | `src/utils/checkpoints.py` | `src/core/` 에서 이동 |

## 3. 미결 사항

| # | 항목 | 현재 상태 | 결정 필요 내용 |
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

- src/ 코드 변경 완료 (config/task/experiment 삭제, checkpoints 이동, scripts 재작성)
- 다음 작업: tests/ 폴더 번호 재조정 (stage1~5 신규 번호 체계로 git mv)
- 이후: notebooks/ → docs/ 순서로 파일 번호 재조정

참고 파일:
- SPEC: `_core/PROJECT-SPEC.md`
- TODO: `_core/PROJECT-TODO.md`
- 핸드오프: `_core/sessions/260620-112500_session-handoff.md`
