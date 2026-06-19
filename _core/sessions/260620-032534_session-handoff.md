---
tags: [project, sessions]
created: 2026-06-20
updated: 2026-06-20
---

# PROJECT-TODO.md task 기술 형식 통일 세션 핸드오프

> 작성일시: 260620-032534
> 세션 목적: PROJECT-TODO.md task 수준(체크박스 항목) 기술 형식 통일
> 이전 핸드오프: 260620-031100_session-handoff.md

## 1. 세션 핵심 요약

PROJECT-TODO.md의 모든 체크박스 항목에 동사를 추가하여 기술 형식을 통일했다.
파일 항목은 `src/` → `구현`, `tests/` → `작성`, `scripts/` → `구현`으로 통일하고,
괄호 설명이 있는 항목은 동사를 괄호 앞에 삽입했다.
outputs/ 항목 6개에는 `생성` 동사를 추가했다.

## 2. 확정된 task 기술 형식 규칙

| 항목 유형 | 형식 | 예시 |
|---|---|---|
| src/ 파일 (괄호 없음) | `` `경로` 구현 `` | `` `src/config.py` 구현 `` |
| src/ 파일 (괄호 있음) | `` `경로` 구현 (`설명`) `` | `` `src/nn/layers.py` 구현 (`Module` training/train/eval 추가) `` |
| tests/ 파일 (괄호 없음) | `` `경로` 작성 `` | `` `tests/stage1/test_config.py` 작성 `` |
| tests/ 파일 (괄호 있음) | `` `경로` 작성 (`설명`) `` | `` `tests/stage4/test_cnn.py` 작성 (CNN 통합 케이스 추가) `` |
| scripts/ 파일 (괄호 없음) | `` `경로` 구현 `` | `` `scripts/train.py` 구현 `` |
| scripts/ 파일 (괄호 있음) | `` `경로` 구현 (`설명`) `` | `` `scripts/train.py` 구현 (`--model` 플래그 추가, ...) `` |
| notebooks/ 파일 | `` `경로` 작성 `` | `` `notebooks/stage1/stage1-1_config-and-task.ipynb` 작성 `` |
| 문서 링크 | `[[링크]] 문서 작성` | `[[docs/stage0/phase0.1_legacy-analysis\|...]] 문서 작성` |
| outputs/ 결과물 | `` `경로` (`파일목록`) 생성 `` | `` `outputs/multiclass/mlp/` (`training_log.png`, ...) 생성 `` |
| 비파일 작업 | 행위에 맞는 동사 유지 | `확인`, `확정`, `분석`, `도출`, `정리`, `파악` 등 |

## 3. 이번 세션 완료 항목

| 항목 | 내용 |
|---|---|
| PROJECT-TODO.md task 형식 통일 | 약 64개 항목 수정 (src/ 25개, tests/ 24개, scripts/ 8개, requirements.txt 1개, outputs/ 6개) |
| PROJECT-LOG.md 갱신 | 세션 작업 내역 1줄 추가 |

## 4. 미결 사항

없음.

## 5. 다음 작업 목록

| 우선순위 | 작업 | 참고 |
|---|---|---|
| 1 | PyTorch 마이그레이션 프로젝트 시작 | `docs/stage7/phase7.5_framework-checklist.md` 체크리스트 참조 |

## 6. 다음 세션 시작 지시문

`session-start 실행` 후 PyTorch 마이그레이션 프로젝트 시작 여부를 확인해 주세요.

참고 파일:
- 핸드오프: `_core/sessions/260620-032534_session-handoff.md`
- 프레임워크 체크리스트: `docs/stage7/phase7.5_framework-checklist.md`
