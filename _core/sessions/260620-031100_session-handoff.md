---
tags: [project, sessions]
created: 2026-06-20
updated: 2026-06-20
---

# Stage 순서 재배치 및 문서 정합화 세션 핸드오프

> 작성일시: 260620-031100
> 세션 목적: Stage 6(CuPy CNN)을 Stage 4로 재배치하고 전체 순서 변경, PROJECT-TODO.md 도입 문장 형식 통일
> 이전 핸드오프: 260619-144056_session-handoff.md

## 1. 세션 핵심 요약

Stage 순서를 `4(실행 객체) → 5(CLI) → 6(CuPy CNN)`에서 `4(CuPy CNN) → 5(실행 객체) → 6(CLI)`로 전면 재배치했다.
docs/, tests/, notebooks/ 폴더명·파일명·내부 텍스트를 모두 갱신하고, PROJECT-SPEC.md와 PROJECT-TODO.md를 동기화했다.
이후 PROJECT-TODO.md의 모든 Phase 도입 문장을 `~하고, ~를 검증한다.` / `~를 작성한다.` 형식으로 통일했다.

## 2. 이번 세션 완료 항목

| 항목 | 내용 |
|---|---|
| Stage 순서 재배치 | Stage 6 → 4, Stage 4 → 5, Stage 5 → 6 (임시 폴더 경유) |
| docs/ 폴더·파일 재구성 | stage4/5/6 폴더명, phase*.md 파일명, 내부 H1 제목·tags·updated 수정 |
| tests/ 폴더 재구성 | stage4/5/6 폴더명 변경 (파일명은 기능 그대로 유지) |
| notebooks/ 폴더·파일 재구성 | stage4/5/6 폴더명, stageN-*.ipynb 파일명 변경 |
| docs/stage7/ 파일명 변경 | phase7.2~7.6 → phase7.1~7.5 (gap 제거) |
| PROJECT-SPEC.md 수정 | §5.5~5.8 헤딩·Phase 목록, §6.4 tests 구조, §6.7 pytest 표, §6.8 notebooks 구조 |
| PROJECT-TODO.md 수정 | 섹션 헤딩 전체, [[링크]] 37개, Phase 도입 문장 11개 |

## 3. 미결 사항

없음.

## 4. 새 Stage 번호 체계

| Stage | 내용 | 주요 파일 |
|---|---|---|
| Stage 0 | 계획 수립 | 변경 없음 |
| Stage 1 | 기본 설정 및 과제 규약 | 변경 없음 |
| Stage 2 | MNIST 데이터 로더 | 변경 없음 |
| Stage 3 | NumPy nn 모듈 및 MLP | 변경 없음 |
| Stage 4 | CuPy 기반 CNN 구현 | docs/stage4/, tests/stage4/, notebooks/stage4/ |
| Stage 5 | 실행 객체 구현 | docs/stage5/, tests/stage5/, notebooks/stage5/ |
| Stage 6 | 클라이언트 코드 구현 | docs/stage6/, tests/stage6/, notebooks/stage6/ |
| Stage 7 | documentation 및 verification | Phase 7.1~7.6 (구 7.2~7.7) |

## 5. 다음 작업 목록

모든 TODO 항목은 완료 상태다. 후속 작업이 필요하다면 PyTorch 마이그레이션 프로젝트를 시작할 수 있다.

| 우선순위 | 작업 | 참고 |
|---|---|---|
| 1 | PyTorch 프로젝트 시작 | `docs/stage7/phase7.5_framework-checklist.md` 체크리스트 참조 |

## 6. 다음 세션 시작 지시문

`session-start 실행` 후 PyTorch 마이그레이션 프로젝트 시작 여부를 확인해 주세요.

참고 파일:
- 핸드오프: `_core/sessions/260620-031100_session-handoff.md`
- 프레임워크 체크리스트: `docs/stage7/phase7.5_framework-checklist.md`
