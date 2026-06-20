---
tags: [session, handoff]
created: 2026-06-20
---

# Stage 5·6 Phase 구조 재편 세션 핸드오프

> 작성일시: 260620-110844
> 세션 목적: Stage 5·6 Phase 구조 재편 및 Phase 명칭 정비
> 이전 핸드오프: 260620-103203_session-handoff.md

## 1. 세션 핵심 요약

Stage 5 Phase를 7개에서 5개로, Stage 6 Phase를 6개에서 4개로 통합 재편했다.
Phase 문서 분량 기준(코드 구현·테스트·문서 템플릿 6섹션)으로 적정성을 검토하고
단독으로 문서 내용이 빈약한 Phase들을 쌍으로 묶었다.
Phase 제목에서 "·" 특수문자를 "및"으로 교체하고, CLI → 스크립트, 영문 키워드 → 한글로 정비했다.

## 2. 확정된 결정사항

| 항목 | 확정 내용 | 비고 |
|---|---|---|
| Stage 5 Phase | 5개 (5.1~5.5) | Trainer+Evaluator, Predictor+Visualizer 통합 |
| Stage 6 Phase | 4개 (6.1~6.4) | 학습+평가, 예측+시각화 통합 |
| src/core/logger.py | Phase 5.4 신설 | epoch별 loss/metric CSV/dict 기록 |
| Phase 6.3 | 실험 배치 스크립트 작성 | experiments/run_all.py |
| Phase 제목 규칙 | "·" 금지, 한글 우선 | docs-rules.md 반영 |

### 확정된 Stage 5 Phase 목록

- Phase 5.1 optimizer 구현
- Phase 5.2 Trainer 및 Evaluator 구현
- Phase 5.3 Predictor 및 Visualizer 구현
- Phase 5.4 Logger 구현
- Phase 5.5 실습 노트북 작성

### 확정된 Stage 6 Phase 목록

- Phase 6.1 학습 및 평가 스크립트 작성
- Phase 6.2 예측 및 시각화 스크립트 작성
- Phase 6.3 실험 배치 스크립트 작성
- Phase 6.4 실습 노트북 작성

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

- Stage 5·6 Phase 구조 및 명칭 확정 완료 (SPEC, TODO 동기화 완료)
- 다음 작업: src/ 코드 변경 (config.py, task.py, experiment.py 삭제, checkpoints.py 이동)
- 이후: tests/ → notebooks/ → docs/ 순서로 파일 번호 재조정

참고 파일:
- SPEC: `_core/PROJECT-SPEC.md`
- TODO: `_core/PROJECT-TODO.md`
- 핸드오프: `_core/sessions/260620-110844_session-handoff.md`
