---
tags: [project, sessions]
created: 2026-06-20
updated: 2026-06-20
---

# experiments/ 파일명 정리 및 PROJECT-SPEC.md 구조 개선 세션 핸드오프

> 작성일시: 260620-045949
> 세션 목적: experiments/ 파일명 변경 + PROJECT-SPEC.md §3 서브섹션 도입 + Stage 0 환경 구성 편입
> 이전 핸드오프: 260620-043541_session-handoff.md

## 1. 세션 핵심 요약

experiments/ 파일명을 `xxx_all.py` → `run_xxx.py` 형식으로 통일했다.
PROJECT-SPEC.md §3 범위 섹션에 서브섹션(3.1~3.3)과 도입 문장을 추가했다.
conda 환경 구성을 Stage 0으로 편입하여 Phase 0.0을 신설하고, 관련 문서를 stage4에서 stage0으로 이동했다.

## 2. 확정된 변경 사항

### experiments/ 파일명

| 구 파일명 | 신 파일명 |
|---|---|
| `train_all.py` | `run_train.py` |
| `evaluate_all.py` | `run_evaluate.py` |
| `predict_all.py` | `run_predict.py` |
| `visualize_all.py` | `run_visualize.py` |
| `run_all.py` | 유지 (import 경로 갱신) |

### PROJECT-SPEC.md §3 서브섹션

| 섹션 | 내용 |
|---|---|
| 3.1 과제 및 모델 | MNIST 기반 3종 과제, MLP/CNN, 프레임워크 호환성 |
| 3.2 코드 구현 | 레거시 분석, src/scripts/tests 구현 |
| 3.3 문서 및 실험 | docs/notebooks/experiments 산출물 |

### Stage 0 편입

- `Phase 0.0 conda 환경 구성` 신설
- `docs/stage4/phase4.0_cupy-setup.md` → `docs/stage0/phase0.0_conda-setup.md` 이동
- `docs/stage0/stage0.md` 제목·개요·Phase 구성 갱신
- `docs/stage4/stage4.md` Phase 4.0 섹션 제거, Phase 번호 오류(6.x→4.x) 수정
- PROJECT-SPEC.md §5.1 제목 → `Stage 0 환경 구성 및 계획 수립`
- PROJECT-SPEC.md §5.5 Phase 4.0 항목 제거
- PROJECT-TODO.md §1 제목·Phase 0.0 섹션 동기화

## 3. 이번 세션 완료 항목

| 항목 | 내용 |
|---|---|
| experiments/ 파일명 변경 | run_train/evaluate/predict/visualize.py 4개 신규, 구 파일 4개 삭제, run_all.py import 갱신 |
| PROJECT-SPEC.md §3 재구성 | 서브섹션 3개 도입, 각 섹션 도입 문장 추가 |
| PROJECT-SPEC.md §6.3 갱신 | experiments/ 파일 목록 신규 파일명으로 업데이트 |
| Stage 0 구조 변경 | Phase 0.0 신설, 환경 문서 이동, 관련 문서 전체 동기화 |
| PROJECT-LOG.md 갱신 | 세션 작업 내역 5줄 추가 |

## 4. 미결 사항

없음.

## 5. 다음 작업 목록

| 우선순위 | 작업 | 참고 |
|---|---|---|
| 1 | PyTorch 마이그레이션 프로젝트 시작 | `docs/stage7/phase7.5_framework-checklist.md` 체크리스트 참조 |

## 6. 다음 세션 시작 지시문

`session-start 실행` 후 PyTorch 마이그레이션 프로젝트 시작 여부를 확인해 주세요.

참고 파일:
- 핸드오프: `_core/sessions/260620-045949_session-handoff.md`
- 프레임워크 체크리스트: `docs/stage7/phase7.5_framework-checklist.md`
