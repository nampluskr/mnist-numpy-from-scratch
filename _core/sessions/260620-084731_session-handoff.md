---
tags: [session, handoff]
created: 2026-06-20
---

# 세션 핸드오프 - 260620-084731

## 1. 세션 요약

이번 세션에서는 `_core/templates/` 두 파일을 전면 재설계하고, 관련 규칙 파일과 지침 파일을 정비했다.
코드 변경은 없었고 문서/템플릿/규칙 파일만 수정했다.

## 2. 완료한 작업

### 2.1. 템플릿 재설계

기존 템플릿(11섹션, 9섹션)이 지나치게 많고 실제 작성된 문서가 핸드아웃 수준으로 간결했다는 문제를 해결했다.
목표: 초보자가 처음 읽어도 이해할 수 있는 책/텍스트 수준의 상세 설명.

**`book-section-template.md` -> `docs-template.md`** (11섹션 -> 6섹션):

| 섹션 | 내용 |
|---|---|
| 1. 개요 | 책임 서술 2~4문장 + **목표** 블록 (서브섹션 없음) |
| 2. 개념 | 수식($...$) + 용어 + 원리를 코드 없이 초보자 수준 상세 설명 |
| 3. 구현 | API 표 + 코드 해설 (H3 자유 분화) |
| 4. 사용법 | 최소 예제 + 통합 예제 |
| 5. 테스트 | 실행 명령 + 검증 표 |
| 6. 요약 | 마무리 + 다음 Phase 링크 |

**`tutorial-notebook-template.ipynb` -> `notebooks-template.ipynb`** (9섹션 -> 5그룹):

| 위치 | 내용 |
|---|---|
| Intro cell | H1 제목 + 설명 + **목표** 블록 |
| 0. 환경 설정 | 3셀 고정 구조 |
| 1. 개요 | 개념 요약 + 표/수식 |
| 2~N. 주제별 실습 | 자유 구성 |
| 요약 | 정리 표 + 설계 원칙 |

환경 설정 3셀 구조:
- 셀 1: `os`, `sys` import + `sys.path` 설정 (`remove-cell` 태그)
- 셀 2: 외부 라이브러리 import
- 셀 3: `src/` 사용자 구현 import

`pathlib.Path` 전면 제거 -> `os.path` 로 교체 (python-rules.md 준수).

### 2.2. 규칙 파일 추가

`_core/templates/README.md` 를 `_core/rules/template-rules.md` 로 이동·재작성.
새 템플릿 구조(6섹션, 3셀 환경설정 등) 기준으로 전면 업데이트.
다른 rules 파일(`coding-rules.md` 등)과 동일한 형식(frontmatter + 번호 섹션 + 서술체).

### 2.3. 지침 파일 업데이트

`CLAUDE.md`, `AGENTS.md` 두 파일 동기화:
- 섹션 5(핵심 행동 규칙): docs/notebooks 작성 시 템플릿 + template-rules.md 적용 지침 추가
- 섹션 6(참조 경로): `템플릿 규칙 | _core/rules/template-rules.md` 행 추가

## 3. 현재 상태

- 브랜치: `refactor/book-notebook-restructure`
- 커밋: `14d81b6` refactor: 템플릿 전면 재설계 및 규칙 파일 추가 (260620)
- 푸시: 완료 (GitHub 원격 브랜치 신규 생성)

## 4. 다음 세션 최우선 작업

### 우선 1: PROJECT-SPEC.md Stage-Phase 재검토

`_core/PROJECT-SPEC.md` 의 `## 5. 진행 단계` 와 `## 6. 확정 구조` 를
핸드오프(260620-080229)에서 확정한 새 Stage 0~7 구성으로 전면 재작성.

참조: `_core/legacy/refs/PROJECT-TODO.md` (구 형식 참고용)

새 Stage 구성 (핸드오프 260620-080229 기준):

| Stage | 제목 | 주요 파일 |
|---|---|---|
| Stage 0 | 환경 구성 및 계획 수립 | conda 환경, 레거시 분석, 구현/테스트 계획 |
| Stage 1 | Utilities | `batching.py`, `random.py`, `io.py`, `training_plots.py` |
| Stage 2 | Data | `mnist.py`, `dataloader.py` |
| Stage 3 | nn modules | `activations.py`, `losses.py`, `metrics.py`, `layers.py`, `conv.py` |
| Stage 4 | Models | `mlp.py`, `cnn.py` |
| Stage 5 | Core | `optimizers.py`, `checkpoints.py`, `trainer.py`, `evaluator.py`, `predictor.py`, `visualizer.py`, `logger.py` |
| Stage 6 | Scripts | `train.py`, `evaluate.py`, `predict.py`, `visualize.py` |
| Stage 7 | Experiments & Outputs | `run_*.py` -> `outputs/` |

### 우선 2: PROJECT-TODO.md 재편

새 Stage 0~7 구성으로 전면 재작성.

### 우선 3: src/ 코드 변경 (핸드오프 260620-080229 기준)

1. `src/config.py`, `src/task.py`, `src/core/experiment.py` 삭제
2. 영향받는 파일 수정 ( MNISTDataset, MLP, CNN, Trainer, Evaluator, Predictor)
3. `src/core/logger.py` 신규 구현
4. `scripts/*.py` 직접 조립 패턴으로 재작성
5. `experiments/run_*.py` CONFIGS에 task 규약 key 추가

### 우선 4: 파일 이동 (git mv)

tests/, docs/, notebooks/ 이동 및 삭제 (핸드오프 260620-080229 섹션 3.3 참고)

### 우선 5: pytest 전체 통과 확인

```bash
conda run -n numpy_py311 pytest tests/ -q
```

### 우선 6: docs/ 전면 재작성

새 `docs-template.md` 기준으로 Stage 1부터 순서대로 재작성.

### 우선 7: notebooks/ 전면 재작성

새 `notebooks-template.ipynb` 기준으로 Stage 1부터 순서대로 재작성.

## 5. 다음 세션 시작 지시문

"session-start 실행 후 이번 세션 핸드오프를 기준으로 작업을 진행해 주세요. PROJECT-SPEC.md와 PROJECT-TODO.md를 새 Stage 0~7 구성으로 재작성하는 것부터 시작해 주세요."
