---
tags: [session, handoff]
created: 2026-06-20
---

# 세션 핸드오프 - 260620-080229

## 1. 세션 요약

이번 세션에서는 코드 작성 없이 계획만 수립했다.
브랜치 `refactor/book-notebook-restructure`를 기반으로 프로젝트 전면 리팩토링 방향을 확정했다.

## 2. 프로젝트 목표 (재확인)

이 프로젝트는 NumPy 사용자를 대상으로 딥러닝 구성요소를 프레임워크별로 순차 학습하는 시리즈의
첫 번째 프로젝트이다. 다음 두 결과물을 동시에 만족해야 한다.

- `docs/` - 책 수준의 상세한 설명 문서. API reference, 구현 해설, 설계 결정 이유, 초보자 관점 설명 포함
- `notebooks/` - 책의 실습 섹션 역할. 동시에 노트북 단독으로도 독립적인 튜토리얼 및 강의자료로 활용 가능

## 3. 확정된 계획

### 3.1. 아키텍처 단순화 (src/ 변경)

제거 대상 3개 파일:

| 파일 | 이유 |
|---|---|
| `src/config.py` | `get_default_config()`는 scripts/에서만 필요. CONFIGS dict로 대체 |
| `src/task.py` | `get_task_spec()` / `transform_targets()`를 각 클래스 내부 직접 처리로 대체 |
| `src/core/experiment.py` | `Experiment` 클래스 제거. scripts/*.py가 직접 조립 |

신규 추가:

| 파일 | 내용 |
|---|---|
| `src/core/logger.py` | epoch별 loss/metric을 수집하고 출력하는 Logger 클래스 |

인터페이스 변경 방향:

- ` MNISTDataset`, `MLP`, `CNN` - `get_task_spec` import 제거, task 문자열로 내부 분기
- `Trainer`, `Evaluator`, `Predictor` - `task_spec` dict 대신 `task` 문자열 수신
- `experiments/run_*.py` - `_CONFIGS` 리스트에 task 규약 key(output_dim, loss, metric, prediction_mode)까지 포함
- `scripts/*.py` - `Experiment` 없이 dataset, dataloader, model, optimizer, trainer, evaluator 직접 조립

### 3.2. 새 Stage 구성 (Stage 0~7, 8단계)

| Stage | 제목 | 구현 파일 |
|---|---|---|
| Stage 0 | Environment Setup | conda 환경, 레거시 분석, 구현/테스트 계획 |
| Stage 1 | Utilities | `src/utils/batching.py`, `random.py`, `io.py`, `training_plots.py` |
| Stage 2 | Data | `src/data/mnist.py`, `dataloader.py` |
| Stage 3 | nn modules | `src/nn/activations.py`, `losses.py`, `metrics.py`, `layers.py`, `conv.py` |
| Stage 4 | Models | `src/models/mlp.py`, `cnn.py` |
| Stage 5 | Core | `src/core/optimizers.py`, `checkpoints.py`, `trainer.py`, `evaluator.py`, `predictor.py`, `visualizer.py`, `logger.py` |
| Stage 6 | Scripts | `scripts/train.py`, `evaluate.py`, `predict.py`, `visualize.py` |
| Stage 7 | Experiments & Outputs | `experiments/run_*.py` -> `outputs/` |

Phase 구성:

| Stage | Phase 목록 |
|---|---|
| Stage 1 | 1.1 batching / 1.2 random / 1.3 io / 1.4 training_plots |
| Stage 2 | 2.1 mnist / 2.2 dataset / 2.3 dataloader |
| Stage 3 | 3.1 activations / 3.2 losses / 3.3 metrics / 3.4 layers / 3.5 conv |
| Stage 4 | 4.1 mlp / 4.2 cnn |
| Stage 5 | 5.1 optimizers / 5.2 checkpoints / 5.3 trainer / 5.4 evaluator / 5.5 predictor / 5.6 visualizer / 5.7 logger |
| Stage 6 | 6.1 train / 6.2 evaluate / 6.3 predict / 6.4 visualize |
| Stage 7 | 7.1 experiments / 7.2 outputs 검증 / 7.3 framework 연계 |

### 3.3. 파일 이동 방향 (git mv)

tests/ 이동:

- `tests/stage3/test_mlp.py`, `test_cnn.py` -> `tests/stage4/`
- `tests/stage3/test_experiment.py`, `tests/stage4/test_experiment.py` - 삭제 (Experiment 제거)
- `tests/stage4/` (optimizers~visualizer) -> `tests/stage5/`
- `tests/stage5/` (train~visualize) -> `tests/stage6/`
- `tests/stage1/test_config.py`, `test_task.py` - 삭제

docs/ 이동:

- `docs/stage3/phase3.5_mlp.md`, `phase3.7_cnn.md` -> `docs/stage4/`
- `docs/stage4/` -> `docs/stage5/`
- `docs/stage5/` -> `docs/stage6/`
- `docs/stage6/` -> `docs/stage7/`
- `docs/stage1/phase1.1_config.md`, `phase1.2_task.md` - 삭제

notebooks/ 이동:

- `notebooks/stage3/stage3-4_mlp.ipynb`, `stage3-5_cnn-architecture.ipynb`, `stage3-6_cnn-training.ipynb` -> `notebooks/stage4/`
- `notebooks/stage4/` -> `notebooks/stage5/`
- `notebooks/stage5/` -> `notebooks/stage6/`
- `notebooks/stage6/` -> `notebooks/stage7/`

## 4. 다음 세션 최우선 작업 순서

### 우선 1: 템플릿 업데이트 (작업 시작 전 필수)

현재 `_core/templates/`에 두 템플릿이 있다. 다음 세션 시작 시 **가장 먼저** 이 템플릿을 검토하고 업데이트한다.

**`_core/templates/book-section-template.md` 검토 기준:**

- 11개 섹션 구조가 이 프로젝트의 Stage별 성격에 맞는지 확인
- 각 Stage 유형(utils 같은 단순 함수 / nn 같은 수학 구현 / core 같은 클래스 조립)에 따라 섹션 가중치 조정 필요 여부 검토
- `_core/rules/docs-rules.md` 규칙 전부 반영 확인 (서술체, 특수문자 금지, 헤더 규칙 등)

**`_core/templates/tutorial-notebook-template.ipynb` 검토 기준:**

- `_core/rules/python-rules.md` 준수 여부 확인
- 독립 실행 가능성 보장 구조 확인
- 노트북 전용 규칙이 필요하다면 `_core/rules/notebook-rules.md` 신규 작성
- 각 실습 단위 구조: Markdown(개념/목적/예상결과) -> Code(계산/API 호출) -> Code(graph/table) -> Markdown(결과 해석/설계 이유) -> Assertion(검증) 패턴 반영

### 우선 2: src/ 코드 변경

1. `src/config.py`, `src/task.py`, `src/core/experiment.py` 삭제
2. 영향받는 파일 수정 ( MNISTDataset, MLP, CNN, Trainer, Evaluator, Predictor)
3. `src/core/logger.py` 신규 구현
4. `scripts/*.py` 직접 조립 패턴으로 재작성
5. `experiments/run_*.py` CONFIGS에 task 규약 key 추가

### 우선 3: 파일 이동 (git mv)

tests/, docs/, notebooks/ 이동 및 삭제 (위 3.3 참고)

### 우선 4: pytest 전체 통과 확인

```bash
conda run -n numpy_py311 pytest tests/ -q
```

### 우선 5: PROJECT-SPEC.md / PROJECT-TODO.md 재편

새 Stage 0~7 구성으로 전면 재작성.

### 우선 6: docs/ 전면 재작성

업데이트된 템플릿 기준으로 Stage 1부터 순서대로 재작성.

### 우선 7: notebooks/ 전면 재작성

업데이트된 템플릿 기준으로 Stage 1부터 순서대로 재작성.

## 5. 현재 상태

- 브랜치: `refactor/book-notebook-restructure`
- 워킹 트리: clean (이번 세션에서 코드/파일 변경 없음)
- 최신 커밋: `d572d6c` chore: 세션 핸드오프 문서 추가 (260620)

## 6. 다음 세션 시작 지시문

"session-start 실행 후 이번 세션 핸드오프를 기준으로 작업을 진행해 주세요. 가장 먼저 `_core/templates/` 두 파일을 검토하고 필요시 업데이트한 뒤, 작업 순서대로 진행해 주세요."
