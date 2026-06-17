---
tags: [project, agent-config]
created: 2026-06-08
updated: 2026-06-18
---

# AGENTS.md - mnist-numpy-from-scratch

이 파일은 Codex가 이 프로젝트를 다룰 때 따르는 진입점 지침이다.
상세 규칙은 `_core/` 를 참조한다.

본 워크스페이스는 하나의 Git Repository 를 하나의 코딩 프로젝트로 간주한다.
AI CLI 는 `src/` 에 코드를 작성·정제하고, `experiments/` 에 실험을 기록하며, `outputs/` 에 결과를 저장한다.

## 1. 프로젝트 개요

- **이름**: Deep Learning from Scratch with Numpy/Cupy
- **목적**: NumPy와 CuPy만으로 MNIST 기반 딥러닝 모델 학습 과정을 구현하고, 이후 PyTorch, TensorFlow, JAX 프로젝트와 동일한 모듈 및 함수 인터페이스를 유지할 수 있는 기준 구현을 작성한다.
- **템플릿**: `project-coding-template`

## 2. 프로젝트 구조

```text
mnist-numpy-from-scratch/
├── _core/        # 프로젝트 운영 파일
├── data/         # 데이터 (경로 또는 설명)
├── src/          # 소스 코드
├── scripts/      # 실행 스크립트
├── notebooks/    # Jupyter 노트북
├── tests/        # 테스트 코드
├── configs/      # 설정 파일
├── experiments/  # 실험 기록
├── docs/         # 프로젝트 문서
└── outputs/      # 결과물 (모델, 그래프, 예측 등)
```

## 3. 폴더 역할

각 폴더의 역할은 다음과 같다.

| 폴더 | 역할 |
|---|---|
| `_core/` | 프로젝트 운영 규칙, 커맨드, 가이드, 세션 |
| `data/` | 데이터 경로 또는 설명 |
| `src/` | 소스 코드 루트 - 재사용 가능한 모듈·클래스·함수 |
| `scripts/` | `src/core/` 실행 객체를 호출하는 CLI 진입점 - 학습·평가·예측·시각화 |
| `notebooks/` | Jupyter 노트북 - 탐색·시각화·프로토타입 |
| `tests/` | 테스트 코드 |
| `configs/` | 설정 파일 |
| `experiments/` | 실험 기록 및 실행 메타데이터 |
| `docs/` | 프로젝트 문서 |
| `outputs/` | 모델, 그래프, 예측, 내보내기 결과 |

## 4. 자료 흐름

자료는 다음 순서로 이동한다.

```text
data/ → notebooks/ → src/ → scripts/ → experiments/ → outputs/
```

각 단계의 역할은 다음과 같다.

| 단계 | 폴더 | 의미 |
|---|---|---|
| Data | `data/` | 원본 데이터 참조 또는 경로 관리 |
| Exploration | `notebooks/` | 탐색·시각화·프로토타입 |
| Implementation | `src/` | 정제된 코드를 모듈로 구조화 |
| Execution | `scripts/` | `src/core/` 실행 객체를 호출하는 CLI 진입점 |
| Experiment | `experiments/` | 실험 실행 기록 및 메타데이터 |
| Output | `outputs/` | 최종 결과물 저장 |

## 5. 핵심 행동 규칙

- 작업 전 `_core/rules/agent-rules.md` 와 `_core/rules/coding-rules.md` 를 참조한다.
- 마크다운 문서 작성·수정 시 `_core/rules/docs-rules.md` 를 따른다.
- Python 스크립트 작성·수정 시 `_core/rules/python-rules.md` 를 따른다.
- 탐색·프로토타입은 `notebooks/` 에서 시작하고, 재사용 코드는 `src/` 로 이동한다.
- `scripts/` 를 통해 `experiments/` 에서 학습·추론·시각화를 실행하고, 결과는 `outputs/` 에 저장한다.
- 새 소스 파일을 만들 때 첫 줄에 파일명과 한 줄 영어 설명을 주석으로 기록한다.
- 루트 폴더 구조는 변경하지 않는다.
- Python 실행 환경은 conda 환경 `numpy_env` 를 사용한다. 모든 Python 명령은 `conda run -n numpy_env` 를 통해 실행한다.

## 6. 참조 경로

| 항목 | 경로 |
|---|---|
| 에이전트 규칙 | `_core/rules/agent-rules.md` |
| 마크다운 규칙 | `_core/rules/docs-rules.md` |
| Python 규칙 | `_core/rules/python-rules.md` |
| 코딩 규칙 | `_core/rules/coding-rules.md` |
| 코딩 가이드 | `_core/docs/coding-guide.md` |
| 주제 분류 | `_core/docs/subject-guide.md` |
| 프로젝트 가이드 | `_core/docs/project-guide.md` |
| 프로젝트 명세 | `_core/docs/project-spec.md` |
| 할일 관리 | `_core/docs/project-todo.md` |
| 작업 이력 | `_core/docs/project-log.md` |
| 세션 시작 | `_core/commands/session-start.md` |
| 세션 종료 | `_core/commands/session-end.md` |
| 세션 핸드오프 | `_core/commands/session-handoff.md` |
| 프로젝트 초기화 | `_core/commands/project-init.md` |
| 프로젝트 상태 | `_core/commands/project-status.md` |
| 프로젝트 업데이트 | `_core/commands/project-update.md` |
| 커밋 메시지 | `_core/commands/commit-message.md` |
