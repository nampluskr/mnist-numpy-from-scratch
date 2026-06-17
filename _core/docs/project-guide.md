---
tags: [project, docs]
created: 2026-06-08
updated: 2026-06-15
---

# project-guide.md

이 프로젝트의 구조와 목적을 설명한다.
사용자가 참조하는 문서이다. 에이전트는 수정하지 않는다.

## 1. 프로젝트 개요

- **이름**: Deep Learning from Scratch with Numpy/Cupy
- **목적**: NumPy와 CuPy만으로 MNIST 기반 딥러닝 모델 학습 과정을 구현하고, 이후 PyTorch, TensorFlow, JAX 프로젝트와 동일한 모듈 및 함수 인터페이스를 유지할 수 있는 기준 구현을 작성한다.
- **템플릿**: `project-coding-template`

## 2. 폴더 구조

각 폴더의 역할은 다음과 같다.

| 폴더 | 역할 |
|---|---|
| `_core/` | 프로젝트 운영 규칙, 커맨드, 가이드, 세션 |
| `data/` | 데이터 경로 또는 설명 |
| `src/` | 소스 코드 루트 |
| `scripts/` | `src/core/` 실행 객체를 호출하는 CLI 진입점 - 학습·평가·예측·시각화 |
| `notebooks/` | Jupyter 노트북 |
| `tests/` | 테스트 코드 |
| `configs/` | 설정 파일 |
| `experiments/` | 실험 기록 및 실행 메타데이터 |
| `docs/` | 프로젝트 문서 |
| `outputs/` | 모델, 그래프, 예측, 내보내기 결과 |

## 3. 최종 결과물

이 프로젝트의 최종 결과물은 세 가지로 구성된다.

| 결과물 | 폴더 | 내용 |
|---|---|---|
| 소스 코드 | `src/` | NumPy MLP, CuPy CNN 및 실행 객체 from-scratch 구현 |
| 클라이언트 코드 | `scripts/` | 학습·평가·예측·시각화 CLI 진입점 |
| 문서 | `docs/` | 튜토리얼, 실행 예제, 후속 프레임워크 연계 기준 |

## 4. _core/ 구조

```text
_core/
├── rules/
│   ├── agent-rules.md
│   ├── docs-rules.md
│   ├── python-rules.md
│   └── coding-rules.md    # coding-template 전용
├── commands/
│   ├── session-start.md
│   ├── session-end.md
│   ├── session-handoff.md
│   ├── project-init.md
│   ├── project-status.md
│   ├── project-update.md
│   └── commit-message.md
├── docs/
│   ├── project-guide.md   # 이 파일
│   ├── project-spec.md
│   ├── project-todo.md
│   ├── project-log.md
│   ├── subject-guide.md
│   └── coding-guide.md    # coding-template 전용
└── sessions/
```
