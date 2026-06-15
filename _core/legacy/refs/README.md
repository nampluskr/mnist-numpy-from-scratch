# Deep Learning from Scratch with Numpy/Cupy 소개

딥러닝을 단계별로 학습하기 위한 NumPy/CuPy 기반 튜토리얼 자료이다.

> 생성일시: 260608-133040
> 수정일시: 260608-140844
> 주제: Machine Learning

**목차**

1. [개요](#1-개요)
2. [내용 및 방법](#2-내용-및-방법)
3. [디렉토리 구조](#3-디렉토리-구조)
4. [결과 및 의의](#4-결과-및-의의)
5. [한계 및 미진사항](#5-한계-및-미진사항)

## 1. 개요

이 프로젝트는 `numpy -> pytorch -> tensorflow -> jax` 순서로 진행되는 딥러닝 프레임워크 학습 시리즈의 첫 번째 프로젝트이다. MNIST 데이터셋을 대상으로 Multiclass Classification, Binary Classification, Regression 과제를 수행하며, NumPy 기반 MLP와 CuPy 기반 CNN을 직접 구현한다.

## 2. 내용 및 방법

프로젝트의 핵심 구현 방향은 프레임워크 간 비교 학습을 위한 동일 인터페이스와 실행 흐름을 유지하는 것이다.

- MLP 모델은 CPU 기반 NumPy 구현으로 학습한다.
- CNN 모델은 GPU 기반 CuPy 구현으로 학습한다.
- `src/` 폴더에 데이터, task, 모델, 실행 객체, 유틸리티 모듈을 구현한다.
- `scripts/` 폴더에 `train.py`, `evaluate.py`, `predict.py`, `visualize.py` 클라이언트 코드를 작성한다.
- `tests/` 폴더 기반으로 코드 파일과 대응 테스트 파일을 함께 작성하는 TDD 흐름을 적용한다.
- `legacy/` 폴더에 사용자가 제공하는 기존 3가지 작업의 레거시 코드를 보관하고 기준 구현으로 재구성한다.

## 3. 디렉토리 구조

프로젝트 루트의 주요 폴더는 입력 자료, 구현 코드, 테스트, 산출물, 최종 문서를 분리하도록 구성한다.

```
project-root/
├── README.md                   # 프로젝트 외부 소개
├── CLAUDE.md                   # Claude Code 용 AI CLI 운영 지침
├── AGENTS.md                   # Codex 용 AI CLI 운영 지침
│
├── _project/                   # 프로젝트 운영 관리
│   ├── PROJECT.md              # 프로젝트 내부 명세
│   ├── PROJECT-TODO.md         # Stage-Phase-Task 진행 관리
│   ├── PROJECT-HISTORY.md      # 완료 Task 상세 기록
│   ├── commands/               # 커스텀 명령어 프롬프트
│   ├── docs/                   # 운영 문서
│   └── rules/                  # 공통 규칙
├── _assets/                    # 바이너리 읽기 전용 입력 보관소
│
├── legacy/                     # 사용자가 제공하는 레거시 코드
├── src/                        # 프로젝트 소스 코드
│   ├── config.py               # 기본 경로와 실행 설정
│   ├── data/                   # MNIST 데이터 로더
│   ├── task.py                 # task별 target, loss, metric 규약
│   ├── models/                 # MLP, CNN 모델
│   ├── core/                   # scripts 에서 참조하는 실행 객체
│   └── utils/                  # 공통 보조 기능
├── scripts/                    # 실행 클라이언트 코드
├── tests/                      # pytest 기반 대응 테스트 코드
├── outputs/                    # 프로젝트 중간/최종 결과 저장
└── docs/                       # Jupyter Book v2 최종 산출물
    ├── myst.yml
    └── contents/
```

## 4. 결과 및 의의

최종 산출물은 NumPy/CuPy만으로 구성한 딥러닝 기준 구현, 실행 가능한 클라이언트 코드, 테스트 코드, Jupyter Book v2 튜토리얼 문서이다. 이 프로젝트는 후속 PyTorch, TensorFlow, JAX 프로젝트에서 동일한 모듈명과 함수 사용법을 유지하기 위한 기준 구조로 활용된다.

## 5. 한계 및 미진사항

현재 한계는 실제 코드 구현이 아직 시작되지 않았고, 레거시 코드의 세부 흐름을 확정된 파일·테스트 단위 구현 계획에 매핑해야 한다는 점이다. Stage 1 이후에는 코드 파일과 대응 테스트 파일을 함께 작성하는 방식으로 진행한다.
