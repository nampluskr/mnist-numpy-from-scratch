---
tags: [docs, project, index]
created: 2026-06-19
updated: 2026-06-20
---

# Deep Learning from Scratch 문서 색인

이 문서는 `mnist-numpy-from-scratch` 프로젝트의 문서 전체 진입점이다.
NumPy와 CuPy만으로 MNIST 기반 딥러닝 모델을 구현하고, 이후 PyTorch, TensorFlow, JAX 프로젝트와 동일한 모듈 및 함수 인터페이스를 유지하는 기준 구현을 목표로 한다.
각 Stage는 독립적인 Chapter로 구성되며, 아래 링크에서 Phase별 상세 문서를 확인할 수 있다.

## 1. Stage 구성

| Stage | 제목 | 내용 | 문서 |
|---|---|---|---|
| Stage 0 | 레거시 코드 분석 및 계획 수립 | 기존 레거시 코드 6+6개 분석, 구현 계획 및 테스트 계획 수립 | [[stage0]] |
| Stage 1 | config 및 task 규약 | 기본 설정, 과제별 target/loss/metric 규약, utility 구현 | [[stage1]] |
| Stage 2 | MNIST DataLoader | 로컬 gz 파일 로딩, Dataset 클래스, DataLoader 구현 | [[stage2]] |
| Stage 3 | nn 모듈 및 모델 구현 | 공통 모듈(activation/loss/metric), MLP, CNN(CuPy/NumPy) 구현 | [[stage3]] |
| Stage 4 | 실행 객체 | optimizer, checkpoint, Trainer, Evaluator, Predictor, Experiment, Visualizer 구현 | [[stage4]] |
| Stage 5 | 클라이언트 코드 | train, evaluate, predict, visualize CLI 스크립트 구현 | [[stage5]] |
| Stage 6 | 문서화 및 검증 | 실험 실행, 과제별 tutorial, framework 연계 준비 | [[stage6]] |

## 2. 참조 문서

프로젝트 목적, 범위, 인터페이스 규약, 진행 단계의 상세 정의는 아래 문서에서 확인한다.

- 프로젝트 명세: `_core/PROJECT-SPEC.md`
- 진행 현황: `_core/PROJECT-TODO.md`
- 작업 이력: `_core/PROJECT-LOG.md`
