---
tags: [docs, stage6, overview]
created: 2026-06-19
updated: 2026-06-19
---

# Stage 6 CuPy CNN

## 1. 개요

Stage 6은 GPU 기반 CuPy 환경을 구성하고, im2col/col2im 기법을 활용한 CNN 모델을 구현하여 Stage 4의 실행 객체와 통합하는 단계이다.
MLP와 동일한 `Module` 인터페이스를 따르므로 Trainer, Evaluator, Predictor를 수정하지 않고 CNN을 그대로 사용할 수 있다.
`src/core/experiment.py`의 `config["model"]` 분기를 통해 MLP와 CNN 중 하나를 선택한다.

## 2. Phase 구성

### 2.1. Phase 6.0 CuPy environment 구성

`numpy_py311`, `cupy_py311_cuda118`, `cupy_py311_cuda121` 세 가지 conda 환경을 구성하고 검증한다.
MLP는 CPU 기반 `numpy_py311`에서, CNN은 GPU 기반 `cupy_py311_cuda118` 또는 `cupy_py311_cuda121`에서 실행한다.
환경별 CuPy 버전과 CUDA 드라이버 호환성을 확인하고 `requirements.txt`에 반영한다.

- [[phase6.0_cupy-setup|Phase 6.0 CuPy environment 구성]]

### 2.2. Phase 6.1 CNN model 구현

`src/nn/conv.py`에 `im2col`, `col2im` 변환 함수와 `Conv2d`, `MaxPool2d`, `Flatten`, `Dropout` 레이어를 구현한다.
`src/models/cnn.py`에 CuPy 기반 CNN 클래스를 구현하며, CuPy가 없는 환경에서는 NumPy로 fallback한다.
`src/nn/layers.py`의 `Module` 클래스에 `training` 플래그와 `train()`, `eval()` 메서드를 추가하여 Dropout 동작을 제어한다.

- [[phase6.1_cnn|Phase 6.1 CNN model 구현]]

### 2.3. Phase 6.2 CNN-core integration 검증

`src/core/experiment.py`에 `config["model"]` 값이 `"cnn"`일 때 CNN을 선택하는 분기를 추가한다.
synthetic MNIST gz 기반 통합 테스트로 MLP와 CNN이 동일한 Experiment 인터페이스에서 동작함을 검증한다.

- [[phase6.2_cnn-integration|Phase 6.2 CNN-core integration 검증]]

## 3. 주요 산출물

| 산출물 | 내용 |
|---|---|
| `src/nn/conv.py` | im2col/col2im + Conv2d, MaxPool2d, Flatten, Dropout |
| `src/models/cnn.py` | CuPy 기반 CNN 클래스 (NumPy fallback 지원) |
| `src/nn/layers.py` | Module training/train/eval 추가 |
| `src/core/experiment.py` | config["model"] CNN 분기 추가 |
| `tests/stage6/` | cnn, experiment 테스트 파일 2개 (73개 테스트 통과) |
