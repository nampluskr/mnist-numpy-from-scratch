---
tags: [docs, stage3, overview]
created: 2026-06-19
updated: 2026-06-20
---

# Stage 3 nn 모듈 및 모델 구현

## 1. 개요

Stage 3은 `src/nn/` 패키지 전체와 MLP/CNN 모델을 함께 구현하는 단계이다.
공통 수학 함수(activation, loss, metric) → MLP(layer + model) → CNN(conv layer + model) 순으로 점진적으로 구현한다.
NumPy 기반 MLP와 CuPy/NumPy 양용 CNN이 동일한 `Module` 인터페이스를 공유하므로, Stage 4의 실행 객체(Trainer/Evaluator/Predictor)가 수정 없이 두 모델을 모두 사용할 수 있다.

## 2. Phase 구성

### 2.1. Phase 3.1 공통 모듈 구현

`src/nn/activations.py`, `src/nn/losses.py`, `src/nn/metrics.py` 세 파일에 공통 수학 함수를 구현한다.
activation 함수는 순전파 계산만 담당하며, 손실 함수는 raw logit을 입력으로 받아 activation을 내부에서 처리한다.

- [[phase3.1_activations|Phase 3.1 activations]]
- [[phase3.2_losses|Phase 3.2 losses]]
- [[phase3.3_metrics|Phase 3.3 metrics]]

### 2.2. Phase 3.2 공통 모듈 문서

공통 모듈 3개에 대한 참조 문서를 작성한다.

### 2.3. Phase 3.3 MLP 구현

`src/nn/layers.py`에 `Module` 기반 클래스와 `Linear`, `Sigmoid`, `ReLU`, `Sequential` 레이어를 구현한다.
`Module` 클래스에는 `training` 플래그와 `train()`, `eval()` 메서드가 포함된다.
`src/models/mlp.py`에서 `Sequential`로 MLP를 조립하고, `forward(x)`는 raw logit을 반환한다.

- [[phase3.4_layers|Phase 3.4 layers]]
- [[phase3.5_mlp|Phase 3.5 MLP model]]

### 2.4. Phase 3.4 MLP 문서

MLP 관련 레이어 모듈과 모델에 대한 참조 문서를 작성한다.

### 2.5. Phase 3.5 CNN 구현

`src/nn/conv.py`에 `im2col`/`col2im` 변환 함수와 `Conv2d`, `MaxPool2d`, `Flatten`, `Dropout` 레이어를 구현한다.
`src/models/cnn.py`에 CuPy 기반 CNN 클래스를 구현하며, CuPy가 없는 환경에서는 NumPy로 fallback한다.
`src/core/experiment.py`에 `config["model"]` 분기를 추가하여 MLP/CNN 선택을 지원한다.

- [[phase3.6_conv|Phase 3.6 conv]]
- [[phase3.7_cnn|Phase 3.7 CNN model]]

### 2.6. Phase 3.6 CNN 문서

CNN 레이어와 모델에 대한 참조 문서를 작성한다.

### 2.7. Phase 3.7 Stage 3 노트북 작성

nn 모듈과 MLP, CNN을 실습하는 교육용 노트북 6개를 작성한다.

## 3. 주요 산출물

| 산출물 | 내용 |
|---|---|
| `src/nn/activations.py` | sigmoid, softmax, identity, relu 함수 |
| `src/nn/losses.py` | cross_entropy, binary_cross_entropy, mse 및 *_grad 함수 |
| `src/nn/metrics.py` | accuracy, binary_accuracy, r2_score 함수 |
| `src/nn/layers.py` | Module, Linear, Sigmoid, ReLU, Sequential 클래스 (training/train/eval 포함) |
| `src/models/mlp.py` | Sequential 기반 MLP 클래스 |
| `src/nn/conv.py` | im2col/col2im + Conv2d, MaxPool2d, Flatten, Dropout |
| `src/models/cnn.py` | CuPy 기반 CNN 클래스 (NumPy fallback 지원) |
| `src/core/experiment.py` | config["model"] CNN 분기 추가 |
| `tests/stage3/` | activations, losses, metrics, layers, mlp, cnn, experiment 테스트 파일 7개 |
| `notebooks/stage3/` | stage3-1 ~ stage3-6 교육용 노트북 6개 |
