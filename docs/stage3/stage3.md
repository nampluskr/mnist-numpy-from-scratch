---
tags: [docs, stage3, overview]
created: 2026-06-19
updated: 2026-06-19
---

# Stage 3 NumPy nn 모듈 및 MLP

## 1. 개요

Stage 3은 PyTorch의 `torch.nn`에 대응하는 NumPy 기반 신경망 구성요소를 `src/nn/` 패키지에 구현하고, 이를 조립하여 MLP 모델을 완성하는 단계이다.
activation 함수, layer 모듈, loss 함수, metric 함수를 각각 독립된 파일에 구현한 뒤, `src/models/mlp.py`에서 Sequential 기반으로 조립한다.
모든 레이어는 `Module` 기반 클래스로 구현하며 `forward()` / `backward()` 인터페이스를 통해 역전파가 일관되게 동작한다.

## 2. Phase 구성

### 2.1. Phase 3.1 activation 구현

`src/nn/activations.py`에 `sigmoid`, `softmax`, `identity`, `relu` 네 가지 activation 함수를 구현한다.
각 함수는 순전파 계산만 담당하며, 역전파 gradient는 `src/nn/losses.py`의 `*_grad` 함수에서 처리한다.

- [[phase3.1_activations|Phase 3.1 activation 구현]]

### 2.2. Phase 3.2 layer module 구현

`src/nn/layers.py`에 `Module` 기반 클래스와 `Linear`, `Sigmoid`, `ReLU`, `Sequential` 레이어를 구현한다.
`Linear.forward(x)`는 `xW + b`를 계산하고, `backward(dout)`는 upstream gradient를 받아 `grad_w`, `grad_b`를 인스턴스에 저장하고 하위 레이어로 전달할 gradient를 반환한다.
`Module` 클래스에는 training/eval 모드 전환(`train()`, `eval()`)이 포함된다.

- [[phase3.2_layers|Phase 3.2 layer module 구현]]

### 2.3. Phase 3.3 loss 및 gradient 구현

`src/nn/losses.py`에 `cross_entropy`, `binary_cross_entropy`, `mse` 손실 함수와 이들의 `*_grad` gradient 함수를 쌍으로 구현한다.
손실 함수는 raw logit을 입력으로 받아 내부에서 activation을 처리하고 scalar loss를 반환한다.
`*_grad` 함수는 `d(loss)/d(logits)`을 반환하며 Trainer에서 backward의 첫 입력으로 사용한다.

- [[phase3.3_losses|Phase 3.3 loss 및 gradient 구현]]

### 2.4. Phase 3.4 metric 구현

`src/nn/metrics.py`에 `accuracy`, `binary_accuracy`, `r2_score` 평가 지표 함수를 구현한다.
세 함수 모두 raw logit을 입력으로 받아 내부에서 예측값 변환 후 scalar metric 값을 반환한다.
Evaluator와 Trainer가 task spec을 통해 적절한 metric 함수를 선택한다.

- [[phase3.4_metrics|Phase 3.4 metric 함수 구현]]

### 2.5. Phase 3.5 MLP model 구현

`src/models/mlp.py`에 `src/nn/` 모듈을 조립하는 `MLP` 클래스를 구현한다.
`784 -> 256 -> 128 -> output_dim` 구조를 `Sequential`로 구성하고, `forward(x)`는 raw logit을 반환한다.
`params`와 `grads`는 list로 노출되어 optimizer가 in-place 업데이트할 수 있도록 한다.

- [[phase3.5_mlp|Phase 3.5 MLP model 구현]]

## 3. 주요 산출물

| 산출물 | 내용 |
|---|---|
| `src/nn/activations.py` | sigmoid, softmax, identity, relu 함수 |
| `src/nn/layers.py` | Module, Linear, Sigmoid, ReLU, Sequential 클래스 |
| `src/nn/losses.py` | cross_entropy, binary_cross_entropy, mse 및 *_grad 함수 |
| `src/nn/metrics.py` | accuracy, binary_accuracy, r2_score 함수 |
| `src/models/mlp.py` | Sequential 기반 MLP 클래스 |
| `tests/stage3/` | activations, layers, losses, metrics, mlp 테스트 파일 5개 |
