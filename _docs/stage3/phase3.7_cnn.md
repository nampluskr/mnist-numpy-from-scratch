---
tags: [docs, stage3]
created: 2026-06-17
updated: 2026-06-20
---

# Phase 3.7 CNN 모델 구현

## 1. 개요

`src/models/cnn.py`에 CuPy 기반 CNN 클래스를 구현한다. MLP와 동일한 외부 인터페이스(`forward`, `backward`, `params`, `grads`)를 유지하여 기존 `Trainer`, `Evaluator`, `Experiment` 코드를 수정 없이 재사용한다.
CuPy 미설치 환경에서는 NumPy로 자동 fallback하여 동일한 코드가 CPU에서도 실행된다.
`src/core/experiment.py`에 `config["model"]` 분기를 추가하여 MLP/CNN 선택을 지원한다.

## 2. 모델 구조

```
Input (N, 784)
  ↓ reshape to (N, 1, 28, 28)
Conv2d(1→32, K=3, pad=1)  →  (N, 32, 28, 28)
ReLU
MaxPool2d(2, 2)            →  (N, 32, 14, 14)
Conv2d(32→64, K=3, pad=1) →  (N, 64, 14, 14)
ReLU
MaxPool2d(2, 2)            →  (N, 64, 7, 7)
Flatten                    →  (N, 3136)
  ↓ CuPy → numpy 변환
Linear(3136→256)
ReLU
Dropout(0.5)
Linear(256→output_dim)
Output (N, output_dim)     ← numpy float32
```

## 3. CuPy/NumPy 경계 처리

MnistDataset은 `(N, 784)` numpy 배열을 반환한다. CNN 내부에서 CuPy 변환과 역변환을 처리하므로 DataLoader 및 손실 함수는 수정이 필요 없다.

| 지점 | 변환 |
|---|---|
| `forward()` 시작 | `xp.asarray(x).reshape(-1, 1, 28, 28)` — numpy → CuPy |
| `Flatten` 직후 | `np.asarray(x_xp)` — CuPy → numpy |
| `backward()` 시작 | `xp.asarray(grad_out)` — numpy → CuPy |

`conv_net`의 params/grads는 CuPy 배열, `fc_net`의 params/grads는 numpy 배열이다.
`SGD.step()`은 `param -= lr * grad`로 in-place 갱신하며, 두 타입 모두 지원한다.

## 4. CuPy fallback

```python
try:
    import cupy as _xp
except ImportError:
    import numpy as _xp
```

## 5. experiment.py CNN 분기

```python
model_type = config.get("model", "mlp")
if model_type == "cnn":
    self.model = CNN(task=task, seed=config["seed"])
else:
    self.model = MLP(task=task, seed=config["seed"])
```

기본값은 `"mlp"`이며 기존 코드와 하위 호환된다.

## 6. 인터페이스 호환성

| 인터페이스 | MLP | CNN |
|---|---|---|
| `model.forward(x)` | `(N, 784) → (N, output_dim) numpy` | 동일 |
| `model.backward(grad)` | numpy grad 입력 | 동일 |
| `model.params` | `list[np.ndarray]` | `list[np.ndarray]` (CuPy fallback → numpy) |
| `model.grads` | `list[np.ndarray]` | 동일 |
| `model.train()` / `eval()` | `Module.training` 전파 | 동일 |

## 7. 테스트 결과

```
pytest tests/stage3/test_cnn.py tests/stage3/test_experiment.py -v
73 passed
```

| 테스트 클래스 | 항목 수 | 검증 내용 |
|---|---|---|
| `TestCNNForward` | 6 | 3종 task shape, dtype, numpy 반환, batch=1 |
| `TestCNNBackward` | 2 | 실행 무오류, grad 갱신 확인 |
| `TestCNNParamsGrads` | 5 | list, 길이, shape 일치, SGD 갱신 |
| `TestCNNTrainEval` | 3 | eval 결정론적, train 확률적, flag 전파 |
| `TestExperimentModelSelection` | 4 | CNN/MLP 분기, 기본값 MLP, loader 조립 |
| `TestCNNExperimentRun` | 12 | 3종 task × log 구조 검증 |
| `TestCNNExperimentLoss` | 4 | multiclass/binary loss 유한값 검증 |
| `TestCNNTrainerStep` | 2 | Trainer 1 epoch 직접 실행 |

## 8. 실행 명령

```bash
pytest tests/stage3/test_cnn.py tests/stage3/test_experiment.py -v
```
