---
tags: [docs, stage0, legacy-analysis]
created: "2026-06-20"
updated: "2026-06-20"
---

# 레거시 코드 구조와 구현 패턴

## 1. 개요

`_core/legacy/src/` 에는 이 프로젝트의 원형이 되는 레거시 코드가 보관되어 있다. 이 코드는 MNIST 데이터셋을 기반으로 multiclass classification, binary classification, regression 세 가지 task를 MLP로 학습하는 완성된 구현이다. Stage 1부터 Stage 6까지 새로운 `src/` 패키지를 설계하기 전에 레거시 코드의 구조와 두 가지 구현 패턴을 이해하고 task별 차이를 도출한다. 이 분석이 `src/` 패키지 구조와 공개 인터페이스 규약의 기반이 된다.

**목표**
- 레거시 코드의 전체 구조와 common 모듈 6개의 역할을 파악한다.
- manual 패턴과 module 패턴의 차이와 각각의 설계 의도를 이해한다.
- task별로 다른 구성 요소(target 변환, loss, metric, gradient, 후처리)를 도출한다.

## 2. 개념

레거시 코드를 분석하면 공통으로 반복되는 파이프라인 구조와 task마다 달라지는 구성 요소를 분리할 수 있다. 이 분리가 새로운 `src/` 패키지 설계의 핵심 원칙이다.

### 2.1. 공통 파이프라인

세 task 모두 동일한 데이터 전처리, 모델 구조, 학습·평가 루프를 공유한다.

공통 파이프라인의 각 단계는 다음과 같다.

- 이미지: `(N, 28, 28)` uint8 -> `(N, 784)` float32, `/255` 정규화
- 모델 구조: `784 -> 256 -> 128 -> output_dim`, hidden activation sigmoid
- 학습 루프: epoch 반복, mini-batch shuffle, forward -> loss/metric -> backward -> SGD update
- 평가 루프: test split 전체 배치 순회, 가중 평균 loss/metric 집계

이 공통 부분은 새로운 `src/data/`, `src/nn/`, `src/core/trainer.py`, `src/core/evaluator.py`에 각각 배치된다.

### 2.2. task별 차이

세 task는 공통 파이프라인을 공유하지만 target 변환, output dimension, 출력 활성화, 손실 함수, 평가 지표, 출력 gradient, 예측 후처리 7개 항목에서 차이가 있다.

task별 차이는 아래와 같다.

| 구분 | multiclass | binary | regression |
|---|---|---|---|
| target 변환 | `one_hot(labels, 10)` | `(labels % 2).reshape(-1, 1)` | `labels / 9.0` |
| target shape | `(N, 10)` float32 | `(N, 1)` float32 | `(N, 1)` float32 |
| output_dim | 10 | 1 | 1 |
| 출력 활성화 | softmax | sigmoid | identity |
| 손실 함수 | cross_entropy | binary_cross_entropy | mse |
| 평가 지표 | accuracy | binary_accuracy | r2_score |
| 출력 gradient | `(preds - y) / N` | `(preds - y) / N` | `2 * (preds - y) / N` |
| 예측 후처리 | argmax | `prob >= 0.5` -> 0/1 | `round(clip(raw * 9.0, 0, 9))` |

multiclass와 binary의 출력 gradient 수식 형태는 같지만 유도 경로가 다르다. multiclass는 softmax와 cross_entropy의 합성 미분에서, binary는 sigmoid와 binary_cross_entropy의 합성 미분에서 각각 `(preds - y) / N`이 도출된다. regression만 MSE 미분 계수 2가 추가된다.

## 3. 구현

레거시 코드는 task 폴더 3개와 common 폴더 1개로 구성된다.

```text
_core/legacy/src/
├── common/
│   ├── mnist.py
│   ├── functions.py
│   ├── modules.py
│   ├── optimizers.py
│   ├── dataloader.py
│   └── trainer.py
├── multiclass/
│   ├── mnist-multiclass-mlp-manual.py
│   └── mnist-multiclass-mlp-module.py
├── binary/
│   ├── mnist-binary-mlp-manual.py
│   └── mnist-binary-mlp-module.py
└── regression/
    ├── mnist-regression-mlp-manual.py
    └── mnist-regression-mlp-module.py
```

task 스크립트는 manual · module 2종씩 3개 task에 걸쳐 총 6개이며, common 모듈은 6개이다.

### 3.1. common 모듈별 역할

common 폴더의 6개 모듈이 각각 담당하는 역할과 새로운 `src/` 패키지와의 매핑은 아래와 같다.

| common 모듈 | 역할 | 새로운 src/ 매핑 |
|---|---|---|
| `mnist.py` | `load_images`, `load_labels`, `one_hot` | `src/data/mnist.py` |
| `functions.py` | activation, loss, metric, gradient 함수 | `src/nn/activations.py`, `src/nn/losses.py`, `src/nn/metrics.py` |
| `modules.py` | `Module`, `Linear`, `Sigmoid`, `ReLU`, `Sequential` | `src/nn/layers.py` |
| `optimizers.py` | `SGD`, `Adam` | `src/core/optimizers.py` |
| `dataloader.py` | `Dataloader` 배치 이터레이터 | `src/data/dataloader.py` |
| `trainer.py` | task별 Classifier, `train`, `evaluate`, `predict` 함수 | `src/core/trainer.py`, `src/core/evaluator.py`, `src/core/predictor.py` |

### 3.2. modules.py - Module 기반 레이어 설계

`modules.py`는 `params`와 `grads`를 리스트로 공유하여 optimizer가 별도 수집 없이 파라미터를 업데이트할 수 있는 설계를 사용한다.

```python
class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w = np.random.randn(in_features, out_features)
        self.b = np.zeros(out_features)
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)

        self.params.extend([self.w, self.b])
        self.grads.extend([self.grad_w, self.grad_b])

    def backward(self, dout):
        self.grad_w[...] = np.dot(self.x.T, dout)  # in-place 저장
        self.grad_b[...] = np.sum(dout, axis=0)
        return np.dot(dout, self.w.T)
```

`self.grad_w[...]`로 in-place 저장하는 이유는 `params`/`grads` 리스트가 동일 배열 객체의 참조를 보유하기 때문이다. `self.grad_w = ...`로 재할당하면 리스트가 가리키는 원본 배열과 연결이 끊어져 optimizer가 갱신된 gradient를 읽지 못한다. 이 설계 원칙은 새로운 `src/nn/layers.py`에서도 그대로 유지된다.

### 3.3. manual · module 두 가지 구현 패턴

task 스크립트는 동일한 결과를 내는 두 가지 구현 패턴으로 작성되어 있다.

manual 패턴은 파라미터를 배열 변수로 직접 선언하고, backward를 스크립트 내부에서 수동으로 계산한다.

```python
# manual 패턴 - 파라미터와 backward 직접 구현
w1 = np.random.randn(784, 256)
b1 = np.zeros(256)

z1 = np.dot(x, w1) + b1
a1 = sigmoid(z1)
preds = softmax(z3)

grad_z3 = (preds - y) / batch_size
grad_w3 = np.dot(a2.T, grad_z3)
w3 -= lr * grad_w3
```

module 패턴은 `Sequential`로 모델을 구성하고 `Classifier`와 `train`/`evaluate`/`predict` 함수를 통해 학습·평가·예측을 실행한다.

```python
# module 패턴 - common 모듈 조합
model = Sequential(
    Linear(784, 256), Sigmoid(),
    Linear(256, 128), Sigmoid(),
    Linear(128, 10),
)
optimizer = SGD(model, lr=LEARNING_RATE)
clf = MulticlassClassifier(model, optimizer)

train_loader = Dataloader(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
for epoch in range(1, NUM_EPOCHS + 1):
    loss, acc = train(clf, train_loader)
```

두 패턴의 차이는 아래와 같다.

| 항목 | manual | module |
|---|---|---|
| 파라미터 관리 | 배열 변수 직접 선언 | `Linear` 내 `params`/`grads` 리스트 |
| backward | 스크립트 내 수동 계산 | `model.backward(dout)` 위임 |
| 파라미터 업데이트 | `w -= lr * grad` 직접 | `optimizer.step()` |
| 데이터 순회 | for 루프 직접 | `Dataloader`, `train()`, `evaluate()` |
| 코드 재사용성 | 낮음 - task마다 전체 복사 | 높음 - common 모듈 조합 |

새로운 `src/` 패키지는 module 패턴을 기반으로 하되, task별 Classifier를 제거하고 task 규약을 `src/data/mnist.py`의 `get_task_spec()`으로 통합하는 방향으로 개선된다.

## 4. 사용법

레거시 코드는 분석·참조 전용이다. 실행이 필요한 경우 아래 명령을 사용한다.

```python
conda run -n numpy_py311 python _core/legacy/src/multiclass/mnist-multiclass-mlp-module.py
```

예상 출력은 다음과 같다.

```text
>> Training:
[ 1/10] loss:1.234 acc:0.678
...
[10/10] loss:0.412 acc:0.882

>> Evaluation:
loss:0.421 acc:0.876

>> Prediction:
Target: 7 | Prediction: 7
...
```

새로운 `src/` 구현과 비교할 때는 동일한 hyperparameter(`SEED=42`, `BATCH_SIZE=64`, `LEARNING_RATE=1e-2`, `NUM_EPOCHS=10`)를 사용하여 결과를 일치시킨다.

## 5. 테스트

Phase 0.0은 코드 작성이 없는 분석 단계이므로 대응하는 테스트 파일이 없다. 레거시 코드가 정상 동작하는지 확인하려면 아래 명령으로 스크립트를 직접 실행한다.

```bash
conda run -n numpy_py311 python _core/legacy/src/multiclass/mnist-multiclass-mlp-module.py
conda run -n numpy_py311 python _core/legacy/src/binary/mnist-binary-mlp-module.py
conda run -n numpy_py311 python _core/legacy/src/regression/mnist-regression-mlp-module.py
```

Stage 1부터 새로운 `src/` 구현과 함께 `tests/` 폴더 기반 TDD가 시작된다.

## 6. 요약

레거시 코드는 task 스크립트 6개와 common 모듈 6개로 구성되며, manual 패턴과 module 패턴 두 가지로 동일한 결과를 구현한다. 공통 파이프라인과 task별 차이를 분리하면 새로운 `src/` 패키지의 책임 경계가 명확하게 도출된다. common 모듈 6개는 Stage 1부터 6까지의 구현 파일에 1:1로 매핑되어 설계 기준이 된다.

다음 Phase에서는 [[phase0.1_conda-setup]]을 다룬다.
