---
tags: [project, stage0]
created: 2026-06-17
updated: 2026-06-17
---

# Phase 0.1 레거시 코드 분석

이 문서는 `_core/legacy/src/` 에 보관된 레거시 코드의 전체 구조, common 모듈별 제공 요소, 두 가지 구현 패턴, task별 차이를 분석한다.

## 1. 레거시 코드 구조

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

## 2. common 모듈별 제공 요소

common 폴더의 6개 모듈이 제공하는 요소를 모듈별로 정리한다.

### 2.1. mnist.py - 데이터 로딩

`mnist.py`는 로컬 gzip 파일에서 이미지와 레이블을 읽어 numpy 배열로 반환하는 함수와 one-hot 인코딩 헬퍼를 제공한다.

| 요소 | 시그니처 | 반환 |
|---|---|---|
| `load_images` | `(data_dir, split="train")` | `(N, 28, 28)` uint8 |
| `load_labels` | `(data_dir, split="train")` | `(N,)` uint8 |
| `one_hot` | `(x, num_classes)` | `(N, num_classes)` float64 |

### 2.2. functions.py - 활성화 함수·손실 함수·지표·기울기

`functions.py`는 forward 전용 활성화 함수, 손실 함수, 평가 지표, 수동 backward에 사용하는 기울기 함수를 제공한다.

활성화 함수와 기울기 함수는 아래와 같다.

| 요소 | 입력 | 비고 |
|---|---|---|
| `identity(x)` | `np.ndarray` | forward 전용 |
| `identity_grad(x)` | `np.ndarray` | ones_like 반환 |
| `relu(x)` | `np.ndarray` | forward 전용 |
| `relu_grad(x)` | `np.ndarray` | binary mask |
| `sigmoid(x)` | `np.ndarray` | numerically stable (부호별 분기) |
| `sigmoid_grad(x)` | sigmoid 출력값 | `x * (1 - x)` |
| `softmax(x)` | `np.ndarray` (1D/2D) | max subtraction 안정화 |

손실 함수와 평가 지표는 아래와 같다.

| 요소 | 입력 | 반환 | 비고 |
|---|---|---|---|
| `cross_entropy(preds, targets)` | softmax 출력, one-hot | scalar | 1e-8 clip |
| `binary_cross_entropy(preds, targets)` | sigmoid 출력, 이진 레이블 | scalar | 1e-8 clip |
| `binary_cross_entropy_grad(preds, targets)` | 위와 동일 | `(N, 1)` | batch_size 나눔 |
| `mse(preds, targets)` | 예측값, 정규화 레이블 | scalar | |
| `mse_grad(preds, targets)` | 위와 동일 | `(N, 1)` | `2*(p-y)/N` |
| `accuracy(preds, targets)` | softmax 출력, one-hot 또는 정수 | scalar | argmax 비교 |
| `binary_accuracy(preds, targets)` | sigmoid 출력, 이진 레이블 | scalar | threshold 0.5 |
| `r2_score(preds, targets)` | 예측값, 정규화 레이블 | scalar | 1e-8 보정 |
| `rmse(preds, targets)` | 예측값, 정규화 레이블 | scalar | mse의 루트 |

### 2.3. modules.py - 레이어 모듈

`modules.py`는 `Module` 기반 클래스와 `params`/`grads` 리스트를 통해 optimizer와 연동하는 레이어를 제공한다.

| 요소 | 책임 |
|---|---|
| `Module` | `params`, `grads` 보유, `__call__` → `forward` 위임 |
| `Linear` | `w`, `b` 초기화, `forward`(matmul+bias), `backward`(grad_w, grad_b 저장, dx 반환) |
| `Sigmoid` | `forward`(출력 캐시), `backward`(sigmoid 미분) |
| `ReLU` | `forward`(mask 저장), `backward`(mask 적용) |
| `Sequential` | `layers` 순차 실행, `params`/`grads`를 전체 레이어에서 취합 |

`Linear.backward`는 `grad_w[...] = ...` 형식으로 in-place 저장하며, `params`/`grads` 리스트가 동일 배열 객체를 가리키기 때문에 optimizer가 별도 수집 없이 업데이트할 수 있다.

### 2.4. optimizers.py - 파라미터 업데이트

`optimizers.py`는 `model.params`와 `model.grads`를 수신하여 in-place 업데이트를 수행하는 두 옵티마이저를 제공한다.

| 요소 | 생성자 인자 | 알고리즘 |
|---|---|---|
| `SGD` | `model, lr` | `param -= lr * grad` |
| `Adam` | `model, lr, beta1=0.9, beta2=0.999` | bias correction 포함 Adam |

두 클래스 모두 `step()` 메서드로 업데이트를 실행하며 반환값이 없다.

### 2.5. dataloader.py - 배치 이터레이터

`dataloader.py`는 이미지 배열과 레이블 배열을 직접 수신하는 배치 이터레이터를 제공한다.

| 요소 | 시그니처 | 비고 |
|---|---|---|
| `Dataloader` | `images, labels, batch_size, shuffle=False, drop_last=False` | `(images_batch, labels_batch)` yield |

`__len__`은 `num_batches`를 반환하고, `__iter__`는 epoch마다 인덱스를 생성하여 배치를 yield한다.

### 2.6. trainer.py - 학습·평가·예측 실행

`trainer.py`는 task별 Classifier 클래스와 공통 루프 함수를 제공한다.

| 요소 | 책임 |
|---|---|
| `MulticlassClassifier` | softmax forward, cross_entropy loss, accuracy metric, manual gradient |
| `BinaryClassifier` | sigmoid forward, binary_cross_entropy loss, binary_accuracy metric, manual gradient |
| `Regressor` | identity forward, mse loss, r2_score metric, manual gradient |
| `train(model, dataloader)` | batch 반복, train_step 호출, 가중 평균 loss/acc 반환 |
| `evaluate(model, dataloader)` | batch 반복, eval_step 호출, 가중 평균 loss/acc 반환 |
| `predict(model, x)` | model.predict 위임 |

각 Classifier의 `train_step`은 forward → loss/metric → dout 계산 → `model.backward(dout)` → `optimizer.step()` 순서로 실행한다.

## 3. manual · module 두 가지 구현 패턴

레거시 task 스크립트는 동일한 결과를 내는 두 가지 구현 패턴으로 작성되어 있다.

### 3.1. manual 패턴

manual 패턴은 파라미터를 배열 변수로 직접 선언하고, backward를 스크립트 내부에서 수동으로 구현한다.

```python
# 파라미터 선언
w1 = np.random.randn(784, 256)
b1 = np.zeros(256)
...

# forward
z1 = np.dot(x, w1) + b1
a1 = sigmoid(z1)
...
preds = softmax(z3)

# backward (수동)
grad_z3 = (preds - y) / batch_size
grad_w3 = np.dot(a2.T, grad_z3)
grad_b3 = np.sum(grad_z3, axis=0)
...

# 파라미터 업데이트
w1 -= LEARNING_RATE * grad_w1
...
```

### 3.2. module 패턴

module 패턴은 `Sequential`로 모델을 구성하고, `Classifier`와 `train`/`evaluate`/`predict` 함수를 통해 학습·평가·예측을 실행한다.

```python
# 모델 구성
model = Sequential(Linear(784, 256), Sigmoid(), Linear(256, 128), Sigmoid(), Linear(128, 10))
optimizer = SGD(model, lr=LEARNING_RATE)
clf = MulticlassClassifier(model, optimizer)

# 데이터 로더
train_loader = Dataloader(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)

# 학습
for epoch in range(1, NUM_EPOCHS + 1):
    loss, acc = train(clf, train_loader)

# 평가
loss, acc = evaluate(clf, test_loader)

# 예측
preds = predict(clf, x)
```

두 패턴의 차이는 아래와 같다.

| 항목 | manual | module |
|---|---|---|
| 파라미터 관리 | 배열 변수 직접 선언 | `Linear` 내 인스턴스 변수, `params`/`grads` 리스트 |
| backward | 스크립트 내 수동 계산 | `model.backward(dout)` 위임 |
| 파라미터 업데이트 | `w -= lr * grad` 직접 | `optimizer.step()` |
| 데이터 순회 | for 루프 직접 | `Dataloader`, `train()`, `evaluate()` |
| 예측 출력 | 스크립트 내 직접 처리 | `predict()` 함수 위임 |
| 코드 재사용성 | 낮음 - task마다 전체 복사 | 높음 - common 모듈 조합 |

## 4. task별 차이

3개 task는 공통 파이프라인을 공유하지만, target 변환, output dimension, 출력 활성화 함수, 손실 함수, 평가 지표, 출력 gradient, 예측 후처리에서 차이가 있다.

공통 파이프라인은 아래와 같다.

- 이미지: `(N, 28, 28)` uint8 → `(N, 784)` float32 `/255` 정규화
- 모델 구조: `784 → 256 → 128 → output_dim`, hidden activation `sigmoid`
- 학습 루프: epoch 반복, mini-batch shuffle, forward → loss/metric → backward → SGD update
- 평가 루프: test split 전체 배치 순회, 가중 평균 loss/metric 집계

task별 차이는 아래와 같다.

| 구분 | multiclass | binary | regression |
|---|---|---|---|
| target 변환 | `one_hot(labels, 10)` | `(labels % 2).reshape(-1, 1)` | `labels / 9.0` |
| target shape | `(N, 10)` float32 | `(N, 1)` float32 | `(N, 1)` float32 |
| output_dim | 10 | 1 | 1 |
| 출력 활성화 | `softmax` | `sigmoid` | `identity` |
| 손실 함수 | `cross_entropy` | `binary_cross_entropy` | `mse` |
| 평가 지표 | `accuracy` | `binary_accuracy` | `r2_score` |
| 출력 gradient | `(preds - y) / N` | `(preds - y) / N` | `2 * (preds - y) / N` |
| 예측 후처리 | `argmax` | `prob >= 0.5` → 0/1 | `round(clip(raw * 9.0, 0, 9))` |

multiclass와 binary의 출력 gradient 수식은 동일하지만, multiclass는 softmax + cross_entropy 합성 미분에서, binary는 sigmoid + binary_cross_entropy 합성 미분에서 각각 같은 형태 `(preds - y) / N` 이 도출된다. regression만 MSE 미분 계수 2가 추가된다.
