---
tags: [stage0, legacy, review]
created: 2026-06-15
updated: 2026-06-17
---

# Phase 0.1 레거시 재검토

## 1. 레거시 구성

레거시 코드는 `_core/legacy/src/` 에 task별 스크립트 6개와 공통 모듈 6개로 구성된다.

각 task는 **manual** 과 **module** 두 가지 구현 방식을 모두 제공한다.

| 폴더 | 파일 | 설명 |
|---|---|---|
| `binary/` | `mnist-binary-mlp-manual.py` | 이진 분류 수동 구현 |
| `binary/` | `mnist-binary-mlp-module.py` | 이진 분류 모듈 추상화 |
| `multiclass/` | `mnist-multiclass-mlp-manual.py` | 다중 분류 수동 구현 |
| `multiclass/` | `mnist-multiclass-mlp-module.py` | 다중 분류 모듈 추상화 |
| `regression/` | `mnist-regression-mlp-manual.py` | 회귀 수동 구현 |
| `regression/` | `mnist-regression-mlp-module.py` | 회귀 모듈 추상화 |

공통 모듈 구성은 다음과 같다.

| 파일 | 제공 요소 |
|---|---|
| `common/mnist.py` | `load_images`, `load_labels`, `one_hot` |
| `common/functions.py` | 활성화 함수, 손실 함수, 평가 지표 |
| `common/modules.py` | `Module`, `Linear`, `Sigmoid`, `ReLU`, `Sequential` |
| `common/optimizers.py` | `SGD`, `Adam` |
| `common/dataloader.py` | `Dataloader` |
| `common/trainer.py` | `MulticlassClassifier`, `BinaryClassifier`, `Regressor`, `train`, `evaluate`, `predict` |

## 2. common 모듈 상세

### 2.1. functions.py

`functions.py` 는 활성화 함수, 손실 함수, 평가 지표를 모두 포함한다.

| 분류 | 함수 |
|---|---|
| 활성화 | `identity`, `identity_grad`, `relu`, `relu_grad`, `sigmoid`, `sigmoid_grad`, `softmax` |
| 손실 | `cross_entropy`, `binary_cross_entropy`, `binary_cross_entropy_grad`, `mse`, `mse_grad`, `rmse` |
| 지표 | `accuracy`, `binary_accuracy`, `r2_score` |

### 2.2. modules.py

`modules.py` 는 `__call__` → `forward` 위임 구조의 `Module` 기반 클래스를 제공한다. `Sequential` 은 레이어 목록을 순서대로 forward하고 역순으로 backward한다. `Linear` 는 `x`, `grad_w`, `grad_b` 를 인스턴스에 캐싱하여 backward에서 재사용한다.

### 2.3. optimizers.py

`SGD` 와 `Adam` 은 생성 시 `model.params`, `model.grads` 를 참조로 받아 `step()` 호출 시 in-place 업데이트한다.

### 2.4. dataloader.py

`Dataloader` 는 `(images, labels)` 배열을 받아 `__iter__` 에서 배치를 yield한다. `shuffle`, `drop_last` 옵션을 지원한다.

### 2.5. trainer.py

`MulticlassClassifier`, `BinaryClassifier`, `Regressor` 는 각각 `train_step`, `eval_step`, `predict` 를 구현한다. 모듈 수준의 `train()`, `evaluate()`, `predict()` 함수는 Dataloader 를 순회하며 집계한다.

## 3. 두 가지 구현 패턴

레거시 스크립트는 manual 과 module 두 가지 패턴으로 동일한 파이프라인을 구현한다.

| 구간 | manual | module |
|---|---|---|
| 모델 | `w1, b1, w2, b2, w3, b3` 변수 직접 선언 | `Sequential(Linear, Sigmoid, ...)` |
| forward | `np.dot`, 활성화 함수 직접 호출 | `model(x)` |
| backward | gradient 변수 직접 계산 | `model.backward(dout)` |
| update | `w -= lr * grad` 직접 적용 | `optimizer.step()` |
| 학습 루프 | 인덱스 permutation → 슬라이싱 | `Dataloader` 순회 |
| task 래퍼 | 없음 | `MulticlassClassifier` / `BinaryClassifier` / `Regressor` |

두 패턴 모두 동일한 전체 구조(데이터 로딩 → 학습 → 평가 → 예측)를 따르며 동일한 결과를 출력한다.

## 4. task별 차이

공통 파이프라인에서 task별로 달라지는 항목은 다음과 같다.

| 항목 | multiclass | binary | regression |
|---|---|---|---|
| target 변환 | one-hot `(N, 10)` | 홀짝 이진화 `(N, 1)` | `label / 9.0` `(N, 1)` |
| BATCH_SIZE | 64 | 64 | 32 |
| output dimension | 10 | 1 | 1 |
| output activation | `softmax` | `sigmoid` | `identity` |
| loss | `cross_entropy` | `binary_cross_entropy` | `mse` |
| metric | `accuracy` | `binary_accuracy` | `r2_score` |
| output gradient | `(preds - y) / N` | `(preds - y) / N` | `2 * (preds - y) / N` |
| prediction 후처리 | `argmax` | `prob >= 0.5` | `round(clip(raw * 9.0, 0, 9))` |

## 5. 구현 파일 매핑

레거시 `common/` 모듈과 task 스크립트의 각 구간을 `src/` 구현 파일에 다음과 같이 매핑한다.

| 레거시 | src 대응 |
|---|---|
| `common/mnist.py` | `src/data/mnist.py` |
| `common/dataloader.py` | `src/data/dataloader.py` |
| `common/functions.py` (활성화) | `src/models/activations.py` |
| `common/functions.py` (손실·지표) | `src/models/losses.py` |
| `common/modules.py` | `src/models/layers.py`, `src/models/mlp.py` |
| `common/optimizers.py` | `src/core/optimizers.py` |
| `common/trainer.py` | `src/core/trainer.py`, `src/core/evaluator.py`, `src/core/predictor.py` |
| task 스크립트 하이퍼파라미터 | `src/config.py` |
| task 스크립트 target 변환 | `src/task.py` |

## 6. 주요 결정사항

레거시 분석 결과로 확정된 설계 원칙은 다음과 같다.

- 레거시 `common/` 모듈을 참조 구현으로 활용하여 `src/` 에서 재구현한다.
- task별 차이를 `src/task.py` 단일 파일에 집중하고, 나머지 파이프라인은 공통으로 유지한다.
- 레거시의 flat script 구조를 `src/core/` 실행 객체 + `scripts/` 클라이언트 구조로 전환한다.
- 하이퍼파라미터는 `src/config.py` 의 `get_default_config()` 로 관리한다.
- 레거시 `common/optimizers.py` 에 대응하는 `src/core/optimizers.py` 를 신설한다.
