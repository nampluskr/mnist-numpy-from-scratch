---
tags: [docs, stage2, transforms, data]
created: "2026-06-21"
updated: "2026-06-21"
---

# 이미지와 레이블 변환 함수

## 1. 개요

`src/data/transforms.py`는 `load_images`/`load_labels`가 반환한 원본 배열을 모델 학습에 적합한 형태로 변환하는 함수를 제공한다. 이미지 변환 함수(`normalize`, `to_flat`)와 레이블 변환 함수(`one_hot`, `binarize`, `to_regression`) 5개로 구성되며, 모두 NumPy 배열을 받아 NumPy 배열을 반환하는 순수 함수이다. Dataset 생성 시 `transform`/`target_transform` 인자로 외부에서 주입하며, Dataset 내부에서 직접 호출하지 않는다.

**목표**
- 원본 `uint8` 이미지를 `[0, 1]` 범위 `float32`로 정규화하고 MLP 입력 형태인 `(N, 784)`로 reshape한다.
- task에 따라 정수 레이블을 `multiclass` one-hot, `binary` 이진화, `regression` 정규화로 변환한다.
- 각 함수를 독립적으로 사용하거나 `lambda`로 자유롭게 조합할 수 있도록 단일 책임 원칙을 유지한다.

## 2. 개념

### 2.1. 이미지 정규화와 reshape

원본 MNIST 이미지는 픽셀 값이 0부터 255 사이의 정수(`uint8`)이다. 신경망은 입력 값이 비슷한 크기여야 gradient가 안정적으로 흐른다. 픽셀 값을 255로 나누면 `[0, 1]` 범위로 정규화되어 학습이 빨라지고 수치 안정성이 높아진다.

또한 원본 이미지 shape은 `(N, 28, 28)`이다. MLP는 1차원 벡터 입력을 요구하므로 `(N, 784)`로 펼쳐야 한다. CNN은 2D 형태를 유지해야 하므로, `to_flat`은 별도 함수로 분리하여 필요한 경우에만 적용한다.

이미지 변환 함수의 역할은 다음과 같다.

| 함수 | 입력 shape/dtype | 출력 shape/dtype | 역할 |
|---|---|---|---|
| `normalize` | `(N, 28, 28) uint8` | `(N, 28, 28) float32` | `/255.0` 정규화 |
| `to_flat` | `(N, 28, 28) any` | `(N, 784) any` | MLP 입력용 reshape |

MLP에 공급할 때는 두 함수를 `lambda x: to_flat(normalize(x))`로 조합한다. CNN에 공급할 때는 `normalize`만 적용하여 2D 형태를 유지한다.

### 2.2. task별 레이블 변환

MNIST 레이블은 0부터 9까지의 정수이다. task에 따라 다른 목적 함수를 사용하므로 레이블을 각 task에 맞는 형태로 변환해야 한다.

**multiclass**: 10개 클래스 중 하나를 예측한다. 레이블 정수를 길이 10인 one-hot 벡터로 변환한다.

$$
y_{\text{one-hot}}[k] = \begin{cases} 1.0 & k = \text{label} \\ 0.0 & \text{otherwise} \end{cases}
$$

레이블 3은 `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`으로 변환된다. cross entropy loss는 이 one-hot 벡터와 모델의 softmax 출력을 비교한다.

**binary**: 숫자가 홀수인지 짝수인지 판별한다. 레이블을 2로 나눈 나머지를 그대로 사용한다.

$$
y_{\text{binary}} = \text{label} \bmod 2 \quad \in \{0, 1\}
$$

홀수(1, 3, 5, 7, 9)는 1.0, 짝수(0, 2, 4, 6, 8)는 0.0이 된다. binary cross entropy loss가 이 값과 sigmoid 출력을 비교한다.

**regression**: 숫자 값 자체를 예측한다. 레이블을 9로 나눠 `[0, 1]` 범위로 정규화한다.

$$
y_{\text{regression}} = \frac{\text{label}}{9}
$$

레이블 0은 0.0, 레이블 9는 1.0이 된다. MSE loss가 이 값과 identity 출력을 비교한다. 예측 결과는 후처리에서 9를 곱해 원래 스케일로 복원한다.

레이블 변환 함수의 역할은 다음과 같다.

| 함수 | task | 변환 규칙 | 출력 shape | 범위 |
|---|---|---|---|---|
| `one_hot` | multiclass | 레이블 위치를 1.0으로 설정 | `(N, 10)` | 0.0 또는 1.0 |
| `binarize` | binary | `labels % 2` | `(N, 1)` | 0.0 또는 1.0 |
| `to_regression` | regression | `labels / 9.0` | `(N, 1)` | `[0.0, 1.0]` |

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `normalize` | 함수 | `(N, 28, 28) uint8` | `(N, 28, 28) float32` | `/255.0` 정규화 |
| `to_flat` | 함수 | `(N, 28, 28) any` | `(N, 784) any` | MLP 입력용 reshape |
| `one_hot` | 함수 | `(N,) uint8` | `(N, 10) float32` | multiclass one-hot 인코딩 |
| `binarize` | 함수 | `(N,) uint8` | `(N, 1) float32` | 홀수/짝수 이진화 |
| `to_regression` | 함수 | `(N,) uint8` | `(N, 1) float32` | `/9.0` 정규화 |

### 3.1. 이미지 변환 구현

`normalize`는 dtype을 `float32`로 변환한 뒤 255로 나눈다. `to_flat`은 첫 번째 차원(`N`)을 유지하고 나머지를 하나로 합친다.

```python
def normalize(images):
    return images.astype(np.float32) / 255.0


def to_flat(images):
    return images.reshape(len(images), -1)
```

`reshape(len(images), -1)`에서 `-1`은 나머지 차원을 자동으로 계산하라는 의미이다. `(N, 28, 28)`이면 `28 * 28 = 784`로 결정된다. dtype은 변경하지 않으므로 `normalize` 이후에 적용하면 `float32` 그대로 유지된다.

### 3.2. 레이블 변환 구현

`one_hot`은 `np.zeros`로 `(N, 10)` 배열을 초기화한 뒤 fancy indexing으로 해당 위치에 1.0을 대입한다.

```python
def one_hot(labels, num_classes=10):
    n = len(labels)
    out = np.zeros((n, num_classes), dtype=np.float32)
    out[np.arange(n), labels] = 1.0
    return out
```

`out[np.arange(n), labels] = 1.0`은 행 인덱스 `[0, 1, 2, ...]`와 열 인덱스 `labels`를 동시에 지정하는 fancy indexing이다. 반복문 없이 N개의 1.0을 한 번에 대입한다.

`binarize`와 `to_regression`은 NumPy 연산만으로 변환을 완료한다.

```python
def binarize(labels):
    return (labels % 2).astype(np.float32).reshape(-1, 1)


def to_regression(labels):
    return labels.astype(np.float32).reshape(-1, 1) / 9.0
```

두 함수 모두 `reshape(-1, 1)`로 `(N,)` 배열을 `(N, 1)`로 변환한다. 모델 출력이 `(N, 1)` shape이므로 loss 계산 시 shape이 일치해야 한다.

## 4. 사용법

개별 함수 사용 예제는 다음과 같다.

```python
import numpy as np
from src.data.transforms import normalize, to_flat, one_hot, binarize, to_regression

images = np.random.randint(0, 256, (4, 28, 28), dtype=np.uint8)
labels = np.array([3, 7, 0, 9], dtype=np.uint8)

print(normalize(images).shape, normalize(images).dtype)
print(to_flat(normalize(images)).shape)
print(one_hot(labels).shape)
print(binarize(labels).T)
print(to_regression(labels).T)
```

예상 출력은 다음과 같다.

```text
(4, 28, 28) float32
(4, 784)
(4, 10)
[[1. 1. 0. 1.]]
[[0.333 0.778 0.    1.   ]]
```

프로젝트 통합 예제는 다음과 같다. `MulticlassDataset`은 내부적으로 아래와 같은 transform 조합을 기본값으로 사용한다.

```python
from src.data.mnist import load_images, load_labels
from src.data import transforms as T

images = load_images("train")
labels = load_labels("train")

# MulticlassDataset 기본 transform 동작 재현
x = T.to_flat(T.normalize(images))
y = T.one_hot(labels)
print(x.shape, x.dtype)
print(y.shape, y.sum(axis=1)[:5])
```

예상 출력은 다음과 같다.

```text
(60000, 784) float32
(60000, 10) [1. 1. 1. 1. 1.]
```

## 5. 테스트

테스트 파일은 `tests/stage2/test_transforms.py`이다.

```bash
conda run -n numpy_py311 pytest tests/stage2/test_transforms.py -v
```

테스트 구성은 다음과 같다.

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestNormalize` | 6 | dtype float32, shape 보존, 범위 `[0, 1]`, 0 픽셀, 255 픽셀 경계값 |
| `TestToFlat` | 4 | output shape `(N, 784)`, dtype 보존, 값 보존, normalize와의 조합 |
| `TestOneHot` | 7 | output shape `(N, 10)`, dtype float32, row sum = 1, 이진 값, 정확한 위치, 경계 레이블 0/9 |
| `TestBinarize` | 5 | output shape `(N, 1)`, dtype float32, 이진 값, 홀수=1 짝수=0 |
| `TestToRegression` | 7 | output shape `(N, 1)`, dtype float32, 범위 `[0, 1]`, 레이블 0=0.0, 레이블 9=1.0, 전체 값 검증 |

단위 테스트는 MNIST 파일 의존 없이 `np.random` 합성 배열과 소수의 레이블 배열로 각 함수를 독립 검증한다.

## 6. 요약

`transforms.py`는 원본 `uint8` 배열을 모델 입력에 적합한 형태로 변환하는 5개의 독립 함수를 제공한다. 이미지 변환(`normalize`, `to_flat`)과 레이블 변환(`one_hot`, `binarize`, `to_regression`)을 분리하여 task와 모델 종류에 따라 자유롭게 조합할 수 있다. Dataset 생성 시 외부에서 주입하는 방식으로 설계하여 후속 프레임워크 프로젝트에서도 동일한 패턴을 유지한다.

다음 Phase에서는 [[phase2.3_dataset]]을 다룬다.
