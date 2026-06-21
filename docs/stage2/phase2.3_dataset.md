---
tags: [docs, stage2, dataset, mnist]
created: "2026-06-20"
updated: "2026-06-21"
---

# Dataset과 transform

## 1. 개요

`src/data/datasets.py`와 `src/data/transforms.py`는 `load_images`/`load_labels`가 반환한 원본 배열을 학습에 사용할 수 있는 형태로 변환하여 `Dataloader`에 공급하는 역할을 한다.

`transforms.py`는 이미지와 레이블에 적용할 변환 함수를 제공한다. 각 함수는 독립적으로 사용하거나 `lambda`로 조합할 수 있다.

`datasets.py`는 `MNISTDataset` base 클래스와 task별 3개 클래스를 제공한다. 생성자에서 `transform`과 `target_transform`을 인자로 받아 이미지와 레이블에 각각 적용한다. 기본값이 지정된 task별 클래스를 사용하면 별도 transform 없이 바로 사용할 수 있고, base 클래스를 직접 사용하면 transform을 자유롭게 구성할 수 있다.

이 구조는 후속 PyTorch, TensorFlow, JAX 프로젝트에서 동일하게 유지된다. 각 프레임워크는 `load_images`/`load_labels`를 공유하면서, 자체 Dataset과 transform을 같은 인터페이스로 구성한다.

## 2. 개념

### 2.1. transform / target_transform 패턴

transform과 target_transform은 Dataset 생성 시 외부에서 주입하는 callable이다. 생성자에서 이미지와 레이블 전체에 한 번만 적용하여 `self.images`와 `self.targets`에 저장한다. 이 eager 적용 방식은 per-sample 변환보다 메모리 접근 효율이 높다.

| 인자 | 적용 대상 | 역할 |
|---|---|---|
| `transform` | images | 정규화, reshape 등 이미지 전처리 |
| `target_transform` | labels | one-hot, 이진화 등 레이블 변환 |

`None`이면 원본 배열을 그대로 사용한다. task별 클래스는 각 task에 맞는 기본값을 내부에서 지정하여, 인자 없이 생성해도 학습에 바로 사용할 수 있다.

### 2.2. task별 target 변환

MNIST 레이블은 0부터 9까지의 정수이다. task에 따라 이 레이블을 다른 형태로 변환해야 모델이 올바른 목적 함수를 학습할 수 있다.

| task | 변환 함수 | 출력 shape | 범위 |
|---|---|---|---|
| `multiclass` | `one_hot` | `(N, 10)` | 0.0 또는 1.0 |
| `binary` | `binarize` | `(N, 1)` | 0.0 또는 1.0 |
| `regression` | `to_regression` | `(N, 1)` | `[0.0, 1.0]` |

`binary` 변환에서 홀수 레이블(1, 3, 5, 7, 9)은 1.0, 짝수 레이블(0, 2, 4, 6, 8)은 0.0으로 변환한다. `regression`에서 레이블 9는 1.0, 레이블 0은 0.0으로 정규화한다.

### 2.3. Dataset 프로토콜

Dataset은 `__len__`과 `__getitem__` 두 메서드를 구현한 객체이다. `__len__`은 전체 샘플 수를 반환하고, `__getitem__`은 인덱스를 받아 단일 `(image, target)` tuple을 반환한다. `Dataloader`는 이 인터페이스만 요구하므로 MNIST 외의 데이터셋도 같은 `Dataloader`와 조합할 수 있다.

## 3. 구현

### 3.1. transforms.py

`transforms.py`는 이미지와 레이블에 적용할 함수를 제공한다. 모든 함수는 NumPy 배열을 받아 NumPy 배열을 반환한다.

공개 함수 목록은 다음과 같다.

| 함수 | 입력 | 출력 | 설명 |
|---|---|---|---|
| `normalize` | `(N, 28, 28) uint8` | `(N, 28, 28) float32` | `/255.0` 정규화 |
| `to_flat` | `(N, 28, 28)` | `(N, 784)` | MLP 입력 형태로 reshape |
| `one_hot` | `(N,) uint8` | `(N, 10) float32` | multiclass one-hot 인코딩 |
| `binarize` | `(N,) uint8` | `(N, 1) float32` | 홀수/짝수 이진화 (`labels % 2`) |
| `to_regression` | `(N,) uint8` | `(N, 1) float32` | 레이블 정규화 (`labels / 9.0`) |

구현 예시는 다음과 같다.

```python
def one_hot(labels, num_classes=10):
    n = len(labels)
    out = np.zeros((n, num_classes), dtype=np.float32)
    out[np.arange(n), labels] = 1.0
    return out
```

`out[np.arange(n), labels] = 1.0`은 각 행의 레이블 위치에 1.0을 대입하는 fancy indexing이다. 나머지 위치는 `np.zeros`의 0.0 초기값이 유지된다.

### 3.2. datasets.py

`datasets.py`는 `MNISTDataset` base 클래스와 task별 3개 클래스를 제공한다.

공개 클래스 목록은 다음과 같다.

| 클래스 | 기본 transform | 기본 target_transform | 설명 |
|---|---|---|---|
| `MNISTDataset` | 없음 (원본 반환) | 없음 (원본 반환) | base 클래스 - transform 자유 구성 |
| `MulticlassDataset` | `normalize + to_flat` | `one_hot` | multiclass 분류용 |
| `BinaryDataset` | `normalize + to_flat` | `binarize` | binary 분류용 |
| `RegressionDataset` | `normalize + to_flat` | `to_regression` | regression용 |

`MNISTDataset` base 클래스의 구현은 다음과 같다.

```python
class MNISTDataset:
    def __init__(self, split, transform=None, target_transform=None, dataset_dir=None):
        images = load_images(split, dataset_dir=dataset_dir)
        labels = load_labels(split, dataset_dir=dataset_dir)
        self.images = transform(images) if transform is not None else images
        self.targets = target_transform(labels) if target_transform is not None else labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]
```

task별 클래스는 `None` 인자를 받으면 기본 transform을 대입한 뒤 base 클래스 생성자에 전달한다.

```python
class MulticlassDataset(MNISTDataset):
    def __init__(self, split, transform=None, target_transform=None, dataset_dir=None):
        if transform is None:
            transform = lambda x: T.to_flat(T.normalize(x))
        if target_transform is None:
            target_transform = T.one_hot
        super().__init__(split, transform, target_transform, dataset_dir=dataset_dir)
```

## 4. 사용법

task별 클래스를 기본 설정으로 사용하는 예제는 다음과 같다.

```python
from src.data.datasets import MulticlassDataset, BinaryDataset, RegressionDataset

ds = MulticlassDataset("train")
image, target = ds[0]
print(image.shape, target.shape)

ds2 = BinaryDataset("train")
image2, target2 = ds2[0]
print(image2.shape, target2.shape)
```

예상 출력은 다음과 같다.

```text
(784,) (10,)
(784,) (1,)
```

transform을 직접 구성하는 예제는 다음과 같다. `MNISTDataset` base 클래스를 사용하면 reshape 없이 2D 이미지를 유지할 수 있다.

```python
from src.data.datasets import MNISTDataset
from src.data import transforms as T

ds = MNISTDataset("train",
    transform=T.normalize,
    target_transform=T.one_hot)
image, target = ds[0]
print(image.shape, target.shape)
```

예상 출력은 다음과 같다.

```text
(28, 28) (10,)
```

프로젝트 통합 예제는 다음과 같다. 3개 task에 대해 Dataset과 `Dataloader`를 조합한다.

```python
from src.data.datasets import MulticlassDataset, BinaryDataset, RegressionDataset
from src.data.dataloader import Dataloader

for DatasetCls in (MulticlassDataset, BinaryDataset, RegressionDataset):
    ds = DatasetCls("train")
    loader = Dataloader(ds, batch_size=64, shuffle=True)
    for images, targets in loader:
        print(DatasetCls.__name__, images.shape, targets.shape)
        break
```

## 5. 테스트

테스트 파일은 `tests/stage2/test_dataset.py`이다.

```bash
conda run -n numpy_py311 pytest tests/stage2/test_dataset.py -v
```

테스트 구성은 다음과 같다.

| 클래스 | 주요 검증 내용 |
|---|---|
| `TestMNISTDataset` | transform=None 시 원본 반환, custom transform 적용, `__len__`, `__getitem__`, 잘못된 split 예외 |
| `TestMulticlassDataset` | images shape `(N, 784)`, dtype float32, 정규화 범위, targets shape `(N, 10)`, one-hot 합, `__getitem__` shape, transform override |
| `TestBinaryDataset` | targets shape `(N, 1)`, dtype float32, 0/1 값 범위, 홀수=1 검증, `__getitem__` shape |
| `TestRegressionDataset` | targets shape `(N, 1)`, dtype float32, `[0, 1]` 범위, 경계값 검증, `__getitem__` shape |
| `TestDatasetsReal` | 실제 MNIST train/test shape 검증 (파일 없으면 skip) |

단위 테스트는 `tmp_path` fixture로 합성 gz 파일(n=20)을 생성하여 실제 MNIST 파일에 의존하지 않는다.

## 6. 요약

`transforms.py`는 이미지와 레이블에 적용할 독립 함수를 제공하고, `datasets.py`는 이 함수를 조합하여 `transform`/`target_transform` 인터페이스로 주입받는 Dataset 클래스를 제공한다. task별 클래스(`MulticlassDataset`, `BinaryDataset`, `RegressionDataset`)는 기본 transform을 내장하여 인자 없이 생성해도 바로 사용할 수 있다. 이 구조는 후속 프레임워크 프로젝트에서 동일하게 유지된다.

다음 Phase에서는 [[phase2.4_dataloader]]를 다룬다.
