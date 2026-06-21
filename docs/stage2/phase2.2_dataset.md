---
tags: [docs, stage2, dataset, mnist]
created: "2026-06-20"
updated: "2026-06-20"
---

# Dataset 클래스와 target 변환

## 1. 개요

`MNISTDataset`은 `load_mnist`로 로드한 원본 배열에 정규화와 task별 target 변환을 적용하여 `Dataloader`가 소비할 수 있는 Dataset 인터페이스를 제공한다. `__len__`과 `__getitem__` 프로토콜을 구현하여 `Dataloader`와의 조합이 가능하다. task 변환 규약은 같은 파일의 `transform_targets`를 내부에서 호출하며, `task_spec`은 `get_task_spec(task)` 결과를 보관한다. 데이터 파이프라인에서 `MNISTDataset`은 `load_mnist` 다음, `Dataloader` 이전 단계에 위치한다.

**목표**
- `load_mnist`의 원본 배열을 `(N, 784)` `float32`, 범위 `[0, 1]`로 정규화한다.
- task에 따라 레이블을 `multiclass` one-hot, `binary` 이진화, `regression` 정규화로 변환한다.
- `__len__`과 `__getitem__`을 구현하여 `Dataloader`와 연동한다.

## 2. 개념

### 2.1. Dataset 프로토콜

딥러닝 프레임워크에서 Dataset은 `__len__`과 `__getitem__` 두 메서드를 구현한 객체를 가리킨다. `__len__`은 전체 샘플 수를 반환하고, `__getitem__`은 인덱스를 받아 단일 `(image, target)` tuple을 반환한다. `Dataloader`는 이 인터페이스만 요구하므로 MNIST 외의 데이터셋도 같은 `Dataloader`와 조합할 수 있다.

이 프로토콜의 핵심 용어는 다음과 같다.

| 용어 | 의미 | 이 프로젝트에서의 역할 |
|---|---|---|
| `__len__` | 전체 샘플 수 반환 | `Dataloader`가 배치 수 계산에 사용 |
| `__getitem__(idx)` | 인덱스로 단일 샘플 반환 | `Dataloader`가 배치 조립에 사용 |
| `task_spec` | task별 스펙 dict | Trainer, Evaluator, Predictor가 task 분기에 사용 |

### 2.2. task별 target 변환

MNIST 레이블은 0부터 9까지의 정수이다. task에 따라 이 레이블을 다른 형태로 변환해야 모델이 올바른 목적 함수를 학습할 수 있다.

| task | 변환 방식 | 출력 shape | 범위 |
|---|---|---|---|
| `multiclass` | one-hot 인코딩 | `(N, 10)` | 0.0 또는 1.0 |
| `binary` | 홀수/짝수 이진화 (`labels % 2`) | `(N, 1)` | 0.0 또는 1.0 |
| `regression` | 레이블 정규화 (`labels / 9.0`) | `(N, 1)` | `[0.0, 1.0]` |

`binary` 변환에서 홀수 레이블(1, 3, 5, 7, 9)은 1.0, 짝수 레이블(0, 2, 4, 6, 8)은 0.0으로 변환한다. `regression`에서 레이블 9는 1.0, 레이블 0은 0.0으로 정규화한다.

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `MNISTDataset` | 클래스 | `split`, `task`, `dataset_dir` | dataset instance | 정규화 및 target 변환 포함 Dataset |
| `transform_targets` | 함수 | `labels: ndarray`, `task: str` | `ndarray` | task별 target 변환 |
| `get_task_spec` | 함수 | `task: str` | `dict` | task별 스펙 dict 반환 |

### 3.1. MNISTDataset

`MNISTDataset`은 생성자에서 `load_mnist`를 호출하여 정규화와 target 변환을 완료한 뒤 `self.images`, `self.targets`, `self.task_spec`에 저장한다.

```python
class MNISTDataset:
    def __init__(self, split, task, dataset_dir=None):
        images, labels = load_mnist(split, dataset_dir=dataset_dir)
        self.images = images.reshape(-1, 784).astype(np.float32) / 255.0
        self.targets = transform_targets(labels, task)
        self.task_spec = get_task_spec(task)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]
```

`images.reshape(-1, 784)`는 `(N, 28, 28)` 원본을 MLP 입력 형태인 `(N, 784)`로 펼친다. `/255.0`과 `.astype(np.float32)` 적용으로 픽셀 범위를 `[0, 1]`로 정규화한다.

### 3.2. transform_targets

`transform_targets`는 task에 따라 레이블 배열을 변환한다.

```python
def transform_targets(labels, task):
    if task == "multiclass":
        n = len(labels)
        targets = np.zeros((n, 10), dtype=np.float32)
        targets[np.arange(n), labels] = 1.0
        return targets
    elif task == "binary":
        return (labels % 2).astype(np.float32).reshape(-1, 1)
    else:  # regression
        return labels.astype(np.float32).reshape(-1, 1) / 9.0
```

`multiclass` one-hot 변환에서 `targets[np.arange(n), labels] = 1.0`은 각 행의 레이블 위치에 1.0을 대입하는 fancy indexing이다. 나머지 위치는 `np.zeros`의 0.0 초기값이 유지된다.

## 4. 사용법

최소 사용 예제는 다음과 같다.

```python
from src.data.mnist import MNISTDataset

ds = MNISTDataset("train", "multiclass")
print(len(ds))
image, target = ds[0]
print(image.shape, target.shape)
```

예상 출력은 다음과 같다.

```text
60000
(784,) (10,)
```

프로젝트 통합 예제는 다음과 같다. task별 Dataset을 생성하고 `Dataloader`와 조합한다.

```python
from src.data.mnist import MNISTDataset
from src.data.dataloader import Dataloader

for task in ("multiclass", "binary", "regression"):
    ds = MNISTDataset("train", task)
    loader = Dataloader(ds, batch_size=64, shuffle=True)
    for images, targets in loader:
        print(task, images.shape, targets.shape)
        break
```

## 5. 테스트

테스트 파일은 `tests/stage2/test_dataset.py`이다.

```bash
conda run -n numpy_py311 pytest tests/stage2/test_dataset.py -v
```

테스트 구성은 다음과 같다.

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestMNISTDatasetLen` | 2 | train/test split 샘플 수 |
| `TestMNISTDatasetImages` | 3 | shape `(N, 784)`, dtype `float32`, 범위 `[0, 1]` |
| `TestMNISTDatasetTargetsMulticlass` | 3 | shape `(N, 10)`, dtype, one-hot 합 검증 |
| `TestMNISTDatasetTargetsBinary` | 4 | shape `(N, 1)`, dtype, 값 범위, 홀수=1 검증 |
| `TestMNISTDatasetTargetsRegression` | 4 | shape `(N, 1)`, dtype, 값 범위, 경계값 검증 |
| `TestMNISTDatasetGetitem` | 4 | tuple 반환, image/target shape |
| `TestMNISTDatasetTaskSpec` | 4 | 필수 키 존재, task별 `output_dim`, `prediction_mode` |
| `TestMNISTDatasetErrors` | 2 | 잘못된 split/task `ValueError` |
| `TestMNISTDatasetReal` | 2 | 실제 MNIST train/test shape 검증 (파일 없으면 skip) |

단위 테스트는 `tmp_path` fixture로 합성 gz 파일(n=20)을 생성하여 실제 MNIST 파일에 의존하지 않는다.

## 6. 요약

`MNISTDataset`은 `load_mnist` 원본 배열을 `(N, 784)` `float32`로 정규화하고, task에 따라 one-hot, 이진화, 정규화 중 하나로 target을 변환한다. `__len__`과 `__getitem__` 프로토콜로 `Dataloader`와 연동하며, `task_spec`으로 Trainer, Evaluator 등 실행 객체에 task 정보를 전달한다.

다음 Phase에서는 [[phase2.3_dataloader]]를 다룬다.
