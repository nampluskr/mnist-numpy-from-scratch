---
tags: [docs, stage2, dataloader, batching]
created: "2026-06-20"
updated: "2026-06-21"
---

# Dataloader와 mini-batch 순회

## 1. 개요

`src/data/dataloader.py`의 `Dataloader`는 `__len__`과 `__getitem__`을 구현한 Dataset을 받아 mini-batch 단위로 순회하는 범용 이터레이터를 제공한다. MNIST 전용이 아니며, 같은 프로토콜을 따르는 모든 Dataset과 함께 동작한다. 데이터 파이프라인에서 `Dataloader`는 Dataset 다음, Trainer/Evaluator 이전 단계에 위치하여 배치 단위 이미지와 target을 학습 루프로 공급한다.

**목표**
- Dataset을 mini-batch 단위로 순회하는 이터레이터를 제공한다.
- `shuffle=True` 옵션으로 매 epoch마다 데이터 순서를 무작위로 섞는다.
- Dataset 종류에 관계없이 `__len__`+`__getitem__` 프로토콜이면 수용한다.

## 2. 개념

### 2.1. mini-batch 이터레이터

학습 루프는 매 epoch마다 전체 데이터를 mini-batch 단위로 분할하여 처리한다. `Dataloader`는 Dataset 인덱스 배열을 생성하고, `batch_size` 단위로 슬라이싱하여 각 배치의 샘플을 `np.stack`으로 조립한다. Dataset 원본 배열을 복사하거나 셔플하지 않고, 인덱스 배열만 조작하므로 메모리 효율이 높다.

`Dataloader`의 핵심 용어는 다음과 같다.

| 용어 | 의미 | 이 프로젝트에서의 역할 |
|---|---|---|
| `batch_size` | 한 번에 처리하는 샘플 수 | `Dataloader`의 분할 단위 |
| `shuffle` | 매 iteration마다 인덱스 순서를 무작위로 재배치 | 학습 일반화를 위해 train split에 `True` 설정 |
| 마지막 배치 | 나머지 샘플로 구성된 배치 | `len(dataset) % batch_size != 0`일 때 발생 |

### 2.2. 인덱스 기반 배치 조립

`shuffle=True`이면 `np.random.permutation(n)`으로 0부터 n-1까지의 인덱스를 무작위 순서로 배열하고, `shuffle=False`이면 `np.arange(n)`으로 원본 순서를 유지한다. 이 인덱스 배열을 `batch_size` 단위로 슬라이싱하여 배치 인덱스를 얻고, `dataset[i]`를 호출하여 샘플을 수집한 뒤 `np.stack`으로 배치 배열을 조립한다.

마지막 배치는 `batch_size`보다 작을 수 있다. drop last 로직은 포함하지 않으며, 필요 시 호출부에서 처리한다.

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `Dataloader` | 클래스 | `dataset`, `batch_size`, `shuffle=False` | dataloader instance | mini-batch 이터레이터 |
| `__len__` | 메서드 | 없음 | `int` | `ceil(len(dataset) / batch_size)` |
| `__iter__` | 메서드 | 없음 | generator | `(images_batch, targets_batch)` tuple yield |

### 3.1. Dataloader 구현

```python
class Dataloader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        indices = np.random.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            images = np.stack([self.dataset[i][0] for i in batch_idx])
            targets = np.stack([self.dataset[i][1] for i in batch_idx])
            yield images, targets
```

`__len__`에서 `(n + batch_size - 1) // batch_size`는 `ceil(n / batch_size)`를 정수 연산으로 계산한다. 마지막 배치가 `batch_size`보다 작은 경우도 배치 수에 포함한다.

`__iter__`에서 `np.stack`은 `dataset[i][0]`의 배열 목록을 하나의 배치 배열로 조립한다. Dataset이 반환하는 배열 형태만 맞으면 MNIST 외 Dataset도 수용한다.

### 3.2. 셔플 동작

`shuffle=True`를 설정하면 `__iter__` 호출 시마다 새로운 `np.random.permutation`이 생성된다. 따라서 같은 `Dataloader` 인스턴스를 반복 사용하면 epoch마다 다른 순서로 배치가 생성된다. Dataset 원본 배열은 변경하지 않는다.

## 4. 사용법

최소 사용 예제는 다음과 같다.

```python
from src.data.datasets import MulticlassDataset
from src.data.dataloader import Dataloader

ds = MulticlassDataset("train")
loader = Dataloader(ds, batch_size=64, shuffle=True)

print(len(loader))

for images, targets in loader:
    print(images.shape, targets.shape)
    break
```

예상 출력은 다음과 같다.

```text
938
(64, 784) (64, 10)
```

프로젝트 통합 예제는 다음과 같다. Trainer 내부의 학습 루프에서 다음과 같이 사용한다.

```python
from src.data.datasets import MulticlassDataset
from src.data.dataloader import Dataloader

train_ds = MulticlassDataset("train")
test_ds = MulticlassDataset("test")

train_loader = Dataloader(train_ds, batch_size=64, shuffle=True)
test_loader = Dataloader(test_ds, batch_size=64, shuffle=False)

for epoch in range(num_epochs):
    for images, targets in train_loader:
        logits = model.forward(images)
        # loss, backward, optimizer.step() ...
```

## 5. 테스트

테스트 파일은 `tests/stage2/test_dataloader.py`이다.

```bash
conda run -n numpy_py311 pytest tests/stage2/test_dataloader.py -v
```

테스트 구성은 다음과 같다.

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestDataloaderLen` | 3 | 나누어떨어지는 경우, 나머지 있는 경우, batch_size > n 경우 |
| `TestDataloaderIter` | 6 | tuple 반환, batch image/target shape, 배치 수, 마지막 배치 크기, 전체 샘플 수 |
| `TestDataloaderOrder` | 1 | shuffle=False일 때 원본 순서 유지 |
| `TestDataloaderShuffle` | 2 | 순서 변경, 전체 샘플 포함 |
| `TestDataloaderIndependence` | 1 | 두 번의 iter가 서로 다른 순서 생성 |

단위 테스트는 MNIST 의존 없이 `ToyDataset`(n=20, feature_dim=4) 합성 데이터로 `Dataloader`의 범용성을 검증한다.

## 6. 요약

`Dataloader`는 Dataset 인덱스 배열을 `batch_size` 단위로 슬라이싱하고 `np.stack`으로 배치를 조립하는 범용 이터레이터이다. Dataset 원본을 변경하지 않고 인덱스만 조작하므로 모든 Dataset 타입을 수용한다. `shuffle=True` 설정 시 매 epoch마다 독립적인 무작위 순서를 생성한다.

다음 Stage에서는 [[phase3.1_activations]]를 다룬다.
