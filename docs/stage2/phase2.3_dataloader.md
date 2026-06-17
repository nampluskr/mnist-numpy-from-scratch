---
tags: [stage2, data, dataloader]
created: 2026-06-17
updated: 2026-06-17
---

# Phase 2.3 dataloader

## 1. 역할

`DataLoader`는 `__len__`과 `__getitem__`을 구현한 Dataset을 받아 mini-batch 단위로 순회하는 범용 이터레이터를 제공한다.
MNIST 전용이 아니며, 같은 프로토콜을 따르는 모든 Dataset과 함께 동작한다.

## 2. 구현

### 2.1. DataLoader(dataset, batch_size, shuffle=False)

입출력 규약은 다음과 같다.

| 항목 | 값 |
|---|---|
| 입력 `dataset` | `__len__`과 `__getitem__`을 구현한 Dataset 인스턴스 |
| 입력 `batch_size` | 배치 크기 (`int`) |
| 입력 `shuffle` | `True`이면 매 iteration마다 index를 무작위 순서로 재배치 (기본값 `False`) |
| `__len__` | `ceil(len(dataset) / batch_size)` |
| `__iter__` yield | `(images_batch, targets_batch)` tuple — shape은 Dataset `__getitem__` 반환값에 따라 결정 |

마지막 배치는 `batch_size`보다 작을 수 있다 (`len(dataset) % batch_size != 0` 인 경우).

### 2.2. 인터페이스

```python
from src.data.mnist import MnistDataset
from src.data.dataloader import DataLoader

ds = MnistDataset("train", "multiclass")
loader = DataLoader(ds, batch_size=32, shuffle=True)

len(loader)                          # 배치 수 — ceil(60000 / 32)

for images, targets in loader:
    # images: (batch_size, 784) float32
    # targets: (batch_size, 10) float32
    ...
```

## 3. 테스트

테스트 파일: `tests/stage2/test_dataloader.py`

MNIST 의존 없이 `ToyDataset`(n=20, feature\_dim=4) synthetic 데이터를 사용하여 DataLoader의 범용성을 검증했다.

| 테스트 그룹 | 항목 수 | 내용 |
|---|---|---|
| `__len__` | 3 | 나누어떨어지는 경우, 나머지 있는 경우, batch\_size > n 경우 |
| `__iter__` 형태 | 6 | tuple 반환, batch image/target shape, 배치 수, 마지막 배치 크기, 전체 샘플 수 |
| 순서 유지 (no shuffle) | 1 | 원본 순서와 동일 |
| 셔플 | 2 | 순서 변경, 전체 샘플 포함 |
| 반복 독립성 | 1 | 두 번의 iter가 서로 다른 순서 |

실행 명령:

```bash
conda run -n numpy_env pytest tests/stage2/test_dataloader.py -v
```

## 4. 설계 결정

- `__iter__` 내부에서 `np.random.permutation`과 `np.arange`로 index 배열을 생성하여 슬라이싱한다. Dataset 원본 배열을 직접 셔플하지 않는다.
- `np.stack`으로 샘플을 배치 배열로 조립하므로 Dataset이 반환하는 배열 형태만 맞으면 MNIST 외 Dataset도 수용한다.
- 마지막 배치 drop 로직은 포함하지 않는다. 필요 시 호출부에서 처리한다.
