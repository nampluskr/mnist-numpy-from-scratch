---
tags: [docs, stage1, batching, random]
created: "2026-06-20"
updated: "2026-06-20"
---

# mini-batch 분할과 난수 시드

## 1. 개요

`src/utils/batching.py`와 `src/utils/random.py`는 학습 루프에서 매 epoch마다 반복되는 mini-batch 생성과 난수 시드 고정을 담당한다. `get_batches`는 하나 이상의 numpy 배열을 받아 지정된 크기의 mini-batch를 순서대로 yield한다. `set_seed`는 numpy 난수 시드를 고정하여 shuffle 순서와 파라미터 초기화 결과를 재현 가능하게 만든다. 두 함수는 Stage 2 이후 모든 학습 루프의 기반이 된다.

**목표**
- `get_batches`로 이미지 배열과 레이블 배열을 같은 인덱스로 mini-batch 분할한다.
- `shuffle=True` 옵션으로 매 epoch마다 데이터 순서를 무작위로 섞는다.
- `set_seed`로 실험 시작 전 난수 시드를 고정하여 동일한 조건에서 재현 가능한 결과를 얻는다.

## 2. 개념

### 2.1. mini-batch 학습

전체 훈련 데이터를 한 번에 forward/backward하면 메모리 부족이 발생할 수 있고, 파라미터 업데이트가 전체 평균에 수렴하여 학습이 느려진다. mini-batch 방식은 데이터를 작은 묶음으로 나누어 묶음 단위로 gradient를 계산하고 파라미터를 업데이트한다. batch size가 작을수록 gradient noise가 커져 일반화에 유리하지만, 수렴이 불안정해질 수 있다.

mini-batch 분할에서 중요한 점은 이미지 배열과 레이블 배열이 항상 같은 인덱스로 슬라이싱되어야 한다는 것이다. 인덱스가 어긋나면 이미지와 레이블이 쌍을 이루지 못해 학습이 망가진다.

이 프로젝트에서 mini-batch 관련 주요 용어는 다음과 같다.

| 용어 | 의미 | 이 프로젝트에서의 역할 |
|---|---|---|
| `batch_size` | 한 번에 처리하는 샘플 수 | `get_batches`의 분할 단위 |
| `shuffle` | 매 epoch마다 데이터 순서를 섞는 옵션 | 학습 일반화를 위해 `True`로 설정 |
| 마지막 배치 | 나머지 샘플로 구성된 배치 | 전체 샘플 수가 `batch_size`의 배수가 아닐 때 발생 |

### 2.2. 난수 시드와 재현성

딥러닝 학습에는 파라미터 초기화, mini-batch shuffle 등 난수를 사용하는 과정이 많다. 동일한 시드를 설정하면 이 과정들이 동일하게 재현되어 실험 결과를 비교할 수 있다. `np.random.seed(seed)`는 numpy 난수 생성기 상태를 초기화한다.

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `get_batches` | 함수 | `*arrays`, `batch_size`, `shuffle=False` | generator | mini-batch tuple을 순서대로 yield |
| `set_seed` | 함수 | `seed: int` | 없음 | numpy 난수 시드 고정 |

### 3.1. get_batches

`get_batches`는 전달된 배열 수에 따라 반환 형태가 달라진다. 배열이 1개이면 단일 `np.ndarray`를 yield하고, 2개 이상이면 같은 인덱스로 슬라이싱한 `tuple`을 yield한다.

```python
def get_batches(*arrays, batch_size, shuffle=False):
    n = len(arrays[0])
    indices = np.random.permutation(n) if shuffle else np.arange(n)

    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        batches = tuple(arr[idx] for arr in arrays)
        if len(batches) == 1:
            yield batches[0]
        else:
            yield batches
```

`np.random.permutation(n)`은 0부터 n-1까지의 정수를 무작위 순서로 배열한 인덱스를 반환한다. 이 인덱스를 모든 배열에 동일하게 적용하므로 이미지와 레이블의 쌍 관계가 유지된다.

### 3.2. set_seed

```python
def set_seed(seed):
    np.random.seed(seed)
```

학습 시작 전 한 번 호출하면 이후 모든 numpy 난수 생성이 동일한 순서를 따른다. 파라미터 초기화와 shuffle 순서가 고정되므로 같은 시드에서 동일한 학습 결과를 재현할 수 있다.

## 4. 사용법

최소 사용 예제는 다음과 같다.

```python
import numpy as np
from src.utils.batching import get_batches
from src.utils.random import set_seed

set_seed(42)

images = np.random.randn(100, 784).astype(np.float32)
labels = np.random.randint(0, 10, size=(100,))

for x_batch, y_batch in get_batches(images, labels, batch_size=32, shuffle=True):
    print(x_batch.shape, y_batch.shape)
    break
```

예상 출력은 다음과 같다.

```text
(32, 784) (32,)
```

프로젝트 통합 예제는 다음과 같다. Trainer 내부의 학습 루프에서 다음과 같이 사용한다.

```python
from src.utils.random import set_seed
from src.utils.batching import get_batches

set_seed(42)

for epoch in range(num_epochs):
    for x_batch, y_batch in get_batches(train_images, train_targets,
                                         batch_size=64, shuffle=True):
        logits = model.forward(x_batch)
        loss = loss_fn(logits, y_batch)
        model.backward(grad)
        optimizer.step()
```

## 5. 테스트

테스트 파일은 `tests/stage1/test_batching.py`와 `tests/stage1/test_random.py`이다.

```bash
conda run -n numpy_py311 pytest tests/stage1/test_batching.py tests/stage1/test_random.py -v
```

테스트 구성은 다음과 같다.

| 파일 | 주요 검증 내용 |
|---|---|
| `test_batching.py` | generator 반환 타입, 단일/복수 배열 yield 형태, batch size 상한, 마지막 배치 크기, 전체 샘플 수 보존, shuffle=False 순서 유지, shuffle=True 다중 배열 인덱스 일치 |
| `test_random.py` | 동일 시드 두 번 호출 시 동일 배열 생성, 다른 시드 호출 시 다른 배열 생성 |

## 6. 요약

`get_batches`는 이미지와 레이블 배열을 항상 같은 인덱스로 슬라이싱하여 mini-batch를 yield한다. `set_seed`는 numpy 난수 시드를 고정하여 실험 재현성을 확보한다. 두 함수는 Stage 2 이후 모든 학습 루프에서 epoch 반복의 기반으로 사용된다.

다음 Phase에서는 [[phase1.2_io-and-checkpoints]]을 다룬다.
