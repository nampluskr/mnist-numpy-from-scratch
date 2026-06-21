---
tags: [docs, stage2, mnist, data]
created: "2026-06-20"
updated: "2026-06-20"
---

# MNIST 원본 데이터 로딩

## 1. 개요

`src/data/mnist.py`의 `load_mnist`는 로컬에 저장된 MNIST gz 파일 4개를 파싱하여 원본 배열을 반환하는 단일 진입점이다. 정규화, target 변환 등 전처리는 이 함수의 책임 범위 밖이며, 각각 `MNISTDataset`과 `transform_targets`가 담당한다. 데이터 파이프라인에서 `load_mnist`는 가장 앞 단계에 위치하며, 이미지와 레이블 원본 배열을 다음 단계(`MNISTDataset`)로 넘긴다.

**목표**
- 로컬 MNIST gz 파일 4개를 파싱하여 `"train"` / `"test"` split을 선택한다.
- 이미지는 `(N, 28, 28)` `uint8`, 레이블은 `(N,)` `uint8` 원본 배열로 반환한다.
- 파일 없음, 잘못된 split 등 오류 조건을 명확한 예외로 처리한다.

## 2. 개념

### 2.1. MNIST 데이터셋

MNIST는 손으로 쓴 숫자 이미지 70,000장(train 60,000 + test 10,000)으로 구성된 표준 벤치마크 데이터셋이다. 각 이미지는 28x28 픽셀의 흑백 이미지이며, 레이블은 0부터 9까지의 정수이다. 원본 파일은 gz 압축된 IDX 바이너리 포맷 4개로 배포된다.

이 프로젝트에서 사용하는 로컬 파일 구성은 다음과 같다.

| 파일 | 설명 |
|---|---|
| `train-images-idx3-ubyte.gz` | 훈련 이미지 60,000장 |
| `train-labels-idx1-ubyte.gz` | 훈련 레이블 60,000개 |
| `t10k-images-idx3-ubyte.gz` | 테스트 이미지 10,000장 |
| `t10k-labels-idx1-ubyte.gz` | 테스트 레이블 10,000개 |

### 2.2. IDX 바이너리 포맷

IDX 포맷은 헤더와 데이터 영역으로 구성된 바이너리 직렬화 형식이다. 모든 정수는 big-endian 32-bit 부호 없는 정수(`">I"`)로 저장된다.

이미지 파일 헤더는 4개의 정수 필드로 구성된다.

| 필드 | 크기 | 의미 |
|---|---|---|
| `magic` | 4 bytes | 파일 타입 식별자 (`2051`) |
| `n` | 4 bytes | 이미지 수 |
| `rows` | 4 bytes | 이미지 높이 (28) |
| `cols` | 4 bytes | 이미지 너비 (28) |

레이블 파일 헤더는 2개의 정수 필드(`magic`, `n`)로 구성되며, 이후 `n` 바이트의 `uint8` 레이블 데이터가 이어진다.

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `load_mnist` | 함수 | `split: str`, `dataset_dir: str` | `(images, labels)` tuple | MNIST 원본 배열 로딩 |
| `get_task_spec` | 함수 | `task: str` | `dict` | task별 스펙 dict 반환 |
| `transform_targets` | 함수 | `labels: ndarray`, `task: str` | `ndarray` | task별 target 변환 |
| `MNISTDataset` | 클래스 | `split`, `task` | dataset instance | 정규화 및 target 변환 포함 |

### 3.1. load_mnist

`load_mnist`는 split에 해당하는 gz 파일 2개를 읽어 `(images, labels)` tuple을 반환한다.

```python
def load_mnist(split, dataset_dir=None):
    if split not in _SPLIT_FILES:
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")
    if dataset_dir is None:
        dataset_dir = _DATASET_DIR

    img_file, lbl_file = _SPLIT_FILES[split]
    images = _load_images(os.path.join(dataset_dir, img_file))
    labels = _load_labels(os.path.join(dataset_dir, lbl_file))
    return images, labels
```

split별 파일 매핑은 다음과 같다.

| split | image 파일 | label 파일 |
|---|---|---|
| `"train"` | `train-images-idx3-ubyte.gz` | `train-labels-idx1-ubyte.gz` |
| `"test"` | `t10k-images-idx3-ubyte.gz` | `t10k-labels-idx1-ubyte.gz` |

### 3.2. IDX 파싱 내부 함수

이미지 파싱은 `_load_images`, 레이블 파싱은 `_load_labels`가 담당한다. 두 함수 모두 파일이 없으면 `FileNotFoundError`를 발생시킨다.

```python
def _load_images(path):
    with gzip.open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)
```

`struct.unpack(">IIII", f.read(16))`는 big-endian 32-bit 정수 4개를 읽어 `(magic, n, rows, cols)` tuple로 반환한다. 헤더 이후의 바이트 전체를 `np.frombuffer`로 읽어 `(n, 28, 28)` 형태로 reshape한다.

## 4. 사용법

최소 사용 예제는 다음과 같다.

```python
from src.data.mnist import load_mnist

images, labels = load_mnist("train")
print(images.shape, images.dtype)
print(labels.shape, labels.dtype)
```

예상 출력은 다음과 같다.

```text
(60000, 28, 28) uint8
(60000,) uint8
```

프로젝트 통합 예제는 다음과 같다. `MNISTDataset`은 `load_mnist`를 내부에서 호출하여 정규화와 target 변환까지 처리한다.

```python
from src.data.mnist import MNISTDataset
from src.data.dataloader import Dataloader

dataset = MNISTDataset("train", "multiclass")
loader = Dataloader(dataset, batch_size=64, shuffle=True)

for images, targets in loader:
    print(images.shape, targets.shape)
    break
```

## 5. 테스트

테스트 파일은 `tests/stage2/test_mnist.py`이다.

```bash
conda run -n numpy_py311 pytest tests/stage2/test_mnist.py -v
```

테스트 구성은 다음과 같다.

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestLoadMnist` | 9 | shape, dtype, 픽셀 범위, 레이블 범위, 잘못된 split/파일 없음 예외 |
| `TestLoadMnistReal` | 4 | 실제 MNIST 파일 기반 shape, dtype 검증 (파일 없으면 skip) |

단위 테스트는 `tmp_path` fixture로 합성 gz 파일을 생성하여 실제 MNIST 파일에 의존하지 않고 실행한다. 실제 MNIST 의존 테스트(`TestLoadMnistReal`)는 `/mnt/d/datasets/mnist` 경로가 없으면 skip 처리된다.

## 6. 요약

`load_mnist`는 로컬 MNIST gz 파일을 IDX 바이너리 포맷으로 파싱하여 원본 `uint8` 배열을 반환한다. 정규화와 target 변환은 이 함수의 책임 범위 밖이며, `MNISTDataset`이 담당한다. 단위 테스트는 합성 데이터로 실행하여 실제 파일 의존을 최소화한다.

다음 Phase에서는 [[phase2.2_dataset]]을 다룬다.
