---
tags: [stage2, data, mnist]
created: 2026-06-17
updated: 2026-06-17
---

# Phase 2.1 MNIST raw data loading

## 1. 역할

`src/data/mnist.py`는 로컬에 저장된 MNIST gz 파일 4개를 읽어 원본 배열을 반환하는 단일 진입점을 제공한다.
정규화, target 변환 등 전처리는 이 파일 범위 밖이며, 각각 호출 측과 `src/task.py`가 담당한다.

## 2. 구현

### 2.1. load_mnist(split, dataset_dir=None)

`load_mnist`는 split에 해당하는 gz 파일 2개를 읽어 `(images, labels)` tuple을 반환한다.

입출력 규약은 다음과 같다.

| 항목 | 값 |
|---|---|
| 입력 `split` | `"train"` 또는 `"test"` - 그 외 값은 `ValueError` 발생 |
| 입력 `dataset_dir` | 기본값 `None` - `None`이면 `get_default_config()["dataset_dir"]` 사용 |
| 출력 `images` | shape `(N, 28, 28)`, dtype `uint8` - 정규화 없는 원본 픽셀값 |
| 출력 `labels` | shape `(N,)`, dtype `uint8` - 원본 클래스 레이블 (0–9) |

split별 파일 매핑은 다음과 같다.

| split | image 파일 | label 파일 |
|---|---|---|
| `"train"` | `train-images-idx3-ubyte.gz` | `train-labels-idx1-ubyte.gz` |
| `"test"` | `t10k-images-idx3-ubyte.gz` | `t10k-labels-idx1-ubyte.gz` |

### 2.2. MNIST 바이너리 포맷

MNIST gz 파일은 IDX 바이너리 포맷을 사용하며, 헤더와 데이터로 구성된다.

image 파일 헤더는 big-endian 32-bit 정수 4개(`magic`, `n`, `rows`, `cols`)로 시작하며, 이후 `n * rows * cols` 바이트의 uint8 픽셀 데이터가 이어진다.

label 파일 헤더는 big-endian 32-bit 정수 2개(`magic`, `n`)로 시작하며, 이후 `n` 바이트의 uint8 레이블 데이터가 이어진다.

## 3. 테스트

`tests/stage2/test_mnist.py`에서 다음 항목을 검증한다.

단위 테스트는 `tmp_path` fixture로 합성 gz 파일을 생성하여 실제 MNIST 파일에 의존하지 않고 실행한다.

| 테스트 | 검증 내용 |
|---|---|
| `test_train_images_shape` | `"train"` split image shape `(N, 28, 28)` |
| `test_train_labels_shape` | `"train"` split labels shape `(N,)` |
| `test_images_dtype_uint8` | image dtype `uint8` |
| `test_labels_dtype_uint8` | label dtype `uint8` |
| `test_test_split_shape` | `"test"` split shape |
| `test_image_pixel_range` | 픽셀값 범위 0–255 |
| `test_label_values` | 레이블값 0–9 |
| `test_invalid_split_raises` | 잘못된 split → `ValueError` |
| `test_missing_file_raises` | 파일 없음 → `FileNotFoundError` |
| `test_real_train_shape` | 실제 데이터 train shape `(60000, 28, 28)` |
| `test_real_test_shape` | 실제 데이터 test shape `(10000, 28, 28)` |
| `test_real_train_dtypes` | 실제 데이터 dtype |
| `test_real_label_range` | 실제 데이터 레이블 범위 0–9 |

실 MNIST 데이터 의존 테스트(`test_real_*`)는 `/mnt/d/datasets/mnist` 경로가 없으면 skip 처리된다.

테스트 실행 명령은 다음과 같다.

```bash
conda run -n numpy_env pytest tests/stage2/test_mnist.py -v
```

```text
[Expected output]
13 passed in 2.44s
```
