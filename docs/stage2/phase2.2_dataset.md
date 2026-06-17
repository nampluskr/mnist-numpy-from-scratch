---
tags: [stage2, data, dataset]
created: 2026-06-17
updated: 2026-06-17
---

# Phase 2.2 Dataset 클래스 구현

## 1. 역할

`MnistDataset`은 `load_mnist`로 로드한 원본 배열에 정규화와 task별 target 변환을 적용하여 배치 이터레이터(`DataLoader`)가 소비할 수 있는 Dataset 인터페이스를 제공한다.
task 변환 규약은 `src/task.py`의 `transform_targets`를 내부에서 호출하며, `task_spec`은 `get_task_spec(task)` 결과를 보관한다.

## 2. 구현

### 2.1. MnistDataset(split, task, dataset_dir=None)

`MnistDataset`은 `__len__`과 `__getitem__`을 구현한 Dataset 프로토콜을 따른다.

입출력 규약은 다음과 같다.

| 항목 | 값 |
|---|---|
| 입력 `split` | `"train"` 또는 `"test"` - 그 외 값은 `ValueError` 발생 |
| 입력 `task` | `"multiclass"`, `"binary"`, `"regression"` - 그 외 값은 `ValueError` 발생 |
| 입력 `dataset_dir` | 기본값 `None` - `None`이면 `get_default_config()["dataset_dir"]` 사용 |
| `self.images` | shape `(N, 784)`, dtype `float32`, 범위 `[0, 1]` |
| `self.targets` | task별 shape, dtype `float32` |
| `self.task_spec` | `get_task_spec(task)` 결과 dict |

### 2.2. task별 target 변환

| task | 변환 | shape | 비고 |
|---|---|---|---|
| `multiclass` | one-hot | `(N, 10)` | 레이블 → one-hot, 값은 0.0 또는 1.0 |
| `binary` | 홀수/짝수 이진화 | `(N, 1)` | `labels % 2` - 홀수=1, 짝수=0 |
| `regression` | 정규화 | `(N, 1)` | `labels / 9.0`, 범위 `[0.0, 1.0]` |

### 2.3. 인터페이스

```python
ds = MnistDataset("train", "multiclass")

len(ds)           # 샘플 수
ds[0]             # (image, target) tuple - image: (784,), target: (10,)
ds.images         # (N, 784) float32
ds.targets        # (N, output_dim) float32
ds.task_spec      # dict - output_dim, prediction_mode 등
```

## 3. 테스트

테스트 파일: `tests/stage2/test_dataset.py`

synthetic gz 파일(n=20, labels=0..9 반복)을 `tmp_path` fixture로 생성하여 실제 데이터셋 의존을 최소화했다.
실제 MNIST 의존 테스트는 `pytest.mark.skipif`로 분리했다.

| 테스트 그룹 | 항목 수 | 내용 |
|---|---|---|
| `__len__` | 2 | train/test split 샘플 수 |
| images | 3 | shape, dtype, 정규화 범위 |
| targets (multiclass) | 3 | shape, dtype, one-hot 검증 |
| targets (binary) | 4 | shape, dtype, 값 범위, 홀수=1 검증 |
| targets (regression) | 4 | shape, dtype, 값 범위, 경계값 검증 |
| `__getitem__` | 4 | tuple 반환, image/target shape |
| `task_spec` | 4 | 필수 키, task별 output_dim·prediction_mode |
| error handling | 2 | 잘못된 split/task `ValueError` |
| real data | 2 | 실제 MNIST train/test shape (skipif) |

실행 명령:

```bash
conda run -n numpy_env pytest tests/stage2/test_dataset.py -v
```

## 4. 설계 결정

- `MnistDataset`은 `load_mnist` 위에서 전처리 책임만 담당하며, `load_mnist`는 Phase 2.1 그대로 유지한다.
- task 변환 로직은 `task.py`의 `transform_targets`를 재사용하여 중복을 제거했다.
- binary 변환 기준(홀수=1/짝수=0)은 MNIST 전용이며, 다른 데이터셋(`FashionMnistDataset` 등)은 각자 `_transform`을 정의한다.
- `dataset_dir=None` 기본값을 유지하여 `DataLoader`와의 조합에서 외부 경로 주입이 가능하다.
