---
tags: [stage1, config]
created: 2026-06-17
updated: 2026-06-17
---

# Phase 1.1 config 구성

## 1. 역할

`src/config.py`는 프로젝트 전체에서 공유하는 기본 설정값을 단일 진입점으로 제공한다.
후속 PyTorch, TensorFlow, JAX 프로젝트에서도 동일한 함수명과 반환 구조를 유지한다.

## 2. 구현

### 2.1. get_default_config()

`get_default_config()`는 인자 없이 호출하며 `dict`를 반환한다.

반환 키와 기본값은 다음과 같다.

| 키 | 기본값 | 설명 |
|---|---|---|
| `dataset_dir` | `"/mnt/d/datasets/mnist"` | 로컬 MNIST gz 파일 경로 |
| `seed` | `42` | 난수 시드 |
| `batch_size` | `64` | 미니배치 크기 |
| `num_epochs` | `10` | 학습 epoch 수 |
| `task` | `"multiclass"` | 과제 유형 - `"multiclass"`, `"binary"`, `"regression"` 중 하나 |
| `split` | `"train"` | 데이터 split - `"train"` 또는 `"test"` |

## 3. 테스트

`tests/stage1/test_config.py`에서 다음 항목을 검증한다.

- 반환 타입이 `dict`인지
- 반환 키가 정확히 6개인지
- 각 키의 기본값이 규약과 일치하는지

테스트 실행 명령은 다음과 같다.

```bash
conda run -n numpy_py311 pytest tests/stage1/test_config.py -v
```

```text
[Expected output]
tests/stage1/test_config.py::TestGetDefaultConfig::test_returns_dict PASSED
tests/stage1/test_config.py::TestGetDefaultConfig::test_required_keys PASSED
tests/stage1/test_config.py::TestGetDefaultConfig::test_dataset_dir PASSED
tests/stage1/test_config.py::TestGetDefaultConfig::test_seed PASSED
tests/stage1/test_config.py::TestGetDefaultConfig::test_batch_size PASSED
tests/stage1/test_config.py::TestGetDefaultConfig::test_num_epochs PASSED
tests/stage1/test_config.py::TestGetDefaultConfig::test_task PASSED
tests/stage1/test_config.py::TestGetDefaultConfig::test_split PASSED
8 passed in 0.11s
```
