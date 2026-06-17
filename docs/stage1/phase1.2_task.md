---
tags: [stage1, task]
created: 2026-06-17
updated: 2026-06-17
---

# Phase 1.2 task

## 1. 역할

`src/task.py`는 multiclass, binary, regression 세 가지 과제 유형에 대한 규약을 단일 진입점으로 관리한다.
후속 PyTorch, TensorFlow, JAX 프로젝트에서도 동일한 함수명과 반환 구조를 유지한다.

## 2. 구현

### 2.1. get_task_spec(task)

`get_task_spec(task)`는 task 문자열을 받아 해당 과제의 규약 정보를 `dict`로 반환한다.
유효하지 않은 task 값이 입력되면 `ValueError`를 발생시킨다.

반환 dict의 최소 포함 키는 다음과 같다.

| 키 | 타입 | 설명 |
|---|---|---|
| `task` | `str` | 과제 유형 식별자 |
| `output_dim` | `int` | 모델 출력 차원 |
| `target_dtype` | `str` | target 배열의 dtype 이름 |
| `prediction_mode` | `str` | 예측 후처리 방식 식별자 |

task별 반환값은 다음과 같다.

| task | `output_dim` | `prediction_mode` |
|---|---|---|
| `"multiclass"` | `10` | `"argmax"` |
| `"binary"` | `1` | `"threshold"` |
| `"regression"` | `1` | `"round_clip"` |

### 2.2. transform_targets(labels, task)

`transform_targets(labels, task)`는 원본 MNIST 레이블 배열(`uint8`, shape `(N,)`)을 받아 task별 규약에 맞는 target 배열을 반환한다.
유효하지 않은 task 값이 입력되면 `ValueError`를 발생시킨다.

task별 변환 규약은 다음과 같다.

| task | 변환 방식 | 출력 shape | dtype |
|---|---|---|---|
| `"multiclass"` | one-hot 인코딩 | `(N, 10)` | `float32` |
| `"binary"` | 홀수→1 / 짝수→0 | `(N, 1)` | `float32` |
| `"regression"` | `label / 9.0` 정규화 | `(N, 1)` | `float32` |

## 3. 테스트

`tests/stage1/test_task.py`에서 다음 항목을 검증한다.

`get_task_spec` 검증 항목은 다음과 같다.

- 반환 타입이 `dict`인지
- 필수 키 4개가 모두 포함되어 있는지
- task별 `output_dim`과 `prediction_mode` 값이 규약과 일치하는지
- 유효하지 않은 task 입력 시 `ValueError`가 발생하는지

`transform_targets` 검증 항목은 다음과 같다.

- task별 출력 shape과 dtype이 규약과 일치하는지
- multiclass one-hot 인코딩의 정확성
- binary 홀수/짝수 변환의 정확성
- regression 정규화 값의 정확성 (부동소수점 허용 오차 적용)
- 유효하지 않은 task 입력 시 `ValueError`가 발생하는지

테스트 실행 명령은 다음과 같다.

```bash
conda run -n numpy_env pytest tests/stage1/test_task.py -v
```

```text
[Expected output]
tests/stage1/test_task.py::test_get_task_spec_returns_dict PASSED
tests/stage1/test_task.py::test_get_task_spec_required_keys PASSED
tests/stage1/test_task.py::test_get_task_spec_multiclass PASSED
tests/stage1/test_task.py::test_get_task_spec_binary PASSED
tests/stage1/test_task.py::test_get_task_spec_regression PASSED
tests/stage1/test_task.py::test_get_task_spec_invalid_raises PASSED
tests/stage1/test_task.py::test_transform_targets_multiclass_shape PASSED
tests/stage1/test_task.py::test_transform_targets_multiclass_dtype PASSED
tests/stage1/test_task.py::test_transform_targets_multiclass_one_hot PASSED
tests/stage1/test_task.py::test_transform_targets_binary_shape PASSED
tests/stage1/test_task.py::test_transform_targets_binary_dtype PASSED
tests/stage1/test_task.py::test_transform_targets_binary_values PASSED
tests/stage1/test_task.py::test_transform_targets_regression_shape PASSED
tests/stage1/test_task.py::test_transform_targets_regression_dtype PASSED
tests/stage1/test_task.py::test_transform_targets_regression_values PASSED
tests/stage1/test_task.py::test_transform_targets_invalid_raises PASSED
16 passed in 0.25s
```
