---
tags: [docs, stage1, io, checkpoints]
created: "2026-06-20"
updated: "2026-06-20"
---

# 파일 입출력과 모델 체크포인트

## 1. 개요

`src/utils/io.py`와 `src/utils/checkpoints.py`는 numpy 배열을 파일로 저장하고 복원하는 역할을 담당한다. `io.py`의 `save_params`와 `load_params`는 키-배열 dict를 `.npz` 파일로 직렬화하는 범용 도구이다. `checkpoints.py`의 `save`와 `load`는 모델의 `params` 리스트를 인덱스 기반으로 저장하고 in-place로 복원한다. CuPy 배열은 저장 시 자동으로 NumPy 배열로 변환되므로 MLP와 CNN 모두 동일한 인터페이스로 체크포인트를 관리한다.

**목표**
- numpy 배열 dict를 `.npz` 파일로 저장하고 동일한 구조의 dict로 복원한다.
- 모델의 `params` 리스트를 `param_0`, `param_1`, ... 키로 `.npz`에 저장하고 in-place로 복원한다.
- CuPy 배열을 NumPy로 자동 변환하여 저장하고, 복원 시 원래 배열 모듈로 되돌린다.

## 2. 개념

### 2.1. .npz 형식

numpy는 `.npy`와 `.npz` 두 가지 바이너리 형식을 제공한다. `.npy`는 배열 1개를 저장하고, `.npz`는 여러 배열을 키-값 쌍으로 묶어 단일 파일에 저장한다. 모델 파라미터는 weight와 bias가 레이어 수만큼 쌍을 이루므로 여러 배열을 묶어 저장하는 `.npz`가 적합하다.

`.npz` 파일의 주요 특성은 다음과 같다.

| 항목 | 내용 |
|---|---|
| 저장 API | `np.savez(path, **dict)` |
| 로드 API | `np.load(path)` - `NpzFile` 객체 반환 |
| 접근 방식 | `data["key"]` 키 기반 접근 |
| 확장자 | `.npz` - 없으면 자동 추가 |

### 2.2. 체크포인트와 모델 파라미터 구조

이 프로젝트의 모델은 `params` 속성에 레이어별 weight와 bias를 리스트로 보관한다. `Linear` 레이어 하나는 `w`와 `b` 2개의 배열을 갖는다. MLP가 3개의 `Linear`를 가지면 `params`에는 6개의 배열이 담긴다.

체크포인트는 이 리스트를 인덱스 기반으로 직렬화한다.

| params 인덱스 | 저장 키 |
|---|---|
| `params[0]` | `param_0` |
| `params[1]` | `param_1` |
| `params[n]` | `param_n` |

### 2.3. NumPy/CuPy 배열 상호 변환

CuPy 배열은 GPU 메모리에 상주한다. `.npz`는 CPU 메모리의 NumPy 배열만 저장할 수 있으므로 CuPy 배열은 저장 전 `.get()`으로 NumPy 배열로 변환해야 한다. 복원 시에는 파라미터가 현재 NumPy 배열인지 CuPy 배열인지 확인하여 동일한 모듈로 변환한다.

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `save_params` | 함수 | `params: dict`, `path: str` | 없음 | key-array dict를 .npz로 저장 |
| `load_params` | 함수 | `path: str` | `dict` | .npz를 key-array dict로 복원 |
| `checkpoints.save` | 함수 | `model`, `path: str` | 없음 | model.params 리스트를 .npz로 저장 |
| `checkpoints.load` | 함수 | `model`, `path: str` | 없음 | .npz에서 model.params를 in-place 복원 |

### 3.1. io.py - 범용 파라미터 저장·로드

`save_params`와 `load_params`는 키-배열 dict를 저장하고 동일한 구조로 복원하는 범용 함수이다.

```python
def save_params(params, path):
    np.savez(path, **params)

def load_params(path):
    data = np.load(path)
    return dict(data)
```

`np.savez`는 `path`에 `.npz` 확장자가 없으면 자동으로 추가한다. `np.load`가 반환하는 `NpzFile` 객체를 `dict()`로 변환하면 일반 파이썬 dict처럼 사용할 수 있다.

### 3.2. checkpoints.py - 모델 체크포인트

`save`는 `model.params`를 `param_0`, `param_1`, ... 키로 저장하고, `load`는 동일한 인덱스로 배열을 꺼내 `param[...] =` 연산으로 in-place 복원한다.

```python
def save(model, path):
    arrays = {f"param_{i}": _to_numpy(p) for i, p in enumerate(model.params)}
    np.savez(path, **arrays)

def load(model, path):
    npz_path = path if str(path).endswith(".npz") else str(path) + ".npz"
    data = np.load(npz_path)
    for i, param in enumerate(model.params):
        param[...] = _to_param_array(param, data[f"param_{i}"])
```

`param[...] =`은 배열 객체 자체를 교체하지 않고 메모리를 in-place로 덮어쓴다. 이 방식을 쓰는 이유는 모델 내부의 다른 속성이 동일한 배열 참조를 공유할 수 있기 때문이다.

CuPy 배열 감지는 다음 헬퍼 함수로 처리한다.

```python
def _to_numpy(array):
    return array.get() if hasattr(array, "get") else np.asarray(array)

def _to_param_array(param, array):
    module = type(param).__module__.split(".")[0]
    if module == "cupy":
        import cupy as cp
        return cp.asarray(array)
    return np.asarray(array)
```

`hasattr(array, "get")`으로 CuPy 배열 여부를 확인한다. `type(param).__module__`이 `"cupy"`로 시작하면 복원 시 CuPy 배열로 변환한다.

## 4. 사용법

`io.py` 최소 사용 예제는 다음과 같다.

```python
import numpy as np
from src.utils.io import save_params, load_params

params = {"w": np.array([[1.0, 2.0]]), "b": np.array([0.5])}
save_params(params, "outputs/params.npz")

loaded = load_params("outputs/params.npz")
print(loaded.keys())
print(loaded["w"])
```

예상 출력은 다음과 같다.

```text
dict_keys(['w', 'b'])
[[1. 2.]]
```

`checkpoints.py` 최소 사용 예제는 다음과 같다.

```python
from src.utils import checkpoints

# 학습 완료 후 저장
checkpoints.save(model, "outputs/mlp_multiclass.npz")

# 새 모델 인스턴스에 복원
new_model = MLP(task="multiclass")
checkpoints.load(new_model, "outputs/mlp_multiclass.npz")
```

프로젝트 통합 예제는 다음과 같다. 학습 스크립트에서는 저장하고, 평가 스크립트에서는 복원하는 흐름으로 사용한다.

```python
# scripts/train.py
from src.utils import checkpoints
trainer.fit(train_loader)
checkpoints.save(model, f"outputs/{exp_name}/model.npz")

# scripts/evaluate.py
from src.utils import checkpoints
checkpoints.load(model, f"outputs/{exp_name}/model.npz")
result = evaluator.evaluate(test_loader)
```

## 5. 테스트

테스트 파일은 `tests/stage1/test_io.py`와 `tests/stage1/test_checkpoints.py`이다.

```bash
conda run -n numpy_py311 pytest tests/stage1/test_io.py tests/stage1/test_checkpoints.py -v
```

테스트 구성은 다음과 같다.

| 파일 | 주요 검증 내용 |
|---|---|
| `test_io.py` | 저장 후 파일 존재 확인, 로드 반환 타입이 dict, 키 집합 일치, 배열 값 일치 |
| `test_checkpoints.py` | save 후 파일 존재 확인, load 후 params 값 일치, in-place 복원 확인, .npz 확장자 자동 처리 |

## 6. 요약

`io.py`는 키-배열 dict를 `.npz`로 저장하고 복원하는 범용 도구이고, `checkpoints.py`는 모델의 `params` 리스트를 인덱스 기반으로 저장하고 in-place로 복원하는 체크포인트 관리자이다. CuPy 배열은 저장과 복원 시 자동으로 NumPy와 상호 변환되므로 MLP와 CNN 모두 동일한 인터페이스를 사용한다.

다음 Phase에서는 [[phase1.3_training-plots]]을 다룬다.
