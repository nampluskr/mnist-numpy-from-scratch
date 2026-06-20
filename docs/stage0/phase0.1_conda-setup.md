---
tags: [docs, stage0, conda-setup]
created: "2026-06-20"
updated: "2026-06-20"
---

# conda 실행 환경 구성

## 1. 개요

이 프로젝트는 MLP와 CNN을 각각 다른 실행 환경에서 학습한다. MLP는 CPU 기반 NumPy 환경에서 실행하고, CNN은 GPU 기반 CuPy 환경에서 실행한다. CuPy는 CUDA Toolkit 버전에 따라 설치 패키지가 달라지므로 두 환경을 별도로 구성한다. conda 환경 3개를 생성하고 NumPy, CuPy, GPU 연산이 정상 동작하는지 각각 검증한다.

**목표**
- `numpy_py311` CPU 기반 MLP 실행 환경을 생성하고 NumPy 동작을 확인한다.
- `cupy_py311_cuda118` CUDA 11.8 기반 CNN 실행 환경을 생성하고 CuPy GPU 연산을 확인한다.
- `cupy_py311_cuda121` CUDA 12 계열 기반 CNN 실행 환경을 생성하고 CuPy GPU 연산을 확인한다.

## 2. 개념

### 2.1. NumPy와 CuPy의 역할 분리

NumPy는 CPU에서 동작하는 n차원 배열 라이브러리이다. MLP는 연산량이 상대적으로 적어 CPU에서도 충분히 학습할 수 있다. CuPy는 NumPy와 동일한 API를 GPU에서 실행하는 라이브러리이다. CNN은 im2col 기반 합성곱 연산의 반복 횟수가 많아 GPU 병렬 연산이 필수적이다.

이 프로젝트에서 모델별 실행 환경은 다음과 같이 분리한다.

| 모델 | 환경 | 이유 |
|---|---|---|
| MLP | `numpy_py311` | CPU NumPy로 충분, 모든 task 학습 가능 |
| CNN | `cupy_py311_cuda118` 또는 `cupy_py311_cuda121` | im2col 합성곱 연산은 GPU 필수 |

### 2.2. CUDA Toolkit 버전과 CuPy 패키지

CuPy는 CUDA Toolkit 버전에 맞는 패키지를 별도로 설치해야 한다. 설치할 패키지명은 아래와 같다.

| CUDA 버전 | PyPI 패키지명 |
|---|---|
| CUDA 11.x 계열 | `cupy-cuda11x` |
| CUDA 12.x 계열 | `cupy-cuda12x` |

`cupy-cuda118`이나 `cupy-cuda121`과 같이 마이너 버전을 지정한 패키지는 PyPI에 존재하지 않는다. CUDA 11.x 전체를 지원하는 `cupy-cuda11x`, CUDA 12.x 전체를 지원하는 `cupy-cuda12x`를 사용한다.

## 3. 구현

### 3.1. 환경 구성 확인

3개 conda 환경의 구성 결과는 아래와 같다.

| 환경명 | Python | NumPy | CuPy | CUDA |
|---|---|---|---|---|
| `numpy_py311` | 3.11 | 2.4.6 | - | - |
| `cupy_py311_cuda118` | 3.11 | 2.4.6 | 13.6.0 | 11.8 |
| `cupy_py311_cuda121` | 3.11 | 2.4.6 | 14.1.1 | 12.9 |

### 3.2. 환경 생성 명령

`numpy_py311` 환경 생성과 패키지 설치는 다음과 같다.

```bash
conda create -n numpy_py311 python=3.11 -y
conda run -n numpy_py311 pip install numpy matplotlib pytest
```

`cupy_py311_cuda118` 환경 생성과 패키지 설치는 다음과 같다.

```bash
conda create -n cupy_py311_cuda118 python=3.11 -y
conda run -n cupy_py311_cuda118 pip install numpy matplotlib pytest
conda run -n cupy_py311_cuda118 pip install cupy-cuda11x
```

`cupy_py311_cuda121` 환경 생성과 패키지 설치는 다음과 같다.

```bash
conda create -n cupy_py311_cuda121 python=3.11 -y
conda run -n cupy_py311_cuda121 pip install numpy matplotlib pytest
conda run -n cupy_py311_cuda121 pip install cupy-cuda12x
```

### 3.3. 명령 실행 형식

이 프로젝트의 모든 Python 명령은 `conda run -n {환경명}` 형식으로 실행한다. conda 환경을 activate하지 않고 직접 지정하므로, 어느 터미널 상태에서도 동일한 환경에서 실행된다.

```bash
# 올바른 실행 형식
conda run -n numpy_py311 python scripts/train.py

# 올바른 테스트 실행 형식
conda run -n numpy_py311 pytest tests/ -q
```

## 4. 사용법

각 환경의 정상 동작은 아래 검증 명령으로 확인한다.

`numpy_py311` 환경 검증은 다음과 같다.

```bash
conda run -n numpy_py311 python -c "
import numpy as np
import matplotlib
import pytest
print('numpy:', np.__version__)
print('matplotlib:', matplotlib.__version__)
print('pytest:', pytest.__version__)
a = np.array([1, 2, 3])
print('array:', a)
"
```

예상 출력은 다음과 같다.

```text
numpy: 2.4.6
matplotlib: 3.11.0
pytest: 9.1.0
array: [1 2 3]
```

`cupy_py311_cuda118` 환경 검증은 다음과 같다.

```bash
conda run -n cupy_py311_cuda118 python -c "
import cupy as cp
print('cupy:', cp.__version__)
print('CUDA:', cp.cuda.runtime.runtimeGetVersion())
a = cp.array([1, 2, 3])
print('GPU array:', a)
"
```

예상 출력은 다음과 같다.

```text
cupy: 13.6.0
CUDA: 11080
GPU array: [1 2 3]
```

`CUDA: 11080`은 CUDA 11.8.0을 의미한다. 마지막 두 자리 `80`이 마이너 버전 8.0에 해당한다.

`cupy_py311_cuda121` 환경 검증은 다음과 같다.

```bash
conda run -n cupy_py311_cuda121 python -c "
import cupy as cp
print('cupy:', cp.__version__)
print('CUDA:', cp.cuda.runtime.runtimeGetVersion())
a = cp.array([1, 2, 3])
print('GPU array:', a)
"
```

예상 출력은 다음과 같다.

```text
cupy: 14.1.1
CUDA: 12090
GPU array: [1 2 3]
```

## 5. 테스트

Phase 0.1은 환경 구성 단계이므로 대응하는 테스트 파일이 없다. 아래 명령으로 각 환경에서 패키지 import가 정상인지 확인한다.

```bash
conda run -n numpy_py311 python -c "import numpy, matplotlib, pytest; print('OK')"
conda run -n cupy_py311_cuda118 python -c "import cupy; a = cupy.array([1]); print('OK')"
conda run -n cupy_py311_cuda121 python -c "import cupy; a = cupy.array([1]); print('OK')"
```

## 6. 요약

3개 conda 환경(`numpy_py311`, `cupy_py311_cuda118`, `cupy_py311_cuda121`)이 생성되어 정상 동작이 확인되었다. MLP는 `numpy_py311`에서, CNN은 두 CuPy 환경 중 시스템에 설치된 CUDA Toolkit 버전에 맞는 환경에서 실행한다. 모든 Python 명령은 `conda run -n {환경명}` 형식으로 실행하며, 이 규칙은 Stage 1부터 Stage 6까지 일관되게 유지된다.

다음 Phase에서는 [[phase0.2_implementation-plan]]을 다룬다.
