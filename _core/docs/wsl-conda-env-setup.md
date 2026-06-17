---
tags: [project, docs, DEV]
created: 2026-06-18
updated: 2026-06-18
---

# WSL Conda Environment Setup

이 문서는 WSL에서 conda 기반 Python, CUDA, deep learning framework 환경을 구성하는 기준을 정의한다.
이 프로젝트는 NumPy 기준 구현을 먼저 작성한 뒤 CuPy, PyTorch, TensorFlow, JAX 프로젝트로 이어지는 시리즈를 전제로 한다.

관련 프로젝트 범위는 [[PROJECT-SPEC]]를 기준으로 한다.

## 1. 문서 목적

이 문서의 목적은 초보자가 WSL에서 GPU 실험 환경을 만들 때 어떤 Python 버전, CUDA 버전, framework 버전을 선택해야 하는지 이해할 수 있도록 기준을 정리하는 것이다.

이 프로젝트에서는 다음 네 가지 framework를 CUDA 11.8 축과 CUDA 12 계열 축에서 평가한다.

- CuPy
- PyTorch
- TensorFlow
- JAX

NumPy는 GPU framework가 아니라 CPU 기반 기준 구현에 사용되는 공통 패키지이다.
따라서 NumPy 전용 CUDA 환경을 따로 만들 필요는 없다.
다만 NumPy 기준 구현을 가볍게 실행하기 위한 CPU 전용 conda 환경은 별도로 둘 수 있다.

## 2. WSL Conda 환경 전제

이 문서에서 말하는 WSL 환경은 Windows 위에서 Linux 사용자 공간을 실행하는 WSL2 환경을 의미한다.
GPU를 사용하려면 Windows에 NVIDIA driver가 설치되어 있어야 하며, WSL 안에서 `nvidia-smi` 명령이 정상적으로 동작해야 한다.

WSL conda 환경 구성의 기본 전제는 다음과 같다.

| 항목 | 기준 |
|---|---|
| 운영 환경 | WSL2 Linux |
| 환경 관리자 | conda |
| Python 버전 | Python 3.11 |
| CUDA 11 평가 축 | CUDA 11.8 |
| CUDA 12 평가 축 | CUDA 12.1 또는 CUDA 12.2 |
| 환경 분리 단위 | framework와 CUDA 버전 조합 |

하나의 conda 환경에 CuPy, PyTorch, TensorFlow, JAX를 모두 설치하지 않는다.
각 framework는 CUDA runtime, cuDNN, NCCL, XLA, protobuf, ml-dtypes, NumPy 버전 요구사항이 다르다.
한 환경에 모두 설치하면 처음에는 동작해도 이후 재설치, 업데이트, 재현 실험에서 충돌이 발생할 가능성이 높다.

## 3. 공통 Python 버전 기준

공통 Python 버전은 Python 3.11로 정한다.
Python 3.11은 CUDA 11.8 축과 CUDA 12 계열 축에서 네 framework를 함께 검토할 때 가장 안정적인 교집합이다.

Python 3.11을 선택하는 이유는 다음과 같다.

| Framework | Python 3.11 선택 이유 |
|---|---|
| CuPy | 최신 CuPy가 Python 3.10 이상을 지원한다. |
| PyTorch | CUDA 11.8 및 CUDA 12.1 조합에서 Python 3.11 wheel 사용이 가능하다. |
| TensorFlow | CUDA 11.8 기준 TensorFlow 2.14.0, CUDA 12.2 기준 TensorFlow 2.15.0 모두 Python 3.11을 지원한다. |
| JAX | CUDA 12 wheel과 최근 JAX 버전에서 Python 3.11이 지원된다. |

Python 3.12 이상은 TensorFlow 2.14.0과 TensorFlow 2.15.0의 CUDA 평가 축에서 제약이 생긴다.
Python 3.10은 가능하지만 시리즈 전체를 장기 운영할 때 Python 3.11보다 오래된 선택이 된다.
따라서 이 프로젝트의 기본 Python 버전은 Python 3.11로 고정한다.

## 4. CUDA 평가 축

이 프로젝트는 CUDA 11.8을 반드시 평가해야 하며, CUDA 12 계열은 framework별 공식 지원 조합을 고려하여 구성한다.

### 4.1. CUDA 11.8 평가 축

CUDA 11.8 축은 이전 세대 CUDA runtime에서 framework 동작을 확인하기 위한 필수 평가 축이다.
TensorFlow, PyTorch, CuPy는 CUDA 11.8 평가 기준을 비교적 명확하게 잡을 수 있다.

CUDA 11.8 축의 framework 기준은 다음과 같다.

| Framework | CUDA 기준 | 비고 |
|---|---:|---|
| CuPy | CUDA 11.8 | `cupy-cuda11x` 계열을 사용한다. |
| PyTorch | CUDA 11.8 | `pytorch-cuda=11.8` 또는 `cu118` wheel을 사용한다. |
| TensorFlow | CUDA 11.8 | `tensorflow==2.14.0` 기준을 사용한다. |
| JAX | CUDA 11.8 | CUDA 11을 지원하는 legacy JAX/JAXLIB 조합이 필요하다. |

JAX는 최신 버전에서 CUDA 11 지원이 기본 축이 아니다.
따라서 JAX CUDA 11.8 환경은 최신 설치 명령보다 legacy wheel과 version pinning 검토가 더 중요하다.

### 4.2. CUDA 12 계열 평가 축

CUDA 12 계열은 하나의 minor version으로 완전히 통일하기 어렵다.
이유는 TensorFlow가 공식 tested build에서 CUDA 12.2를 사용하고, PyTorch는 CUDA 12.1 또는 CUDA 12.4 runtime을 배포하며, JAX는 CUDA 12 wheel을 CUDA 12.3으로 빌드하면서 CUDA 12.1 이상과 호환된다고 설명하기 때문이다.

이 프로젝트에서는 실제 평가 환경 이름에 사용되는 CUDA version을 명시한다.

CUDA 12 계열의 framework 기준은 다음과 같다.

| Framework | 평가 CUDA 표기 | 실제 기준 |
|---|---:|---|
| CuPy | CUDA 12.1 | CUDA 12.x wheel 또는 conda package를 CUDA 12.1 환경에서 평가한다. |
| PyTorch | CUDA 12.1 | `pytorch-cuda=12.1` 또는 `cu121` wheel을 사용한다. |
| TensorFlow | CUDA 12.2 | `tensorflow==2.15.0`의 공식 tested build 기준을 따른다. |
| JAX | CUDA 12.1 | `jax[cuda12]`를 사용하되 CUDA 12.1 이상 호환 wheel임을 기록한다. |

TensorFlow만 CUDA 12.2를 사용하는 이유는 공식 tested build 조합 때문이다.
나머지 framework는 CUDA 12.1 환경을 기준으로 잡는 것이 PyTorch와 CuPy 평가에 가장 명확하다.

## 5. Framework별 환경 구성

각 framework 환경은 독립 conda 환경으로 만든다.
이 방식은 디스크 사용량이 늘어나지만 재현성과 문제 해결이 쉬워진다.

### 5.1. CuPy 환경

CuPy는 NumPy와 비슷한 배열 API를 GPU에서 실행하기 위한 framework이다.
이 프로젝트에서는 후속 GPU 기반 NumPy-like 구현을 비교할 때 CuPy를 사용한다.

CuPy 환경은 다음 두 개를 만든다.

| 환경명 | Python | CUDA 기준 | 용도 |
|---|---:|---:|---|
| `cupy_py311_cuda118` | 3.11 | 11.8 | CUDA 11.8 기반 CuPy 평가 |
| `cupy_py311_cuda121` | 3.11 | 12.1 | CUDA 12.1 기반 CuPy 평가 |

CuPy는 `cupy-cuda11x`, `cupy-cuda12x`처럼 package 이름에 minor version이 직접 드러나지 않는 경우가 있다.
그래도 환경명에는 실제 평가 대상 CUDA version을 적는다.

### 5.2. PyTorch 환경

PyTorch는 후속 PyTorch 프로젝트에서 동일한 MNIST 태스크를 구현하고 비교하기 위한 framework이다.
PyTorch는 CUDA runtime package를 `pytorch-cuda=11.8`, `pytorch-cuda=12.1`처럼 명시할 수 있다.

PyTorch 환경은 다음 두 개를 만든다.

| 환경명 | Python | CUDA 기준 | 용도 |
|---|---:|---:|---|
| `torch_py311_cuda118` | 3.11 | 11.8 | CUDA 11.8 기반 PyTorch 평가 |
| `torch_py311_cuda121` | 3.11 | 12.1 | CUDA 12.1 기반 PyTorch 평가 |

PyTorch는 CUDA 12.2 전용 runtime 이름보다 CUDA 12.1, CUDA 12.4, CUDA 12.6 같은 배포 축을 중심으로 제공된다.
따라서 CUDA 12 계열 평가에서는 `torch_py311_cuda121`을 기준으로 삼는다.

### 5.3. TensorFlow 환경

TensorFlow는 CUDA minor version 제약을 가장 강하게 받는 framework이다.
이 프로젝트에서는 TensorFlow의 공식 tested build 조합을 우선한다.

TensorFlow 환경은 다음 두 개를 만든다.

| 환경명 | Python | CUDA 기준 | TensorFlow 기준 |
|---|---:|---:|---|
| `tf_py311_cuda118` | 3.11 | 11.8 | `tensorflow==2.14.0` |
| `tf_py311_cuda122` | 3.11 | 12.2 | `tensorflow==2.15.0` |

TensorFlow CUDA 12 계열 환경은 다른 framework와 달리 CUDA 12.2를 사용한다.
이는 TensorFlow 2.15.0의 tested build가 CUDA 12.2와 cuDNN 8.9를 기준으로 하기 때문이다.

### 5.4. JAX 환경

JAX는 XLA 기반 array computation과 자동 미분을 제공하는 framework이다.
JAX는 `jax`와 `jaxlib` 조합이 중요하며, GPU 사용 시 CUDA wheel 종류를 확인해야 한다.

JAX 환경은 다음 두 개를 만든다.

| 환경명 | Python | CUDA 기준 | 설치 기준 |
|---|---:|---:|---|
| `jax_py311_cuda118` | 3.11 | 11.8 | CUDA 11 지원 legacy JAX/JAXLIB 조합 |
| `jax_py311_cuda121` | 3.11 | 12.1 | `jax[cuda12]` 기준 |

JAX CUDA 12 환경은 이름을 `jax_py311_cuda121`로 적는다.
다만 설치 package는 CUDA 12.1 전용 wheel이 아니라 `jax[cuda12]`이다.
JAX 문서는 CUDA 12 wheel이 CUDA 12.3으로 빌드되며 CUDA 12.1 이상과 호환된다고 설명한다.
따라서 문서와 실험 로그에는 이 차이를 반드시 기록한다.

## 6. 생성할 Conda 환경 목록

이 프로젝트에서 기본으로 생성할 conda 환경은 framework와 CUDA 평가 축을 기준으로 총 8개이다.

### 6.1. CUDA 11.8 환경

CUDA 11.8 평가용 환경은 다음과 같다.

| 환경명 | Framework | Python | CUDA |
|---|---|---:|---:|
| `cupy_py311_cuda118` | CuPy | 3.11 | 11.8 |
| `torch_py311_cuda118` | PyTorch | 3.11 | 11.8 |
| `tf_py311_cuda118` | TensorFlow | 3.11 | 11.8 |
| `jax_py311_cuda118` | JAX | 3.11 | 11.8 |

### 6.2. CUDA 12 계열 환경

CUDA 12 계열 평가용 환경은 다음과 같다.

| 환경명 | Framework | Python | CUDA |
|---|---|---:|---:|
| `cupy_py311_cuda121` | CuPy | 3.11 | 12.1 |
| `torch_py311_cuda121` | PyTorch | 3.11 | 12.1 |
| `tf_py311_cuda122` | TensorFlow | 3.11 | 12.2 |
| `jax_py311_cuda121` | JAX | 3.11 | 12.1 |

### 6.3. 선택 CPU 기준 환경

NumPy 기준 구현을 빠르게 실행하기 위한 CPU 전용 환경은 선택 사항이다.

선택 CPU 기준 환경은 다음과 같이 둘 수 있다.

| 환경명 | Python | 용도 |
|---|---:|---|
| `numpy_py311` | 3.11 | NumPy 기준 구현, 테스트, 문서 예제 실행 |

이 환경은 CUDA 평가와 무관하다.
가벼운 단위 테스트, CPU 기반 MLP 학습, 문서 예제 실행에 사용한다.

## 7. 설치 기준과 예외 사항

이 섹션은 초보자가 혼동하기 쉬운 예외를 정리한다.

### 7.1. TensorFlow CUDA 12.2 예외

CUDA 12 계열에서 TensorFlow만 `tf_py311_cuda122`로 만든다.
다른 framework가 CUDA 12.1을 쓰더라도 TensorFlow는 공식 tested build 기준을 우선한다.

TensorFlow 기준은 다음과 같다.

| 환경명 | TensorFlow | Python | CUDA | cuDNN |
|---|---:|---:|---:|---:|
| `tf_py311_cuda118` | 2.14.0 | 3.11 | 11.8 | 8.7 |
| `tf_py311_cuda122` | 2.15.0 | 3.11 | 12.2 | 8.9 |

이 차이는 실험 결과 비교에서 중요한 메타데이터이다.
따라서 TensorFlow 결과를 기록할 때는 CUDA version뿐 아니라 TensorFlow version과 cuDNN version도 함께 기록한다.

### 7.2. JAX CUDA 12 wheel 표기

JAX CUDA 12 환경명은 `jax_py311_cuda121`로 둔다.
그러나 설치 기준은 `jax[cuda12]`이다.

JAX CUDA 12 환경의 기록 기준은 다음과 같다.

| 항목 | 값 |
|---|---|
| 환경명 | `jax_py311_cuda121` |
| 설치 target | `jax[cuda12]` |
| wheel build CUDA | 12.3 |
| compatible CUDA | 12.1 이상 |
| evaluation target CUDA | 12.1 |

이 표기는 실제 평가 대상과 package 제공 방식을 동시에 설명하기 위한 것이다.

### 7.3. PyTorch CUDA runtime 표기

PyTorch는 conda 설치에서 `pytorch-cuda=12.1`처럼 CUDA runtime을 명시할 수 있다.
pip wheel을 사용할 경우 `cu121`, `cu118` 같은 index URL을 사용한다.

PyTorch 환경 기록 기준은 다음과 같다.

| 환경명 | conda runtime | pip wheel 표기 |
|---|---|---|
| `torch_py311_cuda118` | `pytorch-cuda=11.8` | `cu118` |
| `torch_py311_cuda121` | `pytorch-cuda=12.1` | `cu121` |

conda와 pip 중 하나를 선택해서 설치한다.
하나의 환경에서 conda PyTorch와 pip PyTorch를 섞어 설치하지 않는다.

## 8. 권장 운영 원칙

환경 운영 원칙은 재현성과 문제 해결 속도를 높이는 방향으로 정한다.

권장 운영 원칙은 다음과 같다.

- framework마다 conda 환경을 분리한다.
- CUDA 11.8과 CUDA 12 계열 환경을 분리한다.
- 환경명에는 Python version과 CUDA version을 함께 적는다.
- 설치 후에는 framework version, Python version, CUDA runtime, GPU 인식 여부를 기록한다.
- 실험 결과에는 conda 환경명을 반드시 남긴다.
- TensorFlow와 JAX는 예외 사항을 실험 로그에 함께 기록한다.
- 같은 환경에 여러 GPU framework를 한꺼번에 설치하지 않는다.

초보자는 환경을 적게 만들고 싶어 할 수 있다.
그러나 GPU framework 환경은 적게 만드는 것보다 명확하게 나누는 것이 장기적으로 더 쉽다.
문제가 생겼을 때 어떤 package가 충돌했는지 추적하기 쉽고, 후속 프로젝트에서 같은 이름의 환경을 재사용할 수 있다.

## 9. 기본 생성 및 설치 명령

이 섹션은 현재 프로젝트에서 먼저 사용할 `numpy_py311`, `cupy_py311_cuda118`, `cupy_py311_cuda121` 환경의 생성 및 설치 절차를 정리한다.
명령은 WSL 터미널에서 실행한다.

### 9.1. NumPy CPU 기준 환경

`numpy_py311` 환경은 CPU 기반 NumPy 기준 구현, pytest 실행, 문서 예제 실행에 사용한다.
CUDA package는 설치하지 않는다.

NumPy CPU 기준 환경 생성 명령은 다음과 같다.

```bash
conda create -n numpy_py311 python=3.11
conda activate numpy_py311
```

필수 라이브러리 설치 명령은 다음과 같다.

```bash
pip install numpy pytest matplotlib jupyterlab ipykernel
```

Jupyter kernel 등록 명령은 다음과 같다.

```bash
python -m ipykernel install --user --name numpy_py311 --display-name "Python (numpy_py311)"
```

설치 확인 명령은 다음과 같다.

```bash
python --version
python -c "import numpy as np; print(np.__version__)"
python -c "import pytest; print(pytest.__version__)"
```

### 9.2. CuPy CUDA 11.8 환경

`cupy_py311_cuda118` 환경은 CUDA 11.8 기준 CuPy 실행을 검증하는 환경이다.
이 환경에는 `cupy-cuda11x`를 설치한다.

CuPy CUDA 11.8 환경 생성 명령은 다음과 같다.

```bash
conda create -n cupy_py311_cuda118 python=3.11
conda activate cupy_py311_cuda118
```

공통 필수 라이브러리 설치 명령은 다음과 같다.

```bash
pip install numpy pytest matplotlib jupyterlab ipykernel
```

CuPy CUDA 11.x package 설치 명령은 다음과 같다.

```bash
pip install cupy-cuda11x
```

Jupyter kernel 등록 명령은 다음과 같다.

```bash
python -m ipykernel install --user --name cupy_py311_cuda118 --display-name "Python (cupy_py311_cuda118)"
```

설치 확인 명령은 다음과 같다.

```bash
python --version
python -c "import numpy as np; print(np.__version__)"
python -c "import cupy as cp; print(cp.__version__); print(cp.cuda.runtime.runtimeGetVersion()); print(cp.array([1, 2, 3]))"
```

`cp.cuda.runtime.runtimeGetVersion()` 출력이 `11080`이면 CUDA 11.8 runtime을 정상 인식한 것이다.

### 9.3. CuPy CUDA 12.1 환경

`cupy_py311_cuda121` 환경은 CUDA 12.1 기준 CuPy 실행을 검증하는 환경이다.
이 환경에는 CUDA 12.x 계열 CuPy package를 설치한다.

CuPy CUDA 12.1 환경 생성 명령은 다음과 같다.

```bash
conda create -n cupy_py311_cuda121 python=3.11
conda activate cupy_py311_cuda121
```

공통 필수 라이브러리 설치 명령은 다음과 같다.

```bash
pip install numpy pytest matplotlib jupyterlab ipykernel
```

CuPy CUDA 12.x package 설치 명령은 다음과 같다.

```bash
pip install "cupy-cuda12x[ctk]"
```

Jupyter kernel 등록 명령은 다음과 같다.

```bash
python -m ipykernel install --user --name cupy_py311_cuda121 --display-name "Python (cupy_py311_cuda121)"
```

설치 확인 명령은 다음과 같다.

```bash
python --version
python -c "import numpy as np; print(np.__version__)"
python -c "import cupy as cp; print(cp.__version__); print(cp.cuda.runtime.runtimeGetVersion()); print(cp.array([1, 2, 3]))"
```

`cp.cuda.runtime.runtimeGetVersion()` 출력이 `12010`이면 CUDA 12.1 runtime을 정상 인식한 것이다.
CUDA 12.x package와 driver 조합에 따라 `12080`처럼 다른 CUDA 12 계열 값이 나올 수 있다.
이 경우에도 CUDA 12 계열 runtime으로 기록한다.

### 9.4. 나머지 framework 환경 뼈대

PyTorch, TensorFlow, JAX 환경은 실제 package version을 확정한 뒤 설치한다.
환경 뼈대만 먼저 만들 때의 명령은 다음과 같다.

```bash
conda create -n torch_py311_cuda118 python=3.11
conda create -n tf_py311_cuda118 python=3.11
conda create -n jax_py311_cuda118 python=3.11
conda create -n torch_py311_cuda121 python=3.11
conda create -n tf_py311_cuda122 python=3.11
conda create -n jax_py311_cuda121 python=3.11
```

## 10. 설치 후 확인 항목

환경을 만든 뒤에는 package 설치보다 검증 기록이 더 중요하다.
각 환경에서 최소한 Python version, framework version, GPU 인식 여부를 확인한다.

공통 확인 명령은 다음과 같다.

```bash
python --version
```

CuPy 확인 명령은 다음과 같다.

```bash
python -c "import cupy as cp; print(cp.__version__); print(cp.cuda.runtime.runtimeGetVersion())"
```

PyTorch 확인 명령은 다음과 같다.

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

TensorFlow 확인 명령은 다음과 같다.

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

JAX 확인 명령은 다음과 같다.

```bash
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

검증 결과는 실험 로그 또는 환경 기록 문서에 남긴다.
특히 TensorFlow와 JAX는 CUDA minor version 해석이 framework별로 다르므로 실제 출력값을 보관한다.

현재 WSL에서 확인한 NVIDIA driver 정보는 다음과 같다.

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce GTX 1080 Ti |
| Driver Version | 572.70 |
| Driver supported CUDA Version | 12.8 |

`nvidia-smi`의 CUDA Version은 설치된 CUDA Toolkit version이 아니라 driver가 지원하는 최대 CUDA runtime version이다.
이 값이 CUDA 12.8이므로 CUDA 12.x runtime package 실행 조건은 충족한다.
CUDA 11.8 runtime package는 NVIDIA driver의 하위 호환성을 기준으로 평가한다.

## 11. 참고 출처

이 문서는 다음 공식 문서를 기준으로 작성한다.
각 문서는 시간이 지나면 내용이 바뀔 수 있으므로 실제 설치 직전 다시 확인한다.

| 항목 | 출처 |
|---|---|
| PyTorch previous versions | https://pytorch.org/get-started/previous-versions/ |
| TensorFlow tested build configurations | https://www.tensorflow.org/install/source#tested-build-configurations |
| TensorFlow pip install guide | https://www.tensorflow.org/install/pip |
| JAX installation guide | https://docs.jax.dev/en/latest/installation.html |
| CuPy installation guide | https://docs.cupy.dev/en/stable/install.html |
