---
tags: [project, docs, stage0]
created: 2026-06-17
updated: 2026-06-20
---

# Phase 0.0 conda 환경 구성

## 1. 개요

MLP 실행용 `numpy_py311`과 CNN 실행용 `cupy_py311_cuda118`, `cupy_py311_cuda121` conda 환경을 구성하고, CuPy 설치 및 GPU 연산이 정상 동작하는지 검증한다.

## 2. 실행 환경

| 항목 | 값 |
|---|---|
| conda 환경 | `numpy_env` |
| CUDA Toolkit | 11.8 |
| CuPy 패키지 | `cupy-cuda11x` 13.6.0 |
| GPU | NVIDIA GeForce GTX 1080 Ti |

## 3. requirements.txt 추가 항목

```text
cupy-cuda11x
```

> `cupy-cuda118`은 PyPI에 존재하지 않는다. CUDA 11.x 계열 전체를 지원하는 `cupy-cuda11x`를 사용한다.

## 4. 설치

```bash
conda run -n numpy_env pip install cupy-cuda11x
```

## 5. 환경 검증

```bash
conda run -n numpy_env python -c "
import cupy
print('CuPy version:', cupy.__version__)
print('CUDA version:', cupy.cuda.runtime.runtimeGetVersion())
a = cupy.array([1, 2, 3])
print('GPU array:', a)
"
```

출력 예시:

```
CuPy version: 13.6.0
CUDA version: 11080
GPU array: [1 2 3]
```

`CUDA version: 11080` → 11.8.0 정상 인식.
