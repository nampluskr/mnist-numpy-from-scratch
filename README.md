# NumPy와 CuPy로 구현하는 MNIST 딥러닝 from scratch

이 저장소는 NumPy와 CuPy만으로 MNIST 기반 딥러닝 학습 과정을 직접 구현하는 프로젝트이다.
`numpy -> pytorch -> tensorflow -> jax` 순서로 이어질 프레임워크 비교 시리즈의 첫 번째 기준 구현이다.

이 프로젝트의 목적은 완성형 학습 프레임워크를 만드는 것이 아니다.
데이터 로딩, target 변환, layer, loss, optimizer, training loop, evaluation, prediction, visualization, GPU 연산이 실제로 어떻게 연결되는지 코드로 확인하는 것이 목적이다.

## 주요 특징

- CPU 기반 NumPy MLP 구현
- GPU 기반 CuPy CNN 구현
- MNIST multiclass classification, binary classification, regression task 지원
- activation, layer, loss, metric, optimizer, training loop 직접 구현
- 학습, 평가, 예측, 시각화 CLI 제공
- `pytest` 기반 stage별 테스트 구조
- 후속 PyTorch, TensorFlow, JAX 프로젝트와 동일한 구조와 사용법을 유지하기 위한 인터페이스 설계

## 프로젝트 범위

이 프로젝트는 MNIST 데이터셋을 세 가지 task로 다룬다.

| Task | 설명 | 출력 |
|---|---|---|
| Multiclass classification | 0-9 digit class 예측 | 10 logits |
| Binary classification | MNIST label을 binary target으로 변환 | 1 logit |
| Regression | MNIST label을 연속값 target으로 변환 | 1 value |

현재 모델 구성은 다음과 같다.

| Model | Backend | 용도 |
|---|---|---|
| MLP | NumPy | CPU 기준 모델 |
| CNN | CuPy | GPU 기반 모델 |

## 저장소 구조

주요 폴더 구조는 다음과 같다.

```text
mnist-numpy-from-scratch/
├── src/          # 재사용 가능한 구현 모듈
├── scripts/      # CLI 실행 진입점
├── tests/        # pytest 테스트 코드
├── docs/         # 공개 학습 문서와 구현 설명
├── data/         # 데이터 경로 또는 설명
├── configs/      # 설정 파일
├── experiments/  # 실험 기록
├── outputs/      # 모델, 그래프, 예측 결과
└── _core/        # 내부 프로젝트 운영 문서
```

`src` 패키지는 후속 프레임워크 프로젝트와 구조를 맞추기 위해 다음처럼 구성한다.

```text
src/
├── data/
│   ├── mnist.py
│   └── dataloader.py
├── nn/
│   ├── activations.py
│   ├── layers.py
│   ├── losses.py
│   ├── metrics.py
│   └── conv.py
├── models/
│   ├── mlp.py
│   └── cnn.py
├── core/
│   ├── optimizers.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── predictor.py
│   ├── visualizer.py
│   └── logger.py
└── utils/
    ├── batching.py
    ├── random.py
    ├── io.py
    ├── checkpoints.py
    └── training_plots.py
```

## 데이터셋

이 프로젝트는 로컬에 저장된 원본 MNIST gzip 파일을 사용한다.
기본 데이터셋 경로는 `DATASET_DIR = "/mnt/d/datasets/mnist"`이다.

필요한 파일은 다음과 같다.

```text
/mnt/d/datasets/mnist/
├── train-images-idx3-ubyte.gz
├── train-labels-idx1-ubyte.gz
├── t10k-images-idx3-ubyte.gz
└── t10k-labels-idx1-ubyte.gz
```

## 실행 환경

기준 Python 버전은 Python 3.11이다. 모든 Python 명령은 `conda run -n {환경명}` 형식으로 실행한다.

확인된 환경 구성은 다음과 같다.

| 환경명 | 용도 | NumPy | CuPy | CUDA |
|---|---|---|---|---|
| `numpy_py311` | MLP (CPU) | 2.4.6 | - | - |
| `cupy_py311_cuda118` | CNN (GPU) | 2.4.6 | 13.6.0 | 11.8 |
| `cupy_py311_cuda121` | CNN (GPU) | 2.4.6 | 14.1.1 | 12.x |

CPU 기반 NumPy 환경은 다음처럼 만든다.

```bash
conda create -n numpy_py311 python=3.11 -y
conda run -n numpy_py311 pip install numpy matplotlib pytest jupyterlab ipykernel
```

CUDA 11.8 기준 CuPy 환경은 다음처럼 만든다.

```bash
conda create -n cupy_py311_cuda118 python=3.11 -y
conda run -n cupy_py311_cuda118 pip install numpy matplotlib pytest jupyterlab ipykernel
conda run -n cupy_py311_cuda118 pip install cupy-cuda11x
```

CUDA 12 계열 CuPy 환경은 다음처럼 만든다.

```bash
conda create -n cupy_py311_cuda121 python=3.11 -y
conda run -n cupy_py311_cuda121 pip install numpy matplotlib pytest jupyterlab ipykernel
conda run -n cupy_py311_cuda121 pip install cupy-cuda12x
```

## 사용법

학습은 저장소 루트에서 실행한다.

```bash
conda run -n numpy_py311 python scripts/train.py --task multiclass --model mlp --epochs 1
conda run -n cupy_py311_cuda118 python scripts/train.py --task multiclass --model cnn --epochs 1
```

평가는 다음처럼 실행한다.

```bash
conda run -n numpy_py311 python scripts/evaluate.py --task multiclass --model mlp
conda run -n cupy_py311_cuda118 python scripts/evaluate.py --task multiclass --model cnn
```

예측은 다음처럼 실행한다.

```bash
conda run -n numpy_py311 python scripts/predict.py --task multiclass --model mlp
conda run -n cupy_py311_cuda118 python scripts/predict.py --task multiclass --model cnn
```

시각화는 다음처럼 실행한다.

```bash
conda run -n numpy_py311 python scripts/visualize.py --task multiclass --model mlp
conda run -n cupy_py311_cuda118 python scripts/visualize.py --task multiclass --model cnn
```

지원하는 task 값은 다음과 같다.

```text
multiclass
binary
regression
```

지원하는 model 값은 다음과 같다.

```text
mlp
cnn
```

## 테스트

전체 테스트는 저장소 루트에서 실행한다.

```bash
conda run -n numpy_py311 pytest tests/ -q
```

테스트는 구현 단계별로 나뉜다.

```text
tests/stage1/  # 공통 유틸리티 (batching, random, io, checkpoints, training_plots)
tests/stage2/  # MNIST loading, Dataset, DataLoader
tests/stage3/  # nn 모듈 (activation, loss, metric, layer, conv)
tests/stage4/  # 모델 (MLP, CNN)
tests/stage5/  # 실행 객체 (optimizer, trainer, evaluator, predictor, visualizer)
tests/stage6/  # 클라이언트 스크립트 (train, evaluate, predict, visualize)
```

## 학습 로드맵

구현은 다음 순서로 진행한다.

| Stage | 주제 |
|---:|---|
| 0 | 환경 구성 및 계획 수립 |
| 1 | 공통 유틸리티 (batching, random, io, checkpoints, training_plots) |
| 2 | MNIST 데이터 로더 (load_mnist, MnistDataset, DataLoader) |
| 3 | nn 모듈 (activation, loss, metric, layer, conv) |
| 4 | 모델 (MLP, CNN) |
| 5 | 실행 객체 (optimizer, trainer, evaluator, predictor, visualizer, logger) |
| 6 | 클라이언트 코드 (scripts, experiments, notebooks) |

## 프레임워크 시리즈 목표

이 저장소는 NumPy/CuPy 기준 구현 역할을 한다.
후속 프로젝트는 같은 문제, 같은 프로젝트 구조, 같은 CLI 사용법을 유지하면서 다른 프레임워크로 구현한다.

```text
numpy -> pytorch -> tensorflow -> jax
```

이 구조를 유지하면 동일한 데이터 파이프라인, 모델 인터페이스, 학습 루프, 평가 흐름, CLI 사용법이 프레임워크별로 어떻게 달라지는지 비교할 수 있다.

