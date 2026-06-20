---
tags: [docs, stage6, notebooks, experiments, results]
created: "2026-06-20"
updated: "2026-06-20"
---

# 실험 노트북과 결과 비교

## 1. 개요

`notebooks/stage6/`의 노트북은 `experiments/run_all.py`로 실행한 학습 결과를 불러와 task별로 MLP와 CNN의 성능을 비교하는 교육용 실습 노트북이다. `stage6-1_cli-and-experiments.ipynb`는 스크립트 직접 호출과 배치 실험 실행 방법을 실습하고, `stage6-2`~`stage6-4`는 각 task의 학습 곡선, 예측 grid, metric을 비교한다.

**목표**
- `scripts/*.py`를 노트북 내에서 직접 호출하는 방법을 실습한다.
- 저장된 CSV 로그와 PNG를 불러와 task별 MLP/CNN 성능을 나란히 비교한다.
- 세 task(multiclass, binary, regression)의 결과를 각 노트북에서 독립적으로 확인한다.

## 2. 개념

### 2.1. 노트북 구성

Stage 6 노트북 4개의 역할은 다음과 같다.

| 노트북 | 핵심 내용 |
|---|---|
| `stage6-1_cli-and-experiments.ipynb` | `scripts/train.py` 직접 호출, `run_all.py` 배치 실행 실습 |
| `stage6-2_multiclass-experiment.ipynb` | multiclass MLP vs CNN 학습 곡선 및 예측 grid 비교 |
| `stage6-3_binary-experiment.ipynb` | binary MLP vs CNN 학습 곡선 및 예측 grid 비교 |
| `stage6-4_regression-experiment.ipynb` | regression MLP vs CNN 학습 곡선 및 예측 grid 비교 |

`stage6-2`~`stage6-4`는 각 노트북이 독립 실행 가능해야 하므로 이전 노트북의 결과에 의존하지 않는다.

### 2.2. 결과 비교 방법

노트북 내 비교는 두 가지 방식으로 이루어진다. 첫째, `outputs/{exp_name}/train_log.csv`를 pandas로 불러와 학습 곡선을 나란히 그린다. 둘째, `outputs/{exp_name}/predictions.png`를 `PIL.Image`로 불러와 노트북 셀에 inline으로 출력한다.

CSV 기반 학습 곡선 비교 예시는 다음과 같다.

```python
import pandas as pd
import matplotlib.pyplot as plt

mlp_log = pd.read_csv("outputs/multiclass_mlp_ep10_lr0.01_bs128/train_log.csv")
cnn_log = pd.read_csv("outputs/multiclass_cnn_ep10_lr0.001_bs128/train_log.csv")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(mlp_log["epoch"], mlp_log["loss"], label="MLP")
axes[0].plot(cnn_log["epoch"], cnn_log["loss"], label="CNN")
axes[0].set_title("Loss")
axes[0].legend()
axes[1].plot(mlp_log["epoch"], mlp_log["metric"], label="MLP")
axes[1].plot(cnn_log["epoch"], cnn_log["metric"], label="CNN")
axes[1].set_title("Metric")
axes[1].legend()
plt.tight_layout()
plt.show()
```

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `notebooks/stage6/stage6-1_cli-and-experiments.ipynb` | 노트북 | - | - | CLI 실습 및 배치 실험 실행 |
| `notebooks/stage6/stage6-2_multiclass-experiment.ipynb` | 노트북 | - | - | multiclass 실험 비교 |
| `notebooks/stage6/stage6-3_binary-experiment.ipynb` | 노트북 | - | - | binary 실험 비교 |
| `notebooks/stage6/stage6-4_regression-experiment.ipynb` | 노트북 | - | - | regression 실험 비교 |

### 3.1. stage6-1: CLI 실습

노트북 내에서 `subprocess`로 `scripts/train.py`를 직접 호출하여 학습을 실행한다.

```python
import subprocess, sys

result = subprocess.run([
    sys.executable, "scripts/train.py",
    "--task", "multiclass", "--model", "mlp",
    "--epochs", "3", "--lr", "0.01", "--batch_size", "128",
], capture_output=True, text=True)

print(result.stdout)
print(result.stderr)
```

`epochs=3`처럼 소규모 설정으로 실행하여 노트북 실행 시간을 단축한다.

### 3.2. stage6-2~4: 실험 결과 비교

task별 노트북은 사전 학습된 결과 파일이 `outputs/`에 있다고 가정하고 CSV와 PNG를 불러와 비교한다.

```python
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("outputs/multiclass_mlp_ep10_lr0.01_bs128/predictions.png")
plt.figure(figsize=(12, 6))
plt.imshow(img)
plt.axis("off")
plt.title("MLP Predictions (multiclass)")
plt.show()
```

`outputs/` 파일이 없을 경우 첫 번째 셀에서 `run_all.py`를 실행하여 생성하도록 안내 텍스트를 작성한다.

## 4. 사용법

최소 사용 예제는 다음과 같다. 노트북 실행 전 학습 결과가 없으면 먼저 배치 실험을 실행한다.

```bash
conda run -n numpy_py311 python experiments/run_all.py
conda run -n numpy_py311 jupyter notebook notebooks/stage6/stage6-2_multiclass-experiment.ipynb
```

프로젝트 통합 예제는 다음과 같다. 전체 비교 노트북을 순서대로 실행하는 흐름이다.

```bash
# 1. 전체 실험 실행
conda run -n numpy_py311 python experiments/run_all.py

# 2. 노트북에서 결과 비교
conda run -n numpy_py311 jupyter nbconvert --to notebook --execute \
  notebooks/stage6/stage6-2_multiclass-experiment.ipynb
```

## 5. 테스트

Stage 6 노트북은 별도 pytest 테스트를 두지 않는다. 노트북 실행 자체가 assertion 기반 검증을 포함하므로 `nbconvert --execute`로 오류 없이 실행되는지 확인하는 것으로 충분하다.

```bash
conda run -n numpy_py311 jupyter nbconvert --to notebook --execute \
  notebooks/stage6/stage6-1_cli-and-experiments.ipynb
```

## 6. 요약

Stage 6 노트북 4개는 `outputs/{exp_name}/`에 저장된 CSV 로그와 PNG를 불러와 task별 MLP/CNN 학습 곡선과 예측 grid를 비교하는 교육용 실습 환경을 제공한다. 각 노트북은 독립 실행 가능하며, 첫 번째 셀에서 필요한 학습 결과가 없을 경우 `run_all.py` 실행을 안내한다.

이 Phase로 Stage 6과 프로젝트 전체 구현이 완료된다.
