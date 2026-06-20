---
tags: [docs, stage5, predictor, visualizer]
created: "2026-06-20"
updated: "2026-06-20"
---

# Predictor와 Visualizer 구현

## 1. 개요

`src/core/predictor.py`의 `Predictor`는 모델의 raw logit 출력을 task에 맞는 예측값으로 후처리하는 실행 객체이다. `src/core/visualizer.py`의 `Visualizer`는 예측 결과와 입력 이미지를 grid 형태로 시각화하여 PNG 파일로 저장한다. `Predictor`는 `Evaluator`와 달리 loss/metric 계산 없이 사람이 읽을 수 있는 예측값 변환에 집중하며, `Visualizer`는 이 결과를 시각적으로 확인하는 도구이다.

**목표**
- task별 argmax, threshold, round_clip 후처리를 `Predictor.predict`에서 자동으로 선택한다.
- raw logit과 decoded prediction을 함께 담은 dict를 반환하여 클라이언트가 두 값을 모두 활용할 수 있게 한다.
- `Visualizer`로 샘플 이미지와 예측 결과를 grid로 저장하여 육안 검증을 지원한다.

## 2. 개념

### 2.1. task별 예측 후처리

모델 `forward`의 출력은 activation이 적용되지 않은 raw logit이다. 사람이 읽을 수 있는 예측값을 얻으려면 task에 맞는 후처리가 필요하다.

task별 후처리 규칙은 다음과 같다.

| task | raw logit shape | 후처리 방법 | decoded 예시 |
|---|---|---|---|
| `multiclass` | `(N, 10)` | `argmax(axis=1)` | 클래스 번호 0~9 |
| `binary` | `(N, 1)` | `sigmoid(logit) >= 0.5` | 0 또는 1 |
| `regression` | `(N, 1)` | `clip(round(logit * 9), 0, 9)` | 정수 0~9 |

`multiclass`에서 `argmax`는 10개 raw score 중 가장 큰 인덱스를 클래스 번호로 해석한다. `binary`에서 sigmoid를 적용한 값이 0.5 이상이면 양성(홀수)으로 판정한다. `regression`에서는 raw 예측값이 `[0, 1]` 범위 실수이므로 9를 곱해 원래 label 스케일로 복원한 뒤 반올림하고 `[0, 9]`로 clip한다.

핵심 용어는 다음과 같다.

| 용어 | 의미 | 이 프로젝트에서의 역할 |
|---|---|---|
| raw logit | activation 없는 모델 출력 | `model.forward(x)` 반환값 |
| decoded prediction | 사람이 읽을 수 있는 예측값 | `argmax`, `threshold`, `round_clip` 결과 |
| posterior | softmax 또는 sigmoid 적용 후 확률값 | `multiclass`에서 `softmax(logit)` 결과 |

### 2.2. 예측 결과 grid 시각화

`Visualizer`는 N개 샘플의 이미지와 예측/정답 레이블을 격자 형태로 배치한 PNG 파일을 생성한다. 한 번에 모든 샘플을 확인할 수 없을 때 grid 방식으로 한 화면에 묶어서 육안 검증을 빠르게 수행할 수 있다.

grid 시각화의 구성 요소는 다음과 같다.

| 요소 | 내용 |
|---|---|
| 이미지 | MNIST 28x28 흑백 이미지 |
| 타이틀 | `pred: {decoded} / true: {label}` 형식 |
| 색상 | 정답이면 검정, 오답이면 빨강으로 타이틀 표시 |
| 파일 | `outputs/{exp_name}/predictions.png` 형태로 저장 |

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `Predictor` | 클래스 | `model`, `task: str` | predictor instance | 예측 후처리 실행 객체 |
| `predict` | 메서드 | `images (N, 784)` float32 | dict | `raw`, `decoded` 포함 |
| `Visualizer` | 클래스 | `task: str` | visualizer instance | 예측 시각화 도구 |
| `save_grid` | 메서드 | `images`, `decoded`, `labels`, `save_path`, `n_cols` | 없음 | 예측 grid PNG 저장 |

### 3.1. Predictor 구현

```python
DECODE_FN = {
    "multiclass": lambda logits: np.argmax(logits, axis=1),
    "binary": lambda logits: (sigmoid(logits) >= 0.5).astype(int).flatten(),
    "regression": lambda logits: np.clip(np.round(logits.flatten() * 9), 0, 9).astype(int),
}

class Predictor:
    def __init__(self, model, task="multiclass"):
        self.model = model
        self.decode_fn = DECODE_FN[task]

    def predict(self, images):
        self.model.eval()
        raw = self.model.forward(images)
        decoded = self.decode_fn(raw)
        return {"raw": raw, "decoded": decoded}
```

`DECODE_FN` dict에 task별 lambda를 등록하여 생성 시 `self.decode_fn`을 바인딩한다. `predict` 메서드는 task에 무관하게 동일한 코드로 동작한다.

### 3.2. Visualizer 구현

```python
class Visualizer:
    def __init__(self, task="multiclass"):
        self.task = task

    def save_grid(self, images, decoded, labels, save_path, n_cols=8):
        n = len(images)
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.8))
        axes = axes.flatten()

        for i in range(n):
            img = images[i].reshape(28, 28)
            pred = decoded[i]
            true = int(labels[i])
            color = "black" if pred == true else "red"
            axes[i].imshow(img, cmap="gray")
            axes[i].set_title(f"pred:{pred}\ntrue:{true}", fontsize=7, color=color)
            axes[i].axis("off")

        for i in range(n, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100)
        plt.close()
```

`images[i].reshape(28, 28)`으로 `(784,)` 배열을 28x28 2D 이미지로 변환한다. 정답과 예측이 다른 경우 타이틀을 빨간색으로 표시하여 오분류 샘플을 즉시 식별할 수 있다.

## 4. 사용법

최소 사용 예제는 다음과 같다.

```python
import numpy as np
from src.models.mlp import MLP
from src.core.predictor import Predictor

model = MLP(task="multiclass", seed=42)
predictor = Predictor(model, task="multiclass")

x = np.random.randn(16, 784).astype(np.float32)
result = predictor.predict(x)

print(result["raw"].shape)
print(result["decoded"].shape)
```

예상 출력은 다음과 같다.

```text
(16, 10)
(16,)
```

프로젝트 통합 예제는 다음과 같다. checkpoint를 로드한 모델로 예측하고 grid를 저장하는 흐름이다.

```python
from src.models.mlp import MLP
from src.core.predictor import Predictor
from src.core.visualizer import Visualizer
from src.utils.checkpoints import load_checkpoint
from src.data.mnist import MnistDataset

task = "multiclass"
model = MLP(task=task, seed=42)
load_checkpoint(model, "outputs/multiclass_mlp/model.npz")

dataset = MnistDataset(split="test", task=task)
images = dataset.images[:32]
labels_raw = dataset.labels_raw[:32]

predictor = Predictor(model, task=task)
result = predictor.predict(images)

visualizer = Visualizer(task=task)
visualizer.save_grid(
    images, result["decoded"], labels_raw,
    save_path="outputs/multiclass_mlp/predictions.png",
    n_cols=8,
)
```

## 5. 테스트

테스트 파일은 `tests/stage5/test_predictor.py`와 `tests/stage5/test_visualizer.py`이다.

```bash
conda run -n numpy_py311 pytest tests/stage5/test_predictor.py tests/stage5/test_visualizer.py -v
```

테스트 구성은 다음과 같다.

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestPredictor` | 4 | 3 task decoded shape, raw/decoded 키 존재, multiclass 값 범위 0~9, eval mode 전환 확인 |
| `TestVisualizer` | 2 | save_grid 파일 생성 확인, n_cols 1일 때 단일 열 동작 |

## 6. 요약

`Predictor`는 `DECODE_FN` dict dispatch로 task별 후처리 함수를 바인딩하고, `predict`는 raw logit과 decoded prediction을 함께 반환한다. `Visualizer.save_grid`는 샘플 이미지와 예측/정답을 grid로 배치하여 오분류를 빨간색으로 표시한 PNG를 저장한다.

다음 Phase에서는 [[phase5.4_logger]]을 다룬다.
