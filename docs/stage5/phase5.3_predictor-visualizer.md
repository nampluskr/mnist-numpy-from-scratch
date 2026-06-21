---
tags: [docs, stage5, predictor, visualizer]
created: "2026-06-20"
updated: "2026-06-21"
---

# Predictor와 Visualizer 구현

## 1. 개요

`src/core/predictor.py`의 `Predictor`는 모델의 raw logit 출력을 task에 맞는 예측값으로 후처리하는 실행 객체이다. `src/core/visualizer.py`의 `Visualizer`는 예측 결과와 입력 이미지를 grid 형태로 시각화하여 PNG 파일로 저장한다. `Predictor`는 `Evaluator`와 달리 loss/metric 계산 없이 사람이 읽을 수 있는 예측값 변환에 집중하며, `Visualizer`는 이 결과를 시각적으로 확인하는 도구이다.

**목표**
- task별 argmax, threshold, round_clip 후처리를 `Predictor.predict`에서 자동으로 선택한다.
- raw logit과 decoded prediction을 함께 담은 dict를 반환하여 클라이언트가 두 값을 모두 활용할 수 있게 한다.
- `Visualizer`로 샘플 이미지와 예측 결과를 grid로 저장하여 육안 검증을 지원한다.

## 2. 개념

### 2.1. Logit과 Prediction의 차이

모델 `forward`의 출력은 activation이 적용되지 않은 raw logit이다. logit은 학습과 loss 계산에는 직접 사용되지만, 사람이 읽을 수 있는 예측값이 아니다. 예측값을 얻으려면 task에 맞는 후처리가 필요하다.

$$
\text{logit} = f_\theta(x) \xrightarrow{\text{후처리}} \text{prediction}
$$

`Predictor`와 `Evaluator`는 같은 모델 출력을 받지만 목적이 다르다.

| 항목 | Evaluator | Predictor |
|---|---|---|
| 입력 | logit, target | logit |
| 처리 | loss/metric 계산 | task별 후처리 |
| 출력 | 숫자 지표 (loss, metric) | 해석 가능한 예측값 |
| 대상 | 모델 성능 측정 | 개별 샘플 예측 |

### 2.2. task별 예측 후처리

task마다 모델 출력의 형태와 의미가 다르기 때문에 후처리 방법이 다르다. `Predictor`는 `task_spec["prediction_mode"]`로 후처리 방식을 결정하며, 세 가지 모드를 지원한다.

**argmax — multiclass**

$$
\text{prediction} = \arg\max_{c}\ \hat{y}_c, \quad \hat{y} \in \mathbb{R}^{N \times 10}
$$

10개 클래스에 대한 raw score 중 가장 큰 인덱스를 예측 클래스로 해석한다. softmax를 적용해도 최대값 위치는 바뀌지 않으므로 argmax는 logit에 직접 적용한다.

**threshold — binary**

$$
\text{prediction} = \mathbf{1}\!\left[\sigma(\hat{y}) \geq 0.5\right], \quad \hat{y} \in \mathbb{R}^{N \times 1}
$$

sigmoid로 `(0, 1)` 확률로 변환한 뒤 0.5를 임계값으로 이진 판정한다. $\sigma(\hat{y}) \geq 0.5$는 $\hat{y} \geq 0$과 동치이므로 logit 부호만으로도 판정할 수 있지만, sigmoid를 거쳐 확률 해석을 명확히 한다.

**round_clip — regression**

$$
\text{prediction} = \text{clip}\!\left(\text{round}(\hat{y} \times 9),\ 0,\ 9\right), \quad \hat{y} \in \mathbb{R}^{N \times 1}
$$

`MNISTDataset`이 레이블을 $\text{label} / 9.0$으로 정규화했으므로, 모델 출력에 9를 곱해 원래 스케일 $[0, 9]$로 복원한다. `round`로 정수화하고 `clip`으로 범위를 보장한다.

task별 후처리 규칙은 다음과 같다.

| task | prediction_mode | logit shape | 후처리 수식 | prediction shape |
|---|---|---|---|---|
| `multiclass` | `argmax` | `(N, 10)` | $\arg\max_c \hat{y}_c$ | `(N,)` int32 |
| `binary` | `threshold` | `(N, 1)` | $\mathbf{1}[\sigma(\hat{y}) \geq 0.5]$ | `(N,)` int32 |
| `regression` | `round_clip` | `(N, 1)` | $\text{clip}(\text{round}(\hat{y} \times 9), 0, 9)$ | `(N,)` int32 |

### 2.3. prediction_mode dispatch 구조

`Trainer`/`Evaluator`의 task dispatch가 `_TASK_FNS` dict를 사용하는 것과 달리, `Predictor`는 `task_spec["prediction_mode"]` 문자열을 직접 참조하여 `if/elif` 분기로 처리한다.

```text
task_spec["prediction_mode"]
    "argmax"     → logits.argmax(axis=1)
    "threshold"  → sigmoid(logits) >= 0.5
    "round_clip" → clip(round(logits * 9), 0, 9)
```

`task_spec`은 `src/task.py`의 `get_task_spec(task)`가 반환하는 dict로, `prediction_mode` 외에도 `output_dim`, `target_dtype` 등을 포함한다. `Predictor`는 이 중 `prediction_mode`만 사용한다.

### 2.4. 반환값 구조

`Predictor.predict`는 `logits`와 `predictions` 두 값을 dict로 함께 반환한다.

```text
{"logits": (N, C) float32, "predictions": (N,) int32}
```

`logits`를 함께 반환하는 이유는 다음과 같다.

- `multiclass`의 경우 각 클래스별 confidence(softmax 후 확률)를 클라이언트가 직접 계산할 수 있다.
- `binary`의 경우 sigmoid 값 자체를 신뢰도로 활용할 수 있다.
- `Visualizer`나 외부 분석 코드가 logit을 추가로 처리할 수 있다.

### 2.5. 예측 결과 grid 시각화

`Visualizer.plot_predictions`는 N개 샘플의 이미지와 예측/정답 레이블을 격자 형태로 배치한 PNG 파일을 저장한다. grid 방식으로 다수 샘플을 한 화면에서 육안으로 확인할 수 있어 모델 오류 패턴을 빠르게 파악할 수 있다.

**grid layout 계산**

$$
\text{cols} = \min(8, N), \quad \text{rows} = \left\lceil \frac{N}{\text{cols}} \right\rceil
$$

$N$개 샘플을 최대 8열로 배치하고, 행 수는 올림 나눗셈으로 결정한다. $N = 16$이면 $2 \times 8$ grid, $N = 5$이면 $1 \times 5$ grid가 된다.

**이미지 변환**

MNIST 이미지는 `MNISTDataset`에서 `(784,)` 1D float32로 저장되어 있다. `images[i].reshape(28, 28)`으로 28×28 2D 배열로 복원한 뒤 `imshow(cmap="gray")`로 렌더링한다.

**정오답 색상 구분**

각 subplot 타이틀에 실제 레이블(T)과 예측값(P)을 표시하고, 정답이면 초록색, 오답이면 빨간색으로 색상을 달리하여 오분류 샘플을 즉시 식별한다.

| 조건 | 타이틀 색상 | 의미 |
|---|---|---|
| `labels[i] == predictions[i]` | 초록 | 정분류 |
| `labels[i] != predictions[i]` | 빨강 | 오분류 |

핵심 용어는 다음과 같다.

| 용어 | 의미 | 이 프로젝트에서의 역할 |
|---|---|---|
| raw logit | activation 없는 모델 출력 | `model.forward(x)` 반환값 |
| prediction | 사람이 읽을 수 있는 예측값 | argmax, threshold, round_clip 결과 |
| prediction_mode | 후처리 방법을 나타내는 문자열 | `task_spec["prediction_mode"]`로 전달 |
| grid | 다수 이미지를 격자 배치한 시각화 | `Visualizer.plot_predictions` 결과 |

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
from src.data.mnist import MNISTDataset

task = "multiclass"
model = MLP(task=task, seed=42)
load_checkpoint(model, "outputs/multiclass_mlp/model.npz")

dataset = MNISTDataset(split="test", task=task)
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
