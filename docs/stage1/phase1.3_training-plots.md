---
tags: [docs, stage1, training-plots, visualization]
created: "2026-06-20"
updated: "2026-06-20"
---

# 학습 곡선 시각화

## 1. 개요

`src/utils/training_plots.py`는 epoch별 학습 로그를 받아 loss와 metric 곡선을 PNG 파일로 저장하는 시각화 도구이다. `Trainer.fit`이 반환하는 로그 리스트를 직접 받아 train/test 두 곡선을 나란히 출력한다. 학습 완료 후 수렴 여부와 과적합 시점을 육안으로 확인하는 용도로 사용한다.

**목표**
- epoch별 train/test loss와 metric 값을 2개의 서브플롯으로 시각화한다.
- PNG 파일을 `outputs/` 디렉토리에 저장하고 저장 경로를 반환한다.
- `plt.show()` 없이 파일 저장만 수행하여 스크립트와 노트북 모두에서 사용 가능하게 한다.

## 2. 개념

### 2.1. 학습 로그 구조

`Trainer.fit`은 epoch마다 train과 test의 loss와 metric을 집계하여 dict 목록으로 반환한다. `plot_training_log`는 이 구조를 그대로 입력으로 받는다.

로그 리스트의 각 원소 구조는 다음과 같다.

| 키 | 타입 | 내용 |
|---|---|---|
| `"epoch"` | int | epoch 번호 (1부터 시작) |
| `"train"` | dict | `{"loss": float, "metric": float}` |
| `"test"` | dict | `{"loss": float, "metric": float}` |

### 2.2. loss 곡선과 metric 곡선

loss 곡선은 학습이 수렴하는지 확인하는 지표이다. train loss는 계속 감소하는데 test loss가 어느 시점부터 증가한다면 과적합이 시작된 것이다. metric 곡선은 task 목표 달성 수준을 나타낸다. multiclass의 accuracy, binary의 binary_accuracy, regression의 r2_score가 metric으로 사용된다.

두 곡선을 나란히 배치하면 loss와 metric의 변화가 어떻게 연동되는지 한눈에 확인할 수 있다.

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `plot_training_log` | 함수 | `logs: list[dict]`, `output_dir: str`, `filename: str` | `str` (저장 경로) | loss/metric 곡선을 PNG로 저장 |

### 3.1. plot_training_log

```python
def plot_training_log(logs, output_dir="outputs", filename="training_log.png"):
    epochs = [log["epoch"] for log in logs]
    train_losses = [log["train"]["loss"] for log in logs]
    test_losses = [log["test"]["loss"] for log in logs]
    train_metrics = [log["train"]["metric"] for log in logs]
    test_metrics = [log["test"]["metric"] for log in logs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="train")
    ax1.plot(epochs, test_losses, label="test")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(epochs, train_metrics, label="train")
    ax2.plot(epochs, test_metrics, label="test")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric")
    ax2.set_title("Metric")
    ax2.legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close(fig)
    return path
```

`plt.close(fig)`는 파일 저장 후 figure 객체를 명시적으로 해제한다. 이를 생략하면 여러 실험을 순차 실행할 때 figure가 메모리에 누적된다.

`os.makedirs(output_dir, exist_ok=True)`는 `output_dir`가 없을 때 자동으로 생성한다. 스크립트 첫 실행 시 `outputs/` 폴더가 없어도 오류 없이 동작한다.

## 4. 사용법

최소 사용 예제는 다음과 같다.

```python
from src.utils.training_plots import plot_training_log

logs = [
    {"epoch": 1, "train": {"loss": 0.9, "metric": 0.65}, "test": {"loss": 0.85, "metric": 0.70}},
    {"epoch": 2, "train": {"loss": 0.7, "metric": 0.75}, "test": {"loss": 0.72, "metric": 0.78}},
    {"epoch": 3, "train": {"loss": 0.5, "metric": 0.83}, "test": {"loss": 0.55, "metric": 0.82}},
]

path = plot_training_log(logs, output_dir="outputs", filename="example.png")
print(path)
```

예상 출력은 다음과 같다.

```text
outputs/example.png
```

프로젝트 통합 예제는 다음과 같다. 학습 스크립트에서 `Trainer.fit` 반환값을 바로 전달한다.

```python
from src.utils.training_plots import plot_training_log

logs = trainer.fit(train_loader, test_loader, epochs=20)
path = plot_training_log(
    logs,
    output_dir=f"outputs/{exp_name}",
    filename="training_log.png"
)
print(f"학습 곡선 저장: {path}")
```

## 5. 테스트

테스트 파일은 `tests/stage1/test_training_plots.py`이다.

```bash
conda run -n numpy_py311 pytest tests/stage1/test_training_plots.py -v
```

테스트 구성은 다음과 같다.

| 클래스 | 주요 검증 내용 |
|---|---|
| `TestPlotTrainingLog` | 반환 경로에 파일 존재 확인, 반환 타입이 str, output_dir 자동 생성 확인, filename 파라미터 반영 확인 |

## 6. 요약

`plot_training_log`는 epoch별 로그 리스트를 받아 loss와 metric 두 곡선을 PNG로 저장한다. `plt.show()` 없이 파일만 저장하므로 CLI 스크립트와 노트북 모두에서 일관되게 사용할 수 있다. Stage 5의 `Trainer.fit` 반환값과 직접 연결되어 실험 결과 시각화의 진입점이 된다.

다음 Phase에서는 [[phase1.4_notebook]]을 다룬다.
