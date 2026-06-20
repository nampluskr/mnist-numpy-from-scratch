---
tags: [docs, stage5, logger]
created: "2026-06-20"
updated: "2026-06-20"
---

# Logger 구현

## 1. 개요

`src/core/logger.py`의 `Logger`는 `Trainer.fit`이 반환하는 epoch별 loss/metric 로그를 저장하고 CSV 파일 또는 dict 형태로 내보내는 도구이다. `Logger`는 학습 중 실시간으로 로그를 누적하고, 학습 완료 후 `to_csv`와 `to_dict`로 결과를 내보낸다. `src/utils/training_plots.py`의 `save_training_plots`와 연동하여 학습 곡선 PNG 파일도 함께 생성할 수 있다.

**목표**
- epoch별 loss/metric 로그를 `Logger`에 누적하고 dict 또는 CSV로 내보낸다.
- `Trainer.fit`이 반환한 로그 리스트를 `Logger.load`로 일괄 등록할 수 있다.
- `training_plots`와 연동하여 학습 곡선 PNG를 저장하는 흐름을 지원한다.

## 2. 개념

### 2.1. 로그 구조

`Trainer.fit`은 epoch마다 `{"epoch": int, "loss": float, "metric": float}` 형태의 dict를 리스트에 담아 반환한다. `Logger`는 이 리스트를 받아 내부적으로 컬럼별 리스트로 분리하여 보관한다.

내부 저장 구조는 다음과 같다.

| 키 | 타입 | 내용 |
|---|---|---|
| `epochs` | list of int | 각 epoch 번호 |
| `losses` | list of float | 각 epoch 학습 loss |
| `metrics` | list of float | 각 epoch 학습 metric |

컬럼별 리스트로 분리하면 `losses`만 꺼내어 그래프를 그리거나 numpy 배열로 변환하기 쉽다.

### 2.2. CSV 내보내기

CSV 파일은 로그를 영구적으로 저장하고 재현성을 보장하는 수단이다. `Logger.to_csv`는 header 행에 `epoch,loss,metric`을 기록하고 epoch별로 한 행씩 작성한다. 이 파일을 불러오면 학습을 다시 실행하지 않고도 그래프를 재현할 수 있다.

```text
epoch,loss,metric
1,1.2341,0.5123
2,0.8912,0.7234
...
```

## 3. 구현

공개 인터페이스는 다음과 같다.

| 이름 | 종류 | 입력 | 출력 | 설명 |
|---|---|---|---|---|
| `Logger` | 클래스 | 없음 | logger instance | 학습 로그 관리 도구 |
| `append` | 메서드 | `epoch: int`, `loss: float`, `metric: float` | 없음 | 로그 항목 1건 추가 |
| `load` | 메서드 | `logs: list of dict` | 없음 | `fit` 반환 로그 리스트 일괄 등록 |
| `to_dict` | 메서드 | 없음 | dict | `epochs`, `losses`, `metrics` 리스트 포함 |
| `to_csv` | 메서드 | `save_path: str` | 없음 | CSV 파일 저장 |

### 3.1. Logger 구현

```python
class Logger:
    def __init__(self):
        self.epochs = []
        self.losses = []
        self.metrics = []

    def append(self, epoch, loss, metric):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.metrics.append(metric)

    def load(self, logs):
        for log in logs:
            self.append(log["epoch"], log["loss"], log["metric"])

    def to_dict(self):
        return {
            "epochs": self.epochs,
            "losses": self.losses,
            "metrics": self.metrics,
        }

    def to_csv(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write("epoch,loss,metric\n")
            for epoch, loss, metric in zip(self.epochs, self.losses, self.metrics):
                f.write(f"{epoch},{loss:.6f},{metric:.6f}\n")
```

`load`는 `Trainer.fit`이 반환한 리스트를 받아 `append`를 반복 호출한다. CSV 저장 경로의 상위 폴더가 없을 경우 `os.makedirs`로 자동 생성한다.

### 3.2. training_plots 연동

`Logger.to_dict()`의 반환값을 `save_training_plots`에 전달하면 학습 곡선 PNG를 저장할 수 있다.

```python
from src.utils.training_plots import save_training_plots

logger = Logger()
logger.load(logs)
save_training_plots(logger.to_dict(), save_path="outputs/exp1/training_curves.png")
```

`save_training_plots`는 `losses`와 `metrics` 리스트를 epoch 축으로 그래프를 그려 PNG로 저장한다. `Logger`와 `training_plots`가 서로 동일한 dict 구조를 약속함으로써 결합도를 낮춘다.

## 4. 사용법

최소 사용 예제는 다음과 같다.

```python
from src.core.logger import Logger

logger = Logger()
logger.append(epoch=1, loss=1.234, metric=0.512)
logger.append(epoch=2, loss=0.891, metric=0.723)

print(logger.to_dict())
```

예상 출력은 다음과 같다.

```text
{"epochs": [1, 2], "losses": [1.234, 0.891], "metrics": [0.512, 0.723]}
```

프로젝트 통합 예제는 다음과 같다. `Trainer.fit` 결과를 `Logger`에 로드하고 CSV와 학습 곡선을 저장하는 전체 흐름이다.

```python
from src.core.logger import Logger
from src.utils.training_plots import save_training_plots

logger = Logger()
logs = trainer.fit(train_loader, epochs=10)
logger.load(logs)

logger.to_csv("outputs/multiclass_mlp/train_log.csv")
save_training_plots(logger.to_dict(), save_path="outputs/multiclass_mlp/training_curves.png")
```

## 5. 테스트

테스트 파일은 `tests/stage5/test_logger.py`이다.

```bash
conda run -n numpy_py311 pytest tests/stage5/test_logger.py -v
```

테스트 구성은 다음과 같다.

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestLogger` | 5 | append 후 리스트 길이 확인, load 후 epochs 수 일치, to_dict 키 존재, to_csv 파일 생성 및 헤더 확인, 빈 Logger to_dict 반환 |

## 6. 요약

`Logger`는 epoch별 loss/metric을 컬럼별 리스트로 누적하여 dict 또는 CSV로 내보낸다. `Trainer.fit`이 반환한 로그 리스트를 `load`로 일괄 등록하고, `to_dict`의 반환값을 `save_training_plots`에 직접 전달하여 학습 곡선 PNG를 생성하는 연동 구조를 갖는다.

다음 Stage에서는 [[phase6.1_train-evaluate]]을 다룬다.
