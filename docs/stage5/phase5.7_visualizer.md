---
tags: [stage5, core, visualizer]
created: 2026-06-17
updated: 2026-06-20
---

# Phase 5.7 Visualizer 구현

## 1. 역할

`src/core/visualizer.py`는 예측 결과를 시각화하고 output_dir에 PNG 파일로 저장한다.
학습 로그 곡선 저장은 `src/utils/training_plots.py`의 helper 함수가 담당한다.

## 2. 구현

### 2.1. Visualizer(output_dir="outputs")

초기화 시 `output_dir`을 생성한다 (중간 디렉터리 포함).

### 2.2. plot_predictions(images, labels, predictions, filename="predictions.png", n=16)

n개 샘플 이미지를 타일 형태로 출력한다. true label과 prediction이 일치하면 초록색, 불일치하면 빨간색으로 표시한다.

```text
입력:
    images:      (N, 784) float32
    labels:      (N,)     int — 실제 레이블 (decoded)
    predictions: (N,)     int — 예측 결과 (Predictor.predict()["predictions"])
출력: output_dir/filename PNG (그리드 레이아웃, 열 수 최대 8)
반환: 저장 경로 str
```

### 2.3. training plot helper

`plot_training_log(logs, output_dir="outputs", filename="training_log.png")`는 `Experiment.run()`이 반환한 per-epoch 로그 목록을 받아 loss/metric 곡선을 그린다.

```text
입력: logs = [{"epoch": 1, "train": {"loss", "metric", ...}, "test": {...}}, ...]
출력: output_dir/filename PNG (1행 2열: Loss | Metric)
반환: 저장 경로 str
```

### 2.4. 인터페이스

```python
from src.core.visualizer import Visualizer
from src.utils.training_plots import plot_training_log

# 학습 곡선
logs = exp.run()
plot_training_log(logs, output_dir="outputs", filename="training_log.png")

# 예측 결과
viz = Visualizer(output_dir="outputs")
result = exp.predictor.predict(test_images)
viz.plot_predictions(test_images, true_labels, result["predictions"], n=16)
```

## 3. 테스트

테스트 파일: `tests/stage5/test_visualizer.py`

synthetic logs와 random image/label 배열을 사용하여 파일 생성 여부와 반환값을 검증한다.

| 테스트 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestPlotPredictions` | 5 | 파일 생성, 파일 비어있지 않음, 반환 경로 str, 커스텀 파일명, n > 샘플 수 처리 |
| `TestVisualizerInit` | 1 | output_dir 자동 생성 |
| `TestPlotTrainingLog` | 5 | helper 파일 생성, 파일 비어있지 않음, 반환 경로 str, 커스텀 파일명, 1 epoch 처리 |

실행 명령:

```bash
conda run -n numpy_py311 pytest tests/stage1/test_training_plots.py tests/stage5/test_visualizer.py -v
```

## 4. 설계 결정

- `plot_predictions`의 `labels`와 `predictions`는 모두 decoded int로 받는다. task별 디코딩 책임은 호출자에게 있다.
- `Visualizer`는 prediction 전용으로 유지하고, 학습 로그 그래프는 `src/utils/training_plots.py`의 helper 함수로 분리한다.
- true/pred 일치 여부를 초록/빨간 색으로 표시하여 오분류 패턴을 시각적으로 파악할 수 있게 한다.
- matplotlib은 `plt.close()`로 각 figure를 즉시 닫아 메모리 누수를 방지한다.
- `output_dir`은 `os.makedirs(..., exist_ok=True)`로 생성하므로 중간 경로가 없어도 동작한다.
