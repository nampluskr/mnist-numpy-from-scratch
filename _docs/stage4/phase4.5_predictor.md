---
tags: [stage5, core, predictor]
created: 2026-06-17
updated: 2026-06-20
---

# Phase 5.5 Predictor 구현

## 1. 역할

`src/core/predictor.py`는 모델의 raw logit을 task별 규약에 따라 최종 예측값으로 변환한다.
`task_spec["prediction_mode"]`에 따라 argmax, threshold, round_clip 세 가지 후처리를 적용한다.

## 2. 구현

### 2.1. prediction_mode별 후처리 규약

| task | prediction\_mode | 후처리 |
|---|---|---|
| `multiclass` | `argmax` | `logits.argmax(axis=1)` → class index `(N,)` |
| `binary` | `threshold` | `sigmoid(logits) >= 0.5` → 0/1 label `(N,)` |
| `regression` | `round_clip` | `round(clip(logits * 9, 0, 9))` → digit 0~9 `(N,)` |

### 2.2. Predictor(model, task_spec)

| 인자 | 설명 |
|---|---|
| `model` | `forward`를 제공하는 모델 인스턴스 |
| `task_spec` | `get_task_spec(task)`가 반환한 dict |

### 2.3. predict(images)

raw logit과 decoded prediction을 모두 포함한 dict를 반환한다.

```text
logits = model.forward(images)
predictions = decode(logits)   # mode에 따라 분기

return {"logits": logits, "predictions": predictions}
```

`predictions`는 항상 `(N,)` shape의 `int32` 배열이다.

### 2.4. 인터페이스

```python
from src.core.predictor import Predictor
from src.models.mlp import MLP
from src.task import get_task_spec

model = MLP(task="multiclass", seed=0)
predictor = Predictor(model, get_task_spec("multiclass"))

result = predictor.predict(images)  # images: (N, 784)
print(result["predictions"])  # [3, 7, 1, ...]
```

## 3. 테스트

테스트 파일: `tests/stage5/test_predictor.py`

`FixedModel`(preset logit 반환)을 사용하여 MLP 없이 후처리 로직만 독립적으로 검증한다.

| 테스트 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestPredictorMulticlass` | 5 | argmax 정확성, dtype int32, shape (N,), logit passthrough |
| `TestPredictorBinary` | 5 | 양/음수 logit → 1/0, zero logit → 1, shape flat, dtype int32 |
| `TestPredictorRegression` | 5 | 0~9 범위, 하한/상한 clip, shape flat, dtype int32 |

실행 명령:

```bash
conda run -n numpy_env pytest tests/stage5/test_predictor.py -v
```

## 4. 설계 결정

- `predict(images)`는 `logits`와 `predictions`를 모두 반환한다. 호출자가 raw logit을 직접 활용할 수 있도록 설계한다.
- `predictions`는 항상 `(N,)` shape으로 flatten한다. binary/regression의 `(N, 1)` 형태를 ravel()로 정리한다.
- `binary`의 임계값은 logit 도메인에서 `>= 0.0`이 아니라 sigmoid 도메인에서 `>= 0.5`로 계산한다. 두 식은 수학적으로 동치이나, 규약 문서의 표현을 그대로 따른다.
- `round_clip`에서 numpy의 banker's rounding(반올림 규칙: 0.5는 짝수 방향)이 적용된다. 학습 레이블도 같은 규칙으로 변환된 값이므로 일관성이 유지된다.
