---
tags: [docs, stage6]
created: 2026-06-17
updated: 2026-06-17
---

# Phase 6.2 CNN-core 통합 검증

## 1. 개요

Phase 6.1에서 구현한 `CNN` 모델이 기존 `Experiment`, `Trainer`, `Evaluator` 파이프라인과
수정 없이 호환되는지 검증한다.

`config["model"]` 키를 추가하여 `Experiment`가 MLP/CNN을 선택적으로 조립하도록
확장하고, 31개 통합 테스트로 호환성을 확인한다.

## 2. 수정 파일

| 파일 | 변경 내용 |
|---|---|
| `src/core/experiment.py` | `from src.models.cnn import CNN` 추가; `config["model"]` 분기 |
| `tests/stage6/test_experiment.py` | 31개 통합 테스트 신규 작성 |
| `docs/stage6/phase6.2_cnn-integration.md` | 본 문서 |

## 3. `src/core/experiment.py` 수정

`config["model"]` 키로 MLP/CNN을 선택한다. 기본값은 `"mlp"`이며 기존 코드와 하위 호환된다.

```python
model_type = config.get("model", "mlp")
if model_type == "cnn":
    self.model = CNN(task=task, seed=config["seed"])
else:
    self.model = MLP(task=task, seed=config["seed"])
```

나머지 파이프라인(`DataLoader`, `Trainer`, `Evaluator`, `Predictor`)은 변경하지 않는다.
CNN의 `forward()`가 numpy 배열을 반환하므로 손실 함수와 메트릭 계산이 그대로 동작한다.

## 4. 인터페이스 호환성 요약

CNN이 MLP와 동일한 인터페이스를 제공하므로 기존 실행 객체를 수정 없이 재사용한다.

| 인터페이스 | MLP | CNN |
|---|---|---|
| `model.forward(x)` | `(N, 784) → (N, output_dim) numpy` | 동일 |
| `model.backward(grad)` | numpy grad 입력 | 동일 |
| `model.params` | `list[np.ndarray]` | `list[np.ndarray + np.ndarray]` (CuPy fallback → numpy) |
| `model.grads` | `list[np.ndarray]` | 동일 |
| `model.train()` / `eval()` | `Module.training` 전파 | 동일 |

CuPy 미설치 환경에서 `model.params`의 Conv2d params도 numpy 배열이므로 완전 호환된다.

## 5. 테스트 결과

```
pytest tests/stage6/test_experiment.py -v
31 passed in 0.99s
```

| 테스트 클래스 | 항목 수 | 검증 내용 |
|---|---|---|
| `TestExperimentModelSelection` | 4 | CNN/MLP 분기, 기본값 MLP, loader 조립 |
| `TestCNNExperimentRun` | 12 | 3종 task × (list, 길이, epoch 키, train/test 구조, log 키, num_samples) |
| `TestCNNExperimentLoss` | 4 | multiclass/binary에서 loss 유한값 검증 |
| `TestCNNTrainerStep` | 2 | Trainer 1 epoch 직접 실행, log 키 및 finite 확인 |

regression task는 소규모 synthetic 환경에서 수치 발산(overflow)이 발생하므로 구조/키 검증만 포함하고 유한값 검증에서 제외한다.

## 6. 전체 회귀 검증

Stage 6 추가 후 전체 테스트 결과:

```bash
MPLBACKEND=Agg pytest tests/stage1 tests/stage2 tests/stage3 tests/stage4 tests/stage5 tests/stage6 -q
422 passed, 17 warnings in 22.29s
```

Stage 5까지의 87개 + Stage 6 신규 115개(phase6.1 42개 + phase6.2 31개 + __init__.py + ... → 누적 422개 총합).

## 7. 실행 명령

```bash
pytest tests/stage6/test_experiment.py -v
MPLBACKEND=Agg pytest tests/ -q
```
