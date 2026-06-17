---
tags: [stage7, scripts, cli, model]
created: 2026-06-18
updated: 2026-06-18
---

# Phase 7.1 CLI 확장: scripts --model 플래그 추가, stage5 테스트 업데이트

## 1. 역할

`scripts/*.py` 4개에 `--model` 플래그를 추가하여 MLP와 CNN 중 하나를 선택해 실행할 수 있도록 한다.
`Experiment`는 이미 `config["model"]` 분기를 지원하므로 스크립트와 테스트만 수정한다.

## 2. 구현

### 2.1. 변경 대상

다음 4개 스크립트 각각에 동일한 방식으로 `--model` 플래그를 추가한다.

| 스크립트 | 변경 위치 |
|---|---|
| `scripts/train.py` | `parse_args()`, `build_config()` |
| `scripts/evaluate.py` | `parse_args()`, `build_config()` |
| `scripts/predict.py` | `parse_args()`, `build_config()` |
| `scripts/visualize.py` | `parse_args()`, `build_config()` |

### 2.2. parse_args() 변경

모든 스크립트의 `parse_args()`에 다음 인자를 추가한다.

```python
parser.add_argument("--model", default="mlp", choices=["mlp", "cnn"])
```

### 2.3. build_config() 변경

모든 스크립트의 `build_config()`에 다음 필드를 추가한다.

```python
"model": args.model,
```

`Experiment.__init__()`은 `config.get("model", "mlp")`로 분기하므로 추가 수정 없이 연동된다.

### 2.4. 인터페이스

```bash
# MLP (기본값)
python scripts/train.py --task multiclass --model mlp

# CNN
python scripts/train.py --task multiclass --model cnn
python scripts/evaluate.py --task multiclass --model cnn --checkpoint outputs/model
python scripts/predict.py --task multiclass --model cnn --n 16
python scripts/visualize.py --task multiclass --model cnn --output_dir outputs/multiclass/cnn
```

## 3. 테스트

테스트 파일: `tests/stage5/test_train.py`, `test_evaluate.py`, `test_predict.py`, `test_visualize.py`

각 테스트 파일에 추가된 클래스는 다음과 같다.

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestTrainModel` | 4 | config model 키, CNN 학습 실행 및 반환값 |
| `TestEvaluateModel` | 4 | config model 키, CNN 평가 실행 및 반환값 |
| `TestPredictModel` | 4 | config model 키, CNN 예측 실행 및 predictions 수 |
| `TestVisualizeModel` | 4 | config model 키, CNN 시각화 실행 및 파일 생성 |

실행 명령:

```bash
conda run -n numpy_env pytest tests/stage5/ -v
```

## 4. 설계 결정

- `--model` 기본값을 `mlp`로 설정하여 기존 사용 방식과 하위 호환을 유지한다.
- `Experiment`에 `config["model"]` 분기가 이미 구현되어 있어 스크립트 계층 변경만으로 CNN 지원이 완성된다.
- CNN 테스트는 synthetic MNIST를 그대로 사용한다. `CNN.forward()`가 `(N, 784)` 입력을 내부에서 `(N, 1, 28, 28)`로 reshape하므로 별도 픽스처가 필요 없다.
