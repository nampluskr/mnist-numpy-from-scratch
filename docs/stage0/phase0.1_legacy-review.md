---
tags: [stage0, legacy, review]
created: 2026-06-15
updated: 2026-06-15
---

# Phase 0.1 레거시 재검토

## 1. 레거시 구성

레거시 코드는 `_core/legacy/src/` 에 3개 파일로 구성된다.

| 파일 | 과제 |
|---|---|
| `mnist-multiclass-mlp.py` | 10클래스 분류 (softmax + cross_entropy) |
| `mnist-binary-mlp.py` | 홀짝 이진 분류 (sigmoid + binary_cross_entropy) |
| `mnist-regression-mlp.py` | 레이블 정규화 회귀 (identity + mse) |

각 파일은 `common.mnist`, `common.functions` 외부 모듈에 의존한다. 해당 모듈은 레거시 폴더에 포함되지 않으므로 구현 시 `src/` 에서 직접 대체한다.

## 2. 공통 파이프라인

3개 파일이 동일하게 따르는 파이프라인 구조는 다음과 같다.

| 구간 | 내용 |
|---|---|
| 데이터 로딩 | train/test split 기준 이미지·레이블 로드, float32 정규화 `[0, 1]` |
| 모델 구조 | `784 → 256 → 128 → output_dim`, hidden activation `sigmoid` |
| 학습 루프 | epoch 반복, mini-batch shuffle, forward, loss/metric, manual backward, SGD 업데이트 |
| 평가 루프 | 학습 파라미터 재사용, test split 전체 순회, 평균 loss/metric 집계 |
| 예측 루프 | test 샘플 일부 대상, task별 후처리 후 결과 출력 |

## 3. task별 차이

공통 파이프라인에서 task별로 달라지는 항목은 다음과 같다.

| 항목 | multiclass | binary | regression |
|---|---|---|---|
| target 변환 | one-hot `(N, 10)` | 홀짝 이진화 `(N, 1)` | `label / 9.0` `(N, 1)` |
| output dimension | `10` | `1` | `1` |
| output activation | `softmax` | `sigmoid` | `identity` |
| loss | `cross_entropy` | `binary_cross_entropy` | `mse` |
| metric | `accuracy` | `binary_accuracy` | `r2_score` |
| output gradient | `(preds - y) / N` | `(preds - y) / N` | `2 * (preds - y) / N` |
| prediction 후처리 | `argmax` | `prob >= 0.5` | `round(clip(raw * 9.0, 0, 9))` |

## 4. 구현 파일 매핑

레거시 파이프라인의 각 구간을 `src/` 구현 파일에 다음과 같이 매핑한다.

| 레거시 구간 | 구현 대상 파일 |
|---|---|
| 데이터 로딩 | `src/data/mnist.py` |
| target 변환, task 규약 | `src/task.py` |
| 모델 구조, forward/backward | `src/models/mlp.py`, `src/models/layers.py`, `src/models/activations.py`, `src/models/losses.py` |
| 학습 루프 | `src/core/trainer.py` |
| 평가 루프 | `src/core/evaluator.py` |
| 예측 루프 | `src/core/predictor.py` |
| 하이퍼파라미터 | `src/config.py` |

## 5. 주요 결정사항

레거시 분석 결과로 확정된 설계 원칙은 다음과 같다.

- `common.mnist`, `common.functions` 의존성을 제거하고 `src/` 에서 직접 구현한다.
- task별 차이를 `src/task.py` 단일 파일에 집중하고, 나머지 파이프라인은 공통으로 유지한다.
- 레거시의 flat script 구조를 `src/core/` 실행 객체 + `scripts/` 클라이언트 구조로 전환한다.
- 하이퍼파라미터는 `src/config.py` 의 `get_default_config()` 로 관리한다.
