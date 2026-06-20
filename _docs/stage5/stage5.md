---
tags: [docs, stage5, overview]
created: 2026-06-19
updated: 2026-06-20
---

# Stage 5 클라이언트 코드

## 1. 개요

Stage 5는 `src/core/`의 실행 객체를 호출하는 CLI 진입점 스크립트를 `scripts/` 폴더에 구현하는 단계이다.
각 스크립트는 argparse로 실행 인자를 파싱하고 `Experiment`를 조립한 뒤 학습, 평가, 예측, 시각화를 실행한다.
스크립트 내부에서 `src/core/` 모듈을 직접 조립하지 않고 `Experiment`를 통해 간접 호출하는 구조를 유지한다.

## 2. Phase 구성

### 2.1. Phase 5.1 training CLI 구현

`scripts/train.py`를 구현한다.
`--task`, `--epochs`, `--batch_size`, `--lr`, `--seed`, `--model` 인자를 파싱하고, `Experiment`를 조립하여 `Trainer.fit()`을 호출한다.
학습 완료 후 checkpoint를 저장하고 training log PNG를 `outputs/` 하위 경로에 저장한다.

- [[phase5.1_train|Phase 5.1 training CLI 구현]]

### 2.2. Phase 5.2 evaluation CLI 구현

`scripts/evaluate.py`를 구현한다.
저장된 checkpoint를 로딩하고 `Evaluator.evaluate()`를 호출하여 test split의 loss와 metric을 출력한다.

- [[phase5.2_evaluate|Phase 5.2 evaluation CLI 구현]]

### 2.3. Phase 5.3 prediction CLI 구현

`scripts/predict.py`를 구현한다.
저장된 checkpoint를 로딩하고 `Predictor.predict()`를 호출하여 test 샘플 일부의 예측 결과를 반환한다.

- [[phase5.3_predict|Phase 5.3 prediction CLI 구현]]

### 2.4. Phase 5.4 visualization CLI 구현

`scripts/visualize.py`를 구현한다.
`plot_training_log()` helper로 학습 로그 곡선을 저장하고, `Visualizer`로 prediction 결과 이미지 grid를 저장한다.
두 산출물 모두 `outputs/{task}/{model}/` 경로에 PNG 파일로 저장된다.

- [[phase5.4_visualize|Phase 5.4 visualization CLI 구현]]

## 3. 주요 산출물

| 산출물 | 내용 |
|---|---|
| `scripts/train.py` | 학습 CLI (--task, --model 포함) |
| `scripts/evaluate.py` | 평가 CLI |
| `scripts/predict.py` | 예측 CLI |
| `scripts/visualize.py` | 시각화 CLI (training log + prediction grid) |
| `tests/stage5/` | 4개 CLI 테스트 파일 (87개 테스트 통과) |
