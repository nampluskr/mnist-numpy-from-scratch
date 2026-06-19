---
tags: [docs, stage4, overview]
created: 2026-06-19
updated: 2026-06-19
---

# Stage 4 실행 객체

## 1. 개요

Stage 4는 학습, 평가, 예측, 시각화를 담당하는 실행 객체를 `src/core/` 패키지에 구현하는 단계이다.
Stage 3에서 완성한 MLP 모델과 Stage 2의 DataLoader를 연결하는 Trainer, Evaluator, Predictor를 구현하고, 이들을 하나의 인터페이스로 조립하는 Experiment를 최상위 진입점으로 둔다.
이 Stage가 완료되면 `Experiment.run()`을 호출하는 것만으로 전체 학습, 평가, 예측, 저장 파이프라인이 실행된다.

## 2. Phase 구성

### 2.1. Phase 4.1 optimizer 구현

`src/core/optimizers.py`에 `SGD`와 `Adam` 옵티마이저를 구현한다.
두 클래스 모두 model의 `params`와 `grads` list를 받아 `step()` 호출 시 in-place로 파라미터를 업데이트하며 반환값이 없다.

- [[phase4.1_optimizers|Phase 4.1 optimizer 구현]]

### 2.2. Phase 4.2 checkpoint 구현

`src/core/checkpoints.py`에 모델 파라미터 저장 및 로딩 함수를 구현한다.
파라미터는 NumPy `.npz` 파일로 저장하며, CuPy 배열도 NumPy로 변환 후 저장하고 로딩 시 대상 모듈의 타입에 맞게 복원한다.

- [[phase4.2_checkpoints|Phase 4.2 checkpoint 구현]]

### 2.3. Phase 4.3 Trainer 구현

`src/core/trainer.py`에 `Trainer` 클래스를 구현한다.
`fit(train_loader)` 메서드는 DataLoader를 수신하여 epoch별 forward, loss/gradient 계산, backward, optimizer step을 반복하고, epoch별 loss와 metric을 집계하여 반환한다.

- [[phase4.3_trainer|Phase 4.3 Trainer 구현]]

### 2.4. Phase 4.4 Evaluator 구현

`src/core/evaluator.py`에 `Evaluator` 클래스를 구현한다.
`evaluate(test_loader)` 메서드는 DataLoader를 수신하여 전체 배치를 순회하고 평균 loss, metric, num_samples를 포함한 dict를 반환한다.
학습 파라미터를 고정하고 gradient를 계산하지 않는다.

- [[phase4.4_evaluator|Phase 4.4 Evaluator 구현]]

### 2.5. Phase 4.5 Predictor 구현

`src/core/predictor.py`에 `Predictor` 클래스를 구현한다.
`predict(images)` 메서드는 raw logit을 task별 후처리(argmax, threshold, round_clip)로 변환하여 raw prediction과 decoded prediction을 함께 담은 dict를 반환한다.

- [[phase4.5_predictor|Phase 4.5 Predictor 구현]]

### 2.6. Phase 4.6 Experiment 구현

`src/core/experiment.py`에 `Experiment` 클래스를 구현한다.
config를 받아 dataset, dataloader, task spec, model, optimizer, trainer, evaluator, predictor를 조립하는 최상위 진입점 역할을 한다.
`config["model"]` 값에 따라 MLP 또는 CNN 모델을 선택하는 분기 로직을 포함한다.

- [[phase4.6_experiment|Phase 4.6 Experiment 구현]]

### 2.7. Phase 4.7 Visualizer 구현

`src/core/visualizer.py`에 `Visualizer` 클래스를, `src/utils/training_plots.py`에 학습 로그 시각화 helper를 구현한다.
`Visualizer`는 prediction 결과 이미지 grid 저장을 전담하고, `plot_training_log(logs, output_dir, filename)` helper는 epoch별 loss/metric 곡선을 PNG로 저장한다.

- [[phase4.7_visualizer|Phase 4.7 Visualizer 구현]]

## 3. 주요 산출물

| 산출물 | 내용 |
|---|---|
| `src/core/optimizers.py` | SGD, Adam 옵티마이저 |
| `src/core/checkpoints.py` | 파라미터 저장/로딩 (NumPy npz, CuPy 호환) |
| `src/core/trainer.py` | 학습 루프 (Trainer.fit) |
| `src/core/evaluator.py` | 평가 루프 (Evaluator.evaluate) |
| `src/core/predictor.py` | 예측 및 task별 후처리 (Predictor.predict) |
| `src/core/experiment.py` | 실행 객체 조립 최상위 진입점 |
| `src/core/visualizer.py` | prediction 결과 이미지 grid 저장 |
| `src/utils/training_plots.py` | 학습 로그 loss/metric 곡선 PNG 저장 |
| `tests/stage4/` | 7개 실행 객체 테스트 파일 (101개 테스트 통과) |
