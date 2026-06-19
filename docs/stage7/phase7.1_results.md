---
tags: [stage7, results, mlp, cnn]
created: 2026-06-18
updated: 2026-06-20
---

# Phase 7.1 experiment 실행 및 result 수집

## 1. 실행 범위

Phase 7.2는 3개 task와 2개 model 조합의 experiment 결과를 수집한다.
모든 조합은 `scripts/train.py`, `scripts/evaluate.py`, `scripts/visualize.py` CLI를 통해 실행한다.

실행 조합은 다음과 같다.

| task | model | output directory |
|---|---|---|
| multiclass | mlp | `outputs/multiclass/mlp/` |
| multiclass | cnn | `outputs/multiclass/cnn/` |
| binary | mlp | `outputs/binary/mlp/` |
| binary | cnn | `outputs/binary/cnn/` |
| regression | mlp | `outputs/regression/mlp/` |
| regression | cnn | `outputs/regression/cnn/` |

## 2. 실행 환경

MLP와 CNN은 서로 다른 기준 environment에서 실행한다.
MLP는 CPU 기반 NumPy 기준 구현이며, CNN은 GPU 기반 CuPy 구현이다.

실행 환경은 다음과 같다.

| model | environment | 용도 |
|---|---|---|
| mlp | `numpy_py311` | CPU 기반 NumPy 기준 실행 |
| cnn | `cupy_py311_cuda121` | CUDA 12 계열 CuPy GPU 실행 |

Codex 실행 환경에서는 WSL GPU device가 노출되지 않아 CNN GPU 실행을 직접 검증하지 못했다.
CNN 결과는 사용자 WSL terminal에서 `cupy_py311_cuda121` environment로 실행하여 수집했다.

## 3. 산출물

각 조합은 checkpoint, training log plot, prediction plot을 산출한다.
산출물 파일은 다음과 같다.

| task | model | checkpoint | training log | prediction plot |
|---|---|---|---|---|
| multiclass | mlp | `outputs/multiclass/mlp/model.npz` | `outputs/multiclass/mlp/training_log.png` | `outputs/multiclass/mlp/predictions.png` |
| multiclass | cnn | `outputs/multiclass/cnn/model.npz` | `outputs/multiclass/cnn/training_log.png` | `outputs/multiclass/cnn/predictions.png` |
| binary | mlp | `outputs/binary/mlp/model.npz` | `outputs/binary/mlp/training_log.png` | `outputs/binary/mlp/predictions.png` |
| binary | cnn | `outputs/binary/cnn/model.npz` | `outputs/binary/cnn/training_log.png` | `outputs/binary/cnn/predictions.png` |
| regression | mlp | `outputs/regression/mlp/model.npz` | `outputs/regression/mlp/training_log.png` | `outputs/regression/mlp/predictions.png` |
| regression | cnn | `outputs/regression/cnn/model.npz` | `outputs/regression/cnn/training_log.png` | `outputs/regression/cnn/predictions.png` |

## 4. 평가 결과

저장된 checkpoint를 불러와 test split 10000개 샘플을 평가했다.
평가 결과는 다음과 같다.

| task | model | loss | metric | samples |
|---|---|---:|---:|---:|
| multiclass | mlp | `0.4499` | `0.8839` | `10000` |
| multiclass | cnn | `0.2106` | `0.9345` | `10000` |
| binary | mlp | `0.2923` | `0.8814` | `10000` |
| binary | cnn | `0.1780` | `0.9347` | `10000` |
| regression | mlp | `0.0408` | `0.5984` | `10000` |
| regression | cnn | `0.0703` | `0.3104` | `10000` |

Multiclass와 binary metric은 accuracy이다.
Regression metric은 R2 score이다.

## 5. 결과 해석

CNN은 classification 계열 task에서 MLP보다 높은 accuracy를 보였다.
Multiclass CNN은 test accuracy `0.9345`, binary CNN은 test accuracy `0.9347`을 기록했다.

Regression에서는 MLP가 CNN보다 높은 R2 score를 보였다.
Regression CNN은 1 epoch 기준으로 학습이 가능하지만, synthetic stage6 테스트와 실제 실행 로그에서 수치 안정성 검토 여지가 남아 있다.

## 6. 남은 작업

Phase 7.2의 result 수집은 완료했다.
CNN tutorial 3종은 Phase 7.3, Phase 7.4, Phase 7.5 문서에서 task별 실행 절차와 결과를 정리한다.
