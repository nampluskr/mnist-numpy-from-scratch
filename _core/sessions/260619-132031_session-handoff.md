---
tags: [project, sessions]
created: 2026-06-19
updated: 2026-06-19
---

# 교육용 노트북 체계 구축 세션 핸드오프

> 작성일시: 260619-132031
> 세션 목적: notebooks/ 폴더에 교육용 Jupyter 노트북 체계 설계 및 PROJECT-SPEC/TODO 반영
> 이전 핸드오프: 260619-102832_session-handoff.md

## 1. 세션 핵심 요약

`docs/stage0~7` 문서 체계와 독립적으로 교육용 커리큘럼을 형성하는 Jupyter 노트북 16개를
`notebooks/stage1~7/` 폴더 구조로 설계 완료하였다.
각 Stage의 마지막 Phase로 노트북 작성 Phase를 추가하는 방식으로 PROJECT-SPEC.md와
PROJECT-TODO.md를 업데이트해야 한다. 노트북 파일 자체는 아직 작성되지 않았다.

## 2. 사용자 요청 및 의도

| 요청 내용 | 배경 목적 |
|---|---|
| notebooks/ 에 프로젝트 결과를 시각화하는 노트북 생성 | 프로젝트 코드를 실습 기반으로 학습 |
| docs/ 와 마찬가지로 체계적 구성 | 문서와 독립적인 커리큘럼으로 활용 가능해야 함 |
| PROJECT-TODO.md / PROJECT-SPEC.md 에도 반영 | 노트북 작성을 공식 프로젝트 Task로 관리 |
| Stage별 마지막 Phase로 노트북 작성 추가 | 코드 구현 → 테스트 → 문서 → 노트북 4단계 워크플로우 |
| Stage 단위 노트북, 학습 단위별 파일 분리 | phase 1:1 매칭은 일부 노트북이 너무 얇음 |

## 3. 확정된 결정사항

| 항목 | 확정 내용 | 비고 |
|---|---|---|
| 노트북 위치 | 각 Stage 마지막 Phase (Phase 1.4, 2.4, 3.6, 4.8, 5.5, 6.3, 7.7) | 별도 Stage 8 아님 |
| 파일명 규칙 | `stage{N}-{순번}_{kebab-case-keywords}.ipynb` | 예: stage3-1_activations.ipynb |
| 노트북 총 개수 | 16개 | Stage 1~7 각 1~4개 |
| 노트북 형식 | 마크다운(한국어 설명) + 코드(영어) + 그래프(plt.show() inline) | src 저장 함수 사용 안 함 |
| 실행 환경 | numpy_py311 (CNN은 CuPy try/except fallback) | |
| 문서-노트북 관계 | docs와 독립, 노트북만으로 커리큘럼 완성 | N:1 (여러 phase → 1 노트북) |
| 작업 순서 | PROJECT-SPEC.md → PROJECT-TODO.md → 노트북 파일 작성 | |

## 4. notebooks/ 폴더 구조 (확정)

```
notebooks/
├── stage1/
│   └── stage1-1_config-and-task.ipynb           # Phase 1.4 (1개)
├── stage2/
│   ├── stage2-1_mnist-loading.ipynb             # Phase 2.4 (2개)
│   └── stage2-2_dataset-and-dataloader.ipynb
├── stage3/
│   ├── stage3-1_activations.ipynb               # Phase 3.6 (4개)
│   ├── stage3-2_layers.ipynb
│   ├── stage3-3_losses-and-metrics.ipynb
│   └── stage3-4_mlp.ipynb
├── stage4/
│   ├── stage4-1_optimizers.ipynb                # Phase 4.8 (3개)
│   ├── stage4-2_trainer-and-evaluator.ipynb
│   └── stage4-3_experiment.ipynb
├── stage5/
│   └── stage5-1_cli-scripts.ipynb               # Phase 5.5 (1개)
├── stage6/
│   ├── stage6-1_cnn-architecture.ipynb          # Phase 6.3 (2개)
│   └── stage6-2_cnn-training.ipynb
└── stage7/
    ├── stage7-1_multiclass-experiment.ipynb     # Phase 7.7 (3개)
    ├── stage7-2_binary-experiment.ipynb
    └── stage7-3_regression-experiment.ipynb
```

## 5. PROJECT-SPEC.md 변경 내용 (미적용)

### §3 범위 — 마지막 항목 다음에 추가
```
- 교육용 Jupyter 노트북을 `notebooks/` 폴더에 Stage별로 작성하여,
  각 Stage의 코드를 직접 실행·시각화·검증하는 실습 커리큘럼을 제공한다.
```

### §5 진입부 문장 교체
```
(현재) Stage 1부터 각 파일마다 코드 작성 Task와 테스트 작성 Task를 분리하여 진행한다.
(변경) Stage 1부터 코드 구현 → 테스트 → 문서 → 노트북 4단계 워크플로우로 진행한다.
```

### §5.2~5.8 — 각 Stage Phase 목록 마지막에 노트북 Phase 추가

| 위치 | 추가 내용 |
|---|---|
| §5.2 Stage 1 | `- Phase 1.4 Stage 1 노트북 작성` |
| §5.3 Stage 2 | `- Phase 2.4 Stage 2 노트북 작성` |
| §5.4 Stage 3 | `- Phase 3.6 Stage 3 노트북 작성` |
| §5.5 Stage 4 | `- Phase 4.8 Stage 4 노트북 작성` |
| §5.6 Stage 5 | `- Phase 5.5 Stage 5 노트북 작성` |
| §5.7 Stage 6 | `- Phase 6.3 Stage 6 노트북 작성` |
| §5.8 Stage 7 | `- Phase 7.7 Stage 7 노트북 작성` |

### §6.8 notebooks 폴더 구조 — §6.7 다음에 신설
`notebooks/` 폴더 구조와 파일별 학습 단위 표를 포함한다.

## 6. PROJECT-TODO.md 변경 내용 (미적용)

각 Stage 마지막 Phase 섹션(### X.X) 다음에 아래 내용을 삽입한다.

### ### 2.3 다음 (Stage 1)
```markdown
### 2.4. Phase 1.4 Stage 1 노트북 작성

Stage 1에서 구현한 config, task spec, utility를 실습하는 교육용 노트북 작성.

- [ ] `notebooks/stage1/stage1-1_config-and-task.ipynb` 작성
```

### ### 3.3 다음 (Stage 2)
```markdown
### 3.4. Phase 2.4 Stage 2 노트북 작성

MNIST 데이터 로딩, Dataset, DataLoader를 실습하는 교육용 노트북 작성.

- [ ] `notebooks/stage2/stage2-1_mnist-loading.ipynb` 작성
- [ ] `notebooks/stage2/stage2-2_dataset-and-dataloader.ipynb` 작성
```

### ### 4.5 다음 (Stage 3)
```markdown
### 4.6. Phase 3.6 Stage 3 노트북 작성

신경망 기초 모듈과 MLP를 실습하는 교육용 노트북 작성.

- [ ] `notebooks/stage3/stage3-1_activations.ipynb` 작성
- [ ] `notebooks/stage3/stage3-2_layers.ipynb` 작성
- [ ] `notebooks/stage3/stage3-3_losses-and-metrics.ipynb` 작성
- [ ] `notebooks/stage3/stage3-4_mlp.ipynb` 작성
```

### ### 5.7 다음 (Stage 4)
```markdown
### 5.8. Phase 4.8 Stage 4 노트북 작성

학습 프레임워크(Experiment, Trainer, Evaluator, Predictor, Checkpoint)를 실습하는 교육용 노트북 작성.

- [ ] `notebooks/stage4/stage4-1_optimizers.ipynb` 작성
- [ ] `notebooks/stage4/stage4-2_trainer-and-evaluator.ipynb` 작성
- [ ] `notebooks/stage4/stage4-3_experiment.ipynb` 작성
```

### ### 6.4 다음 (Stage 5)
```markdown
### 6.5. Phase 5.5 Stage 5 노트북 작성

CLI 스크립트(train/evaluate/predict/visualize)를 Python에서 직접 호출하는 데모 노트북 작성.

- [ ] `notebooks/stage5/stage5-1_cli-scripts.ipynb` 작성
```

### ### 7.3 다음 (Stage 6)
```markdown
### 7.4. Phase 6.3 Stage 6 노트북 작성

CNN 모델 구조, CuPy 환경, MLP 대비 파라미터 비교를 실습하는 교육용 노트북 작성.

- [ ] `notebooks/stage6/stage6-1_cnn-architecture.ipynb` 작성
- [ ] `notebooks/stage6/stage6-2_cnn-training.ipynb` 작성
```

### ### 8.6 다음 (Stage 7)
```markdown
### 8.7. Phase 7.7 Stage 7 노트북 작성

3종 태스크 전체 실험(MLP + CNN 비교)을 실습하는 교육용 노트북 3개 작성.

- [ ] `notebooks/stage7/stage7-1_multiclass-experiment.ipynb` 작성
- [ ] `notebooks/stage7/stage7-2_binary-experiment.ipynb` 작성
- [ ] `notebooks/stage7/stage7-3_regression-experiment.ipynb` 작성
```

## 7. 노트북 내용 요약

| 노트북 | 핵심 학습 내용 | 주요 그래프 |
|---|---|---|
| stage1-1_config-and-task | config dict, 3 task spec 비교, target 변환 | target 분포 bar |
| stage2-1_mnist-loading | load_mnist, shape/dtype, 픽셀 분포 | 샘플 16장 grid, histogram |
| stage2-2_dataset-and-dataloader | MnistDataset 3종, DataLoader 배치 | target 시각화 |
| stage3-1_activations | sigmoid/relu/softmax 수식 및 그래프 | 4종 함수 그래프 |
| stage3-2_layers | Linear forward/backward, Sequential | shape 추적 표 |
| stage3-3_losses-and-metrics | 3 loss 값+gradient, 3 metric 데모 | loss 곡선 비교 |
| stage3-4_mlp | MLP params shape, 수동 step/epoch | 수동 학습 곡선 |
| stage4-1_optimizers | SGD vs Adam, lr 민감도 비교 | 수렴 비교 그래프 |
| stage4-2_trainer-and-evaluator | Trainer.fit(), Evaluator.evaluate(), task dispatch | 3 task log 비교 |
| stage4-3_experiment | Experiment, Predictor, Checkpoint | 학습 곡선, 예측 grid |
| stage5-1_cli-scripts | scripts/*.py main(args) 직접 호출 | visualize 출력 |
| stage6-1_cnn-architecture | im2col 원리, CNN vs MLP 파라미터 비교 | shape 추적 |
| stage6-2_cnn-training | CNN Experiment, CuPy fallback | 예측 grid, 비교 표 |
| stage7-1_multiclass-experiment | multiclass MLP+CNN 전체 실험 | training curve, 예측 grid |
| stage7-2_binary-experiment | binary MLP+CNN 전체 실험 | training curve, 예측 grid |
| stage7-3_regression-experiment | regression MLP+CNN 전체 실험 | training curve, 예측 grid |

## 8. 미결 사항

| # | 항목 | 현재 상태 | 결정 필요 내용 |
|---|---|---|---|
| 1 | PROJECT-SPEC.md 수정 | 미적용 | 위 §5의 내용을 파일에 반영 필요 |
| 2 | PROJECT-TODO.md 수정 | 미적용 | 위 §6의 내용을 파일에 반영 필요 |
| 3 | notebooks/ 파일 작성 | 미시작 | 16개 노트북 작성 필요 |

## 9. 다음 작업 목록

| 우선순위 | 작업 | 관련 파일 |
|---|---|---|
| 1 | PROJECT-SPEC.md 업데이트 | `_core/PROJECT-SPEC.md` |
| 2 | PROJECT-TODO.md 업데이트 | `_core/PROJECT-TODO.md` |
| 3 | notebooks/ 폴더 생성 및 노트북 작성 (Stage 1 → 7 순서) | `notebooks/stage1~7/` |

## 10. 다음 세션 시작 지시문

아래는 이전 세션에서 설계 완료한 교육용 노트북 체계이다.
이 핸드오프 문서의 내용을 기반으로 아래 순서로 작업을 진행해 주세요.

1. `_core/PROJECT-SPEC.md` — §3, §5.2~5.8, §6.8 업데이트 (§5 변경 내용 참조)
2. `_core/PROJECT-TODO.md` — Phase 1.4, 2.4, 3.6, 4.8, 5.5, 6.3, 7.7 추가 (§6 변경 내용 참조)
3. `notebooks/stage1~7/` 폴더 생성 후 노트북 16개 작성 (§4 폴더 구조, §7 내용 참조)

참고 파일:
- 핸드오프: `_core/sessions/260619-132031_session-handoff.md`
- 프로젝트 명세: `_core/PROJECT-SPEC.md`
- 할일 관리: `_core/PROJECT-TODO.md`
