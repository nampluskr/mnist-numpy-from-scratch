---
tags: [project, sessions]
created: 2026-06-19
updated: 2026-06-19
---

# 교육용 노트북 작성 세션 핸드오프

> 작성일시: 260619-140633
> 세션 목적: notebooks/ 폴더 교육용 노트북 작성 (Phase 1.4/2.4/3.6 일부)
> 이전 핸드오프: 260619-132031_session-handoff.md

## 1. 세션 핵심 요약

이전 세션에서 설계한 교육용 노트북 16개 중 4개를 작성하였다.
PROJECT-SPEC.md와 PROJECT-TODO.md도 핸드오프에서 계획한 내용 그대로 반영하였다.

## 2. 이번 세션 완료 항목

| 항목 | 파일 | 비고 |
|---|---|---|
| PROJECT-SPEC.md 갱신 | `_core/PROJECT-SPEC.md` | §3, §5, §5.2~5.8, §6.8 |
| PROJECT-TODO.md 갱신 | `_core/PROJECT-TODO.md` | Phase 1.4/2.4/3.6/4.8/5.5/6.3/7.7 신설 |
| 노트북 1번 작성 | `notebooks/stage1/stage1-1_config-and-task.ipynb` | Phase 1.4 완료 |
| 노트북 2번 작성 | `notebooks/stage2/stage2-1_mnist-loading.ipynb` | Phase 2.4 부분 |
| 노트북 3번 작성 | `notebooks/stage2/stage2-2_dataset-and-dataloader.ipynb` | Phase 2.4 완료 |
| 노트북 4번 작성 | `notebooks/stage3/stage3-1_activations.ipynb` | Phase 3.6 부분 |

## 3. 미결 사항

| # | 항목 | 현재 상태 |
|---|---|---|
| 1 | notebooks/stage3/ 노트북 3개 미작성 | stage3-2_layers, stage3-3_losses-and-metrics, stage3-4_mlp |
| 2 | notebooks/stage4~7/ 노트북 9개 미작성 | Phase 4.8/5.5/6.3/7.7 |

## 4. 남은 노트북 목록

```
 5. notebooks/stage3/stage3-2_layers.ipynb
 6. notebooks/stage3/stage3-3_losses-and-metrics.ipynb
 7. notebooks/stage3/stage3-4_mlp.ipynb
 8. notebooks/stage4/stage4-1_optimizers.ipynb
 9. notebooks/stage4/stage4-2_trainer-and-evaluator.ipynb
10. notebooks/stage4/stage4-3_experiment.ipynb
11. notebooks/stage5/stage5-1_cli-scripts.ipynb
12. notebooks/stage6/stage6-1_cnn-architecture.ipynb
13. notebooks/stage6/stage6-2_cnn-training.ipynb
14. notebooks/stage7/stage7-1_multiclass-experiment.ipynb
15. notebooks/stage7/stage7-2_binary-experiment.ipynb
16. notebooks/stage7/stage7-3_regression-experiment.ipynb
```

## 5. 다음 작업 목록

| 우선순위 | 작업 | 관련 파일 |
|---|---|---|
| 1 | 5번 노트북 작성 | `notebooks/stage3/stage3-2_layers.ipynb` |
| 2 | 6번 노트북 작성 | `notebooks/stage3/stage3-3_losses-and-metrics.ipynb` |
| 3 | 7번 노트북 작성 | `notebooks/stage3/stage3-4_mlp.ipynb` |
| 4 | Stage 4~7 노트북 작성 | `notebooks/stage4~7/` |

## 6. 다음 세션 시작 지시문

`session-start` 실행 후 5번 노트북 `notebooks/stage3/stage3-2_layers.ipynb` 작성을 이어서 진행해 주세요.

참고 파일:
- 핸드오프: `_core/sessions/260619-140633_session-handoff.md`
- 소스: `src/nn/layers.py`
