---
tags: [project, docs]
created: 2026-06-15
updated: 2026-06-21 (Phase 3.6 노트북 4개, Phase 4.3 노트북 2개 작성)
---

# PROJECT-LOG.md

이 프로젝트의 주요 작업 이력을 기록한다.
에이전트가 주요 변경 후 갱신한다.

| Date | 작업 내용 | 비고 |
|---|---|---|
| 2026-06-15 | 워크스페이스 초기화 - `_core/legacy/refs/`의 PROJECT.md, PROJECT-TODO.md 내용을 `_core/PROJECT-SPEC.md`, `_core/PROJECT-TODO.md`에 반영 | PROJECT-TODO.md는 전체 미완료 상태로 초기화 |
| 2026-06-15 | CLAUDE.md, project-guide.md 플레이스홀더 채움 (프로젝트명, 목적, 날짜) | |
| 2026-06-17 | Phase 1.1 완료 - requirements.txt, src/config.py, tests/stage1/test_config.py, docs/stage1/phase1.1_config.md | |
| 2026-06-17 | 환경 확정 - numpy_env (Python 3.11), jupyterlab, ipykernel 설치 및 커널 등록 | |
| 2026-06-17 | 구조 확정 - stage 폴더명 0패딩 제거, tests/__init__.py 금지, pyproject.toml 삭제, conftest.py 경로 설정 | coding-rules.md §8 반영 |
| 2026-06-17 | Phase 2.3 완료 - src/data/dataloader.py, tests/stage2/test_dataloader.py (13개), docs/stage2/phase2.3_dataloader.md | Stage 2 전체 54개 테스트 통과 |
| 2026-06-17 | session-end.md Step 6 추가 - 종료 브리핑 후 사용자 승인을 받아 커밋·푸시 진행하는 절차 추가 | |
| 2026-06-17 | 레거시 코드 전체 구조 추가 - task 스크립트 6개(manual·module 각 3종) + common 모듈 6개, numpy_env 실행 검증 완료 | 기존 단일 파일 3개 삭제 |
| 2026-06-17 | Stage 0 문서 전면 재작성 - "재검토" 프레임 제거, 레거시 분석·구현 계획·테스트 계획 수립 중심으로 재편 | phase0.1~0.3 |
| 2026-06-17 | PROJECT-SPEC.md 전면 업데이트 - src/models/ 하위 구성요소, core/optimizers.py 추가, tests stage 단위 구조, 인터페이스 규약 확장 | |
| 2026-06-17 | PROJECT-TODO.md 재구성 - Stage 0 미완료 초기화(다음 세션 재진행), Stage 4 Phase 4.1 optimizers 추가 및 재번호 | |
| 2026-06-17 | Stage 0 Phase 0.1~0.3 전면 재작성 - phase0.1_legacy-analysis.md, phase0.2_implementation-plan.md, phase0.3_test-plan.md 신규 작성, 기존 3개 파일 삭제 | Stage 0 전체 완료 |
| 2026-06-17 | Phase 명칭 전면 개선 - PROJECT-TODO.md, PROJECT-SPEC.md Phase 1.1~7.3 명칭을 "동사구: 항목 나열" 형식으로 개선, 10개 phase 문서 H1 반영 | |
| 2026-06-17 | em dash 전면 제거 및 문서 규칙 추가 - 19개 파일의 em dash를 ` - `로 치환, docs-rules.md에 키보드 입력 불가 문자 사용 금지 조항 추가 | |
| 2026-06-17 | Phase 명 구분자 변경 - phase 헤딩 행의 ` - `를 `: `로 변경 (PROJECT-TODO.md, PROJECT-SPEC.md, docs/stage*/phase*.md) | |
| 2026-06-17 | src 구조 재설계 - models/ 하위 layers/activations/losses를 nn/ 패키지로 분리, torch.nn 대응 명시, PROJECT-SPEC.md §5.4·§6.2·§6.5·§6.6 갱신 | PyTorch 방식 통일 결정 |
| 2026-06-17 | Stage 3 전면 재구성 - Phase 2개를 4개로 분리(activations/layers/losses/mlp), src/nn/ 4파일 신규, mlp.py Sequential 기반 재작성, tests/stage3 테스트 69개 통과 | logit 출력, *_grad 함수 도입 |
| 2026-06-17 | Stage 3 문서 4개 작성 - phase3.1_activations.md, phase3.2_layers.md, phase3.3_losses.md, phase3.4_mlp.md, 구 버전 phase3.1_mlp.md 삭제 | Stage 3 전체 완료 |
| 2026-06-17 | Phase 4.1 완료 - src/core/optimizers.py (SGD, Adam), tests/stage4/test_optimizers.py (12개), docs/stage4/phase4.1_optimizers.md | |
| 2026-06-17 | Phase 4.2 완료 - src/core/checkpoints.py (save/load), tests/stage4/test_checkpoints.py (7개), docs/stage4/phase4.2_checkpoints.md | |
| 2026-06-17 | Phase 4.3 완료 - src/core/trainer.py (Trainer.fit), tests/stage4/test_trainer.py (16개), docs/stage4/phase4.3_trainer.md | |
| 2026-06-17 | Phase 4.4 완료 - src/core/evaluator.py (Evaluator.evaluate), tests/stage4/test_evaluator.py (16개), docs/stage4/phase4.4_evaluator.md | |
| 2026-06-17 | Phase 4.5 완료 - src/core/predictor.py (Predictor.predict), tests/stage4/test_predictor.py (15개), docs/stage4/phase4.5_predictor.md | argmax/threshold/round_clip 3가지 후처리 |
| 2026-06-17 | Phase 4.6 완료 - src/core/experiment.py (Experiment.run), tests/stage4/test_experiment.py (24개), docs/stage4/phase4.6_experiment.md | synthetic MNIST gz 기반 통합 테스트 |
| 2026-06-17 | Phase 4.7 완료 - src/core/visualizer.py (Visualizer), tests/stage4/test_visualizer.py (11개), docs/stage4/phase4.7_visualizer.md | Stage 4 전체 101개 테스트 통과 |
| 2026-06-17 | Phase 5.1 완료 - scripts/train.py, tests/stage5/test_train.py (23개), docs/stage5/phase5.1_train.md | |
| 2026-06-17 | Phase 5.2 완료 - scripts/evaluate.py, tests/stage5/test_evaluate.py (19개), docs/stage5/phase5.2_evaluate.md | |
| 2026-06-17 | Phase 5.3 완료 - scripts/predict.py, tests/stage5/test_predict.py (19개), docs/stage5/phase5.3_predict.md | |
| 2026-06-17 | Phase 5.4 완료 - scripts/visualize.py, tests/stage5/test_visualize.py (26개), docs/stage5/phase5.4_visualize.md | Stage 5 전체 87개 테스트 통과 |
| 2026-06-17 | Phase 6.1 완료 - src/nn/layers.py (training/train/eval 추가), src/nn/conv.py (im2col/col2im + Conv2d/MaxPool2d/Flatten/Dropout), src/models/cnn.py, tests/stage6/test_cnn.py (42개), docs/stage6/phase6.1_cnn.md | CuPy/numpy 양용, fallback 지원 |
| 2026-06-17 | Phase 6.2 완료 - src/core/experiment.py (model 분기 추가), tests/stage6/test_experiment.py (31개), docs/stage6/phase6.2_cnn-integration.md | Stage 6 전체 73개 테스트 통과, 전체 422개 통과 |
| 2026-06-17 | Phase 6.0 완료 - requirements.txt cupy-cuda11x 추가, numpy_env 설치 및 검증(CuPy 13.6.0 / CUDA 11.8), docs/stage6/phase6.0_cupy-setup.md | cupy-cuda118 미존재, cupy-cuda11x 사용 |
| 2026-06-17 | CLAUDE.md §5 핵심 행동 규칙에 Python 실행 환경(numpy_env) 명시 추가 | |
| 2026-06-18 | Phase 7.1 완료 - scripts/*.py --model 플래그 추가(mlp/cnn), tests/stage5 TestXxxModel 클래스 추가(GPU 없을 때 skip), docs/stage7/phase7.1_cli-extension.md | 95 passed, 8 skipped (CuPy GPU skip) |
| 2026-06-18 | CuPy 환경 교체 - cupy-cuda11x에서 cupy-cuda12x[ctk]로 변경 (CUDA 12.8 드라이버 호환), requirements.txt 갱신 | CUDA 12.8 드라이버에 nvrtc 11.x 미존재 |
| 2026-06-18 | dtype float32 버그 수정 - src/nn/layers.py Linear, src/nn/conv.py Conv2d 초기화 시 scale(float64) 곱 후 astype(float32) 순서 변경 | float32 * float64 업캐스트 방지 |
| 2026-06-18 | CuPy 14.x 호환 수정 - src/models/cnn.py np.asarray() 호출을 .get()으로 변경, tests/stage6/test_cnn.py to_np() 헬퍼 추가 | 176 passed (stage5+stage6 전체) |
| 2026-06-18 | Phase 7.2 부분 완료 - multiclass, binary, regression MLP output 3종 생성 및 checkpoint 평가 | CNN output과 results 문서는 미진행 |
| 2026-06-18 | Phase 7.3~7.5 MLP 튜토리얼 작성 - multiclass, binary, regression MLP 평가 결과와 실행 절차 문서화 | CNN 튜토리얼은 미진행 |
| 2026-06-18 | Phase 7.6 완료 - 후속 PyTorch, TensorFlow, JAX 프로젝트 연계를 위한 interface 규약과 PyTorch 마이그레이션 체크리스트 작성 | CNN 평가는 중지 상태로 기록 |
| 2026-06-18 | 프로젝트 운영 문서 규칙 정합화 - PROJECT-TODO.md Phase 헤더 번호 체계, 도입 문장, Obsidian 내부 링크 정리 | PROJECT-SPEC.md 및 docs Phase H1 제목 동기화 |
| 2026-06-18 | docs/stage7 구조 정리 - task별 하위 폴더 제거, phase7.3~7.5 튜토리얼 문서를 stage7 바로 아래로 이동 | PROJECT-TODO.md와 세션 핸드오프 경로 참조 갱신 |
| 2026-06-19 | docs/ 포털 문서 9개 신규 작성 - docs/index.md(최상위 진입점) + docs/stageN/stageN.md(Stage 0~7 Chapter 소개 문서) | Obsidian 그래프 고립 노드 해소, Phase 링크 불릿 형식 적용 |
| 2026-06-19 | 교육용 노트북 체계 구축 - PROJECT-SPEC.md §3·§5·§6.8 갱신, PROJECT-TODO.md Phase 1.4/2.4/3.6/4.8/5.5/6.3/7.7 추가, 노트북 4개 작성 | 총 16개 중 4개 완료 |
| 2026-06-19 | Phase 3.6 완료 - stage3-2_layers.ipynb, stage3-3_losses-and-metrics.ipynb, stage3-4_mlp.ipynb 작성 | |
| 2026-06-19 | Phase 4.8 완료 - stage4-1_optimizers.ipynb, stage4-2_trainer-and-evaluator.ipynb, stage4-3_experiment.ipynb 작성 | |
| 2026-06-19 | Phase 5.5 완료 - stage5-1_cli-scripts.ipynb 작성 | 16개 중 11개 완료 |
| 2026-06-19 | Phase 6.3 완료 - stage6-1_cnn-architecture.ipynb (im2col 원리, shape 추적, MLP 비교), stage6-2_cnn-training.ipynb (3종 task CNN 학습, CuPy fallback, MLP 비교) | 16개 중 13개 완료 |
| 2026-06-19 | Phase 7.7 완료 - stage7-1_multiclass-experiment.ipynb, stage7-2_binary-experiment.ipynb, stage7-3_regression-experiment.ipynb 작성 | 16개 전체 완료 |
| 2026-06-20 | Stage 순서 재배치 - Stage 6(CuPy CNN)을 Stage 4로 이동, 기존 Stage 4/5가 Stage 5/6으로 순환 재배치 | docs/, tests/, notebooks/ 폴더명·파일명·내부 텍스트 전체 갱신 |
| 2026-06-20 | 전면 리팩토링 계획 수립 - Stage 0~7(8단계) 재편, config.py/task.py/experiment.py 제거, CONFIGS 방식 전환, docs/ 책 수준 재작성, notebooks/ 독립 튜토리얼 재작성 계획 확정 | 브랜치 refactor/book-notebook-restructure, 코드 변경 없음 |
| 2026-06-20 | PROJECT-TODO.md Phase 도입 문장 형식 통일 - 모든 Phase 도입 문장을 `~하고, ~를 검증한다.` / `~를 작성한다.` 형식으로 통일 | 11개 문장 수정 |
| 2026-06-20 | PROJECT-TODO.md task 수준 기술 형식 통일 - 파일 항목에 동사 추가(src/→구현, tests/→작성, scripts/→구현), 괄호 설명 앞 동사 삽입, outputs/ 항목에 `생성` 추가 | 약 64개 항목 수정 |
| 2026-06-20 | 폴더 역할 체계 재정의 - CLAUDE.md·PROJECT-SPEC.md §3·§6.3 업데이트 (src/scripts/docs/notebooks/experiments/ 역할 명확화) | experiments/ 역할: CLI scripts 실행 예제 Python 스크립트 |
| 2026-06-20 | experiments/ 신규 구축 - subprocess 기반 batch job 4개(train/evaluate/predict/visualize_all.py) + run_all.py 작성, CONFIGS 주입 방식 채택 | exp_name 도입, outputs/{exp_name}/ 저장 구조 |
| 2026-06-20 | tests/ __init__.py 3개 삭제 - tests/stage2,3,4/__init__.py 제거 (coding-rules.md 규칙 준수) | |
| 2026-06-20 | experiments/ 파일명 변경 - train/evaluate/predict/visualize_all.py → run_train/evaluate/predict/visualize.py, run_all.py import 경로 갱신 | |
| 2026-06-20 | PROJECT-SPEC.md §3 서브섹션 도입 - 3.1 과제 및 모델 / 3.2 코드 구현 / 3.3 문서 및 실험 3개 서브섹션으로 재구성, 도입 문장 추가 | |
| 2026-06-20 | Stage 0 환경 구성 편입 - Phase 0.0 conda 환경 구성 신설, docs/stage4/phase4.0_cupy-setup.md → docs/stage0/phase0.0_conda-setup.md 이동, PROJECT-SPEC.md §5.1 제목 변경, Stage 4에서 Phase 4.0 제거 | | |

## 260621 Stage 2 data 파이프라인 전면 재설계 및 Phase 2.2~2.5 완료

**완료 항목**
- `src/data/mnist.py` 재작성 — `load_images`/`load_labels` 순수 로딩 함수만 유지 (후속 프레임워크 공통 재사용)
- `src/data/transforms.py` 신규 작성 — `normalize`, `to_flat`, `one_hot`, `binarize`, `to_regression`
- `src/data/datasets.py` 신규 작성 — `MNISTDataset` base + `MulticlassDataset`, `BinaryDataset`, `RegressionDataset`
- `src/task.py` 신규 작성 — `get_task_spec()` 이동 (data 패키지 순환 의존 방지)
- `src/data/__init__.py` 갱신 — transforms, datasets export 추가
- `src/models/mlp.py`, `cnn.py` import 수정 — `from src.task import get_task_spec`
- `scripts/train.py`, `evaluate.py`, `predict.py`, `visualize.py` — Dataset 생성 코드 변경
- `tests/stage2/test_mnist.py`, `test_dataset.py` 재작성
- `tests/stage2/test_transforms.py` 신규 작성 (29개 테스트 통과)
- `tests/stage4/test_trainer.py`, `test_evaluator.py`, `test_predictor.py` import 수정
- Phase 2.2 `docs/stage2/phase2.2_transforms.md` 신규 작성
- Phase 2.3 `docs/stage2/phase2.3_dataset.md` 재작성 (구 phase2.2_dataset.md 파일명 변경)
- Phase 2.4 `docs/stage2/phase2.4_dataloader.md` 파일명 변경 (구 phase2.3_dataloader.md)
- Phase 2.5 노트북 3개 신규 작성: stage2-3_multiclass-dataset, stage2-4_binary-dataset, stage2-5_regression-dataset
- PROJECT-SPEC.md, PROJECT-TODO.md Stage 2 Phase 목록 및 노트북 구조 갱신

**산출물**

| 파일/산출물 | 내용 |
|---|---|
| `src/data/mnist.py` | `load_images`/`load_labels` 순수 로딩만 유지, 후속 프레임워크 공통 재사용 |
| `src/data/transforms.py` | 이미지 변환 2개 + 레이블 변환 3개 순수 함수 |
| `src/data/datasets.py` | `MNISTDataset` base + task별 3개 Dataset (eager transform 주입) |
| `src/task.py` | `get_task_spec()` 독립 파일로 이동 |
| `tests/stage2/test_transforms.py` | 5개 클래스, 29개 테스트 통과 |
| `docs/stage2/phase2.2_transforms.md` | 이미지 정규화/reshape 개념(수식), task별 레이블 변환(수식), 구현, 사용법, 테스트표 |
| `docs/stage2/phase2.3_dataset.md` | transforms.py + datasets.py 통합 문서 (파일명 변경) |
| `docs/stage2/phase2.4_dataloader.md` | Dataloader 문서 (파일명 변경) |
| `notebooks/stage2/stage2-3_multiclass-dataset.ipynb` | normalize/to_flat/one_hot, MulticlassDataset, one-hot heatmap, Dataloader 배치 |
| `notebooks/stage2/stage2-4_binary-dataset.ipynb` | binarize, BinaryDataset, 홀짝 막대·파이 차트, 홀수/짝수 샘플 시각화 |
| `notebooks/stage2/stage2-5_regression-dataset.ipynb` | to_regression, RegressionDataset, 변환 함수 그래프, digit별 분포, 복원 검증 |

**결정사항**

| 항목 | 결정 내용 |
|---|---|
| `load_images`/`load_labels` 역할 | 순수 로딩만 담당. 정규화·변환은 transforms.py + datasets.py 계층이 담당 |
| `get_task_spec()` 위치 | `src/task.py` 독립 파일 — data 패키지와 core 패키지 간 순환 의존 방지 |
| transform 적용 방식 | eager (생성자에서 전체 배열에 한 번) — `__getitem__` 호출 시 추가 연산 없음 |
| task별 Dataset 설계 | 기본 transform 내장 + 외부 override 허용 (`transform=None` 체크) |

## 260621 Phase 3.6, Phase 4.3 교육용 노트북 작성

**완료 항목**
- Phase 3.6 노트북 4개 작성: stage3-1_activations, stage3-2_losses-and-metrics, stage3-3_layers, stage3-4_conv-architecture
- Phase 4.3 노트북 2개 작성: stage4-1_mlp, stage4-2_cnn-model
- PROJECT-TODO.md Phase 3.6, Phase 4.3 완료 처리

**산출물**

| 파일/산출물 | 내용 |
|---|---|
| `notebooks/stage3/stage3-1_activations.ipynb` | sigmoid/softmax/relu/identity 수식, 수치안정성 검증, 4종 시각화, task별 활성화 선택 |
| `notebooks/stage3/stage3-2_losses-and-metrics.ipynb` | CE/BCE/MSE loss + gradient 검증, 3 task 손실 곡선 시각화, accuracy/binary_accuracy/r2_score 검증 |
| `notebooks/stage3/stage3-3_layers.ipynb` | Linear(He init, forward/backward, in-place), Sigmoid/ReLU, Sequential(params 수집), 수동 SGD 20 steps |
| `notebooks/stage3/stage3-4_conv-architecture.ipynb` | im2col shape 원리, col2im 역변환, Conv2d/MaxPool2d/Flatten/Dropout, CNN shape 추적표, CuPy try/except fallback |
| `notebooks/stage4/stage4-1_mlp.ipynb` | MLP 파라미터 구성(6개, 235,146개), seed 재현성, task별 forward/backward 검증, 3 task 수동 SGD 학습 곡선 |
| `notebooks/stage4/stage4-2_cnn-model.ipynb` | CNN 파라미터(8개, 824,458개), CuPy/numpy 경계 확인, Dropout train/eval 차이, MLP vs CNN 인터페이스 비교 |

## 260621 Stage 4 MLP/CNN 문서 개념 섹션 상세화

**완료 항목**
- `docs/stage4/phase4.1_mlp.md` 개념 섹션 상세화 (2개 소절 -> 9개 소절)
- `docs/stage4/phase4.2_cnn.md` 개념 섹션 상세화 (3개 소절 -> 13개 소절)

**산출물**

| 파일/산출물 | 내용 |
|---|---|
| `docs/stage4/phase4.1_mlp.md` | MLP란 무엇인가, 구조, 파라미터 수, Forward/Backward 수식 전개, params/grads 설계, seed 파생, logit 입력 설계 의미, 학습 한 스텝 전체 흐름 |
| `docs/stage4/phase4.2_cnn.md` | CNN이란 무엇인가, 구조, 합성곱/feature map, 공간 크기 변화 추적, 파라미터 수, Forward/Backward 단계별 설명, CuPy/numpy 경계 설계, Dropout 역할, fallback 설계, params/grads 설계, MLP/CNN 인터페이스 비교, 학습 한 스텝 전체 흐름 |

## 260621 Stage 5 src/core/ 구현 완료 및 Stage 6 tests/ 정비

**완료 항목**
- Phase 5.1: `tests/stage5/test_optimizers.py` 작성 (7 passed)
- Phase 5.4: `src/core/logger.py` 구현 + `tests/stage5/test_logger.py` 작성 (5 passed)
- Phase 6.1: `tests/stage6/test_train.py`, `tests/stage6/test_evaluate.py` 정위치 배치
- Phase 6.2: `tests/stage6/test_predict.py`, `tests/stage6/test_visualize.py` 정위치 배치
- `tests/stage5/` 정비: scripts/ 테스트 4개를 `tests/stage6/`으로 이동
- PROJECT-TODO.md Phase 5.1~5.4, 6.1~6.2 완료 처리
- PROJECT-TODO.md Phase 명칭 PROJECT-SPEC.md 기준으로 동기화 (7개 항목)

**산출물**

| 파일/산출물 | 내용 |
|---|---|
| `src/core/logger.py` | `Logger`: append / load / to_dict / to_csv |
| `tests/stage5/test_optimizers.py` | TestSGD(3) + TestAdam(4), 7 passed |
| `tests/stage5/test_logger.py` | TestLogger(5), 5 passed |
| `tests/stage6/test_train.py` | `scripts/train.py` 검증 - build_config, main 반환값, checkpoint 저장 |
| `tests/stage6/test_evaluate.py` | `scripts/evaluate.py` 검증 - build_config, main 반환값, checkpoint 로드 |
| `tests/stage6/test_predict.py` | `scripts/predict.py` 검증 - 예측 수·dtype, n clips |
| `tests/stage6/test_visualize.py` | `scripts/visualize.py` 검증 - PNG 파일 생성, logs 길이 |

**검증**

| 명령 | 결과 |
|---|---|
| `pytest tests/stage5/ -q` | 12 passed |
| `pytest tests/stage6/ -q` | 95 passed, 8 skipped |

## 260620 Stage 2~4 docs/ 전체 작성 완료

**완료 항목**
- Phase 2.1 문서 작성: `docs/stage2/phase2.1_mnist.md`
- Phase 2.2 문서 작성: `docs/stage2/phase2.2_dataset.md`
- Phase 2.3 문서 작성: `docs/stage2/phase2.3_dataloader.md`
- Phase 3.1 문서 작성: `docs/stage3/phase3.1_activations.md`
- Phase 3.2 문서 작성: `docs/stage3/phase3.2_losses.md`
- Phase 3.3 문서 작성: `docs/stage3/phase3.3_metrics.md`
- Phase 3.4 문서 작성: `docs/stage3/phase3.4_mlp-layers.md`
- Phase 3.5 문서 작성: `docs/stage3/phase3.5_cnn-layers.md`
- Phase 4.1 문서 작성: `docs/stage4/phase4.1_mlp.md`
- Phase 4.2 문서 작성: `docs/stage4/phase4.2_cnn.md`
- PROJECT-TODO.md Stage 2~4 문서 항목 완료 처리

**산출물**

| 파일/산출물 | 내용 |
|---|---|
| `docs/stage2/phase2.1_mnist.md` | IDX 바이너리 포맷, load_mnist, split 매핑, 합성 gz 테스트 |
| `docs/stage2/phase2.2_dataset.md` | Dataset 프로토콜, task별 target 변환, __getitem__ 인터페이스 |
| `docs/stage2/phase2.3_dataloader.md` | 인덱스 기반 배치 조립, shuffle, 마지막 배치 처리 |
| `docs/stage3/phase3.1_activations.md` | sigmoid 수치 안정성, softmax max subtraction, relu/identity |
| `docs/stage3/phase3.2_losses.md` | logit 입력 설계, activation 내부 처리, gradient 수식 단순화 |
| `docs/stage3/phase3.3_metrics.md` | accuracy/binary_accuracy logit 기반, R2 score 수식 |
| `docs/stage3/phase3.4_mlp-layers.md` | Module 인터페이스, Linear backward in-place, Sequential params 집계 |
| `docs/stage3/phase3.5_cnn-layers.md` | im2col/col2im 원리, Conv2d/MaxPool2d, Dropout inverted scaling |
| `docs/stage4/phase4.1_mlp.md` | 3층 MLP 구조, seed 파생, logit 출력 설계 |
| `docs/stage4/phase4.2_cnn.md` | Conv-FC 구조, CuPy/numpy 경계 처리, fallback 패턴 |

## 260620 Stage 1 docs/ 및 notebooks/ 전체 작성 완료

**완료 항목**
- `notebooks/` 기존 16개 파일 `_notebooks/`로 이동 (아카이브)
- Phase 1.1 문서 작성: `docs/stage1/phase1.1_batching-and-random.md`
- Phase 1.2 문서 작성: `docs/stage1/phase1.2_io-and-checkpoints.md`
- Phase 1.3 문서 작성: `docs/stage1/phase1.3_training-plots.md`
- `tests/stage1/test_checkpoints.py` 신규 작성 (4 tests, 24 passed)
- Phase 1.4 노트북 작성: `notebooks/stage1/stage1-1_utils.ipynb`
- PROJECT-TODO.md Stage 1 Phase 1.1~1.4 완료 처리

**산출물**

| 파일/산출물 | 내용 |
|---|---|
| `docs/stage1/phase1.1_batching-and-random.md` | get_batches 개념, shuffle 인덱스 동기화, set_seed 재현성 |
| `docs/stage1/phase1.2_io-and-checkpoints.md` | .npz 형식, save_params/load_params, checkpoints.save/load CuPy 변환 |
| `docs/stage1/phase1.3_training-plots.md` | plot_training_log, 2-subplot PNG 저장, plt.close 필요성 |
| `tests/stage1/test_checkpoints.py` | save/load 파일 생성, in-place 복원, .npz 확장자 자동 처리 |
| `notebooks/stage1/stage1-1_utils.ipynb` | 5개 유틸리티 실습 (batching/random/io/checkpoints/training_plots) |
| `_notebooks/` | 기존 notebooks/ 16개 아카이브 이동 |

**결정사항**

| 항목 | 결정 내용 |
|---|---|
| notebooks/ 관리 방식 | 기존 파일 `_notebooks/`로 이동, `notebooks/`는 템플릿 기반 신규 작성만 유지 |

## 260620 Stage 0 docs/ 전체 작성 완료

**완료 항목**
- Phase 0.0 문서 작성: `docs/stage0/phase0.0_legacy-analysis.md`
- Phase 0.1 문서 작성: `docs/stage0/phase0.1_conda-setup.md`
- Phase 0.2 문서 작성 2개: `phase0.2_implementation-plan.md`, `phase0.2_framework-interface.md`
- Phase 0.3 문서 작성: `docs/stage0/phase0.3_test-plan.md`
- README.md 갱신: src/ 구조, 실행 환경(conda run -n 형식·버전 표), 사용법, 테스트, 학습 로드맵
- PROJECT-TODO.md Stage 0 전체 완료 처리

**산출물**

| 파일/산출물 | 내용 |
|---|---|
| `docs/stage0/phase0.0_legacy-analysis.md` | 레거시 구조·패턴 분석 (common 6개, manual/module 패턴, task별 차이) |
| `docs/stage0/phase0.1_conda-setup.md` | conda 3환경 생성 명령 및 검증 (numpy_py311 / cuda118 / cuda121) |
| `docs/stage0/phase0.2_implementation-plan.md` | 레거시→src 매핑, src/ 구조, 책임 범위, Stage 1-6 Phase 분할 |
| `docs/stage0/phase0.2_framework-interface.md` | 공개 진입점·입출력 규약, task별 규약, 메서드 반환값 규약 |
| `docs/stage0/phase0.3_test-plan.md` | tests/ 구조, 파일별 테스트 항목, TDD 원칙, pytest 실행 명령 |
| `README.md` | src/ 구조 최신화, conda run -n 형식 통일, 학습 로드맵 Stage 0-6 |

**결정사항**

| 항목 | 결정 내용 |
|---|---|
| 확인된 환경 버전 | numpy_py311: NumPy 2.4.6 / cuda118: CuPy 13.6.0 / cuda121: CuPy 14.1.1 |
| 실행 명령 형식 | 모든 Python 명령 `conda run -n {환경명}` 형식으로 통일 |

## 260620 src/ 코드 변경 - config/task/experiment 삭제 및 checkpoints 이동

**완료 항목**
- `src/config.py` 삭제 - 기본값을 각 스크립트 `_DEFAULTS` 상수로 이동
- `src/task.py` 삭제 - `get_task_spec()` / `transform_targets()` 를 `src/data/mnist.py`로 흡수
- `src/core/experiment.py` 삭제 - scripts/ 4개를 직접 조립 방식으로 재작성
- `src/core/checkpoints.py` → `src/utils/checkpoints.py` git mv
- 연관 테스트 삭제 (test_config, test_task, test_experiment x2)
- import 갱신 (mlp, cnn, test_trainer, test_evaluator, test_predictor, test_checkpoints)
- pytest 전체 통과 확인 (351 passed, 8 skipped)

**산출물**

| 파일/산출물 | 내용 |
|---|---|
| `src/data/mnist.py` | `get_task_spec()`, `transform_targets()`, `_TASK_SPECS` 흡수, `_DATASET_DIR` 기본값 상수화 |
| `src/utils/checkpoints.py` | `src/core/` 에서 이동 (내용 무변경) |
| `scripts/train.py` | `Experiment` 제거, Dataloader/Model/Optimizer/Trainer/Evaluator 직접 조립 |
| `scripts/evaluate.py` | `Experiment` 제거, 직접 조립 |
| `scripts/predict.py` | `Experiment` 제거, 직접 조립 |
| `scripts/visualize.py` | `Experiment` 제거, 직접 조립 |

**결정사항**

| 항목 | 결정 내용 |
|---|---|
| `get_task_spec()` 이동 위치 | `src/data/mnist.py` — task 규약은 dataset과 함께 관리 |
| scripts/ 기본값 | `_DEFAULTS` 모듈 상수로 하드코딩 (`get_default_config()` 대체) |
| `Experiment` 대체 | 각 스크립트에서 직접 조립 (PROJECT-SPEC §6.3 기준 준수) |

## 260620 Stage 5·6 Phase 구조 재편 및 명칭 정비

**완료 항목**
- Stage 5 Phase 재편: 7개 → 5개 (Trainer+Evaluator 통합, Predictor+Visualizer 통합, Logger 신설)
- Stage 6 Phase 재편: 6개 → 4개 (학습+평가 통합, 예측+시각화 통합, 실험 배치 스크립트 신설)
- Phase 제목 명칭 정비: CLI → 스크립트, 영문 키워드 → 한글(학습 및 평가, 예측 및 시각화)
- Phase 제목 특수문자 수정: "·" → "및" (docs-rules.md 키보드 입력 불가 문자 금지 반영)
- docs-rules.md·template-rules.md 사용자 수정 반영

**산출물**

| 파일/산출물 | 내용 |
|---|---|
| `_core/PROJECT-SPEC.md` | Stage 5·6 Phase 목록 재편, src/core/logger.py 추가, Phase 명칭 정비 |
| `_core/PROJECT-TODO.md` | Stage 5·6 Phase Task 재편 및 번호 재조정 |

**결정사항**

| 항목 | 결정 내용 |
|---|---|
| Stage 5 Phase 수 | 7개 → 5개 (5.2 Trainer+Evaluator, 5.3 Predictor+Visualizer, 5.4 Logger, 5.5 노트북) |
| Stage 6 Phase 수 | 6개 → 4개 (6.1 학습+평가, 6.2 예측+시각화, 6.3 실험 배치, 6.4 노트북) |
| src/core/logger.py | Phase 5.4 신설 — epoch별 loss/metric CSV/dict 기록 |

## 260620 PROJECT-TODO.md 신규 Stage 0~7 구조 재작성

**완료 항목**
- PROJECT-TODO.md 신규 Stage 0~7 구조로 전면 재작성 (모든 체크박스 미완료 초기화)
- PROJECT-BOOK-PLAN.md 삭제 (원칙은 docs-rules.md, template-rules.md, PROJECT-SPEC.md에 반영 완료)
- PROJECT-SPEC.md Stage 7 삭제 → Stage 6 Phase 6.6에 실험 노트북 3개 통합
- PROJECT-SPEC.md Phase 0.2에 후속 프레임워크 공통 인터페이스 규약 문서 task 추가
- Phase 명칭 수정: 0.1 개발 환경 구성 / 2.1 MNIST 데이터 로딩 / 3.4 MLP 레이어 구현 / 3.5 CNN 레이어 구현
- Stage 2 도입 문장 수정: `src/data/` 패키지에 ~로 시작

**산출물**

| 파일/산출물 | 내용 |
|---|---|
| `_core/PROJECT-TODO.md` | Stage 0~7 구조 전면 재작성, 모든 체크박스 미완료 |
| `_core/PROJECT-SPEC.md` | Stage 7 삭제, Stage 6 Phase 6.6 노트북 4개로 확장, Phase 명칭 정비 |
| `_core/PROJECT-BOOK-PLAN.md` | 삭제 (원칙 반영 완료) |

**결정사항**

| 항목 | 결정 내용 |
|---|---|
| Stage 수 | 0~6 (7개) — Stage 7 삭제 |
| 실험 노트북 위치 | Stage 6 Phase 6.6 (cli + multiclass/binary/regression 비교 4개) |
| framework 연계 문서 | Phase 0.2 마지막 task로 배치 |
| 체크박스 상태 | 전체 미완료 — 문서·노트북 템플릿 기준 재작성 예정 |

## 260618 Stage 7 CNN 검증 및 문서화

**완료 항목**
- Phase 7.2 CNN output 3종 생성 결과 확인 및 result 문서 작성
- Phase 7.3~7.5 CNN tutorial 문서 작성
- CuPy checkpoint 저장/로드 호환 수정
- 루트 지침 문서의 Python 실행 환경 기준을 3개 conda environment 기준으로 갱신

**산출물**

| 파일/산출물 | 내용 |
|---|---|
| `docs/stage7/phase7.2_results.md` | MLP/CNN 6종 experiment 평가 결과와 산출물 경로 정리 |
| `docs/stage7/phase7.3_tutorial-cnn.md` | multiclass CNN 실행 절차와 평가 결과 정리 |
| `docs/stage7/phase7.4_tutorial-cnn.md` | binary CNN 실행 절차와 평가 결과 정리 |
| `docs/stage7/phase7.5_tutorial-cnn.md` | regression CNN 실행 절차와 평가 결과 정리 |
| `src/core/checkpoints.py` | CuPy parameter를 NumPy `.npz`로 저장하고 load 시 대상 parameter module로 변환 |
| `_core/PROJECT-TODO.md` | Phase 7.2~7.5 CNN 관련 항목 완료 처리 |

**결정사항**

| 항목 | 결정 내용 |
|---|---|
| Python 실행 환경 | MLP는 `numpy_py311`, `cupy_py311_cuda118`, `cupy_py311_cuda121` 중 목적에 맞게 실행하고 CNN은 `cupy_py311_cuda118` 또는 `cupy_py311_cuda121`에서 실행한다. |
| CNN 결과 수집 | Codex 환경에서는 GPU device가 노출되지 않아 사용자 WSL terminal의 `cupy_py311_cuda121` 실행 결과를 기준으로 문서화한다. |

## 260619 Stage 3 문서 번호 정합화 및 Visualizer 책임 분리

**완료 항목**
- Stage 3 metric phase를 `Phase 3.4`로 정리하고 MLP phase를 `Phase 3.5`로 이동
- `Visualizer`를 prediction 결과 시각화 전용 클래스로 축소
- training log 시각화를 `src/utils/training_plots.py` helper 함수로 분리
- 관련 테스트와 Stage 4/5 문서 갱신

**산출물**

| 파일/산출물 | 내용 |
|---|---|
| `docs/stage3/phase3.4_metrics.md` | metric 문서를 Phase 3.4 기준으로 rename |
| `docs/stage3/phase3.5_mlp.md` | MLP 문서를 Phase 3.5 기준으로 rename |
| `src/core/visualizer.py` | `plot_training_log` 제거, `plot_predictions` 전용 클래스 유지 |
| `src/utils/training_plots.py` | `plot_training_log(logs, output_dir, filename)` helper 추가 |
| `scripts/visualize.py` | training log helper와 `Visualizer` 조합 방식으로 변경 |
| `tests/stage1/test_training_plots.py` | training log helper 테스트 추가 |
| `tests/stage4/test_visualizer.py` | prediction visualization 테스트로 정리 |
| `_core/PROJECT-SPEC.md`, `_core/PROJECT-TODO.md` | Stage 3 phase 번호와 visualizer 책임 설명 갱신 |

**검증**

| 명령 | 결과 |
|---|---|
| `conda run -n numpy_py311 pytest tests/stage1/test_training_plots.py tests/stage4/test_visualizer.py tests/stage5/test_visualize.py -v` | 39 passed, 2 skipped |

**결정사항**

| 항목 | 결정 내용 |
|---|---|
| Visualizer 책임 | `Visualizer`는 prediction 이미지 grid 저장만 담당한다. |
| Training log plot | 학습 로그 그래프는 `src/utils/training_plots.py`의 helper 함수가 담당한다. |
| 호환성 | 기존 `Visualizer.plot_training_log(...)` 메서드는 유지하지 않고 제거한다. |

## 260619 docs/ 포털 문서 작성

**완료 항목**
- docs/index.md 최상위 진입 문서 작성 (8개 Stage 링크)
- docs/stage0~stage7/stageN.md Stage Chapter 소개 문서 8개 작성
- Phase 링크 형식 확정: `- [[filename|표시텍스트]]` 불릿 형식
- Obsidian 그래프 구조 개선: 38개 고립 노드 -> index-stage-phase 계층 클러스터

**산출물**

| 파일/산출물 | 내용 |
|---|---|
| `docs/index.md` | 전체 문서 진입점 - 8개 Stage 링크 및 한 줄 요약 테이블 |
| `docs/stage0/stage0.md` | Stage 0 Chapter 소개 - 3개 Phase 요약 및 링크 |
| `docs/stage1/stage1.md` | Stage 1 Chapter 소개 - 3개 Phase 요약 및 링크 |
| `docs/stage2/stage2.md` | Stage 2 Chapter 소개 - 3개 Phase 요약 및 링크 |
| `docs/stage3/stage3.md` | Stage 3 Chapter 소개 - 5개 Phase 요약 및 링크 |
| `docs/stage4/stage4.md` | Stage 4 Chapter 소개 - 7개 Phase 요약 및 링크 |
| `docs/stage5/stage5.md` | Stage 5 Chapter 소개 - 4개 Phase 요약 및 링크 |
| `docs/stage6/stage6.md` | Stage 6 Chapter 소개 - 3개 Phase 요약 및 링크 |
| `docs/stage7/stage7.md` | Stage 7 Chapter 소개 - 6개 Phase(10개 문서) 요약 및 링크 |

**결정사항**

| 항목 | 결정 내용 |
|---|---|
| 최상위 진입 문서 파일명 | `docs/index.md` - Obsidian Folder Notes 플러그인 연계, 그래프 허브 역할 명확 |
| Stage 소개 문서 파일명 | `docs/stageN/stageN.md` - 해당 폴더의 README 역할, Obsidian `[[stageN]]` 링크로 참조 |
| Phase 링크 형식 | `- [[filename|표시텍스트]]` 불릿 형식 - `->` 대비 시각적으로 자연스러움 |
| 링크 방향 | 상위 -> 하위 단방향 (docs-rules.md 준수), Phase 문서에 역방향 링크 없음 |

## 260619 Python 및 Markdown 규칙 보강과 코드 스타일 정합화

**완료 항목**
- 마크다운 문서의 UTF-8 인코딩 규칙과 생성 후 확인 절차 추가
- Python 파일 문자 사용 규칙과 `np.savez(path, **arrays)` 예외 규칙 추가
- `src/` 코드의 한국어 주석, 특수문자, 탭 들여쓰기 정리
- `tests/` 코드의 한국어 주석, 특수문자, 탭 들여쓰기 정리

**산출물**

| 파일/산출물 | 내용 |
|---|---|
| `_core/rules/docs-rules.md` | Markdown UTF-8 인코딩 규칙과 확인 명령 추가 |
| `_core/rules/python-rules.md` | 키보드 입력 가능 문자 사용 규칙과 라이브러리 API 예외 추가 |
| `src/` | 코드 로직 변경 없이 헤더, 주석, docstring 문자 규칙 정합화 |
| `tests/` | 테스트 로직 변경 없이 헤더, 주석, 들여쓰기 문자 규칙 정합화 |

**검증**

| 명령 | 결과 |
|---|---|
| `file -bi _core/rules/docs-rules.md` | `text/plain; charset=utf-8` |
| `conda run -n numpy_py311 python -m compileall -q src` | 통과 |
| `conda run -n numpy_py311 python -m compileall -q tests` | 통과 |
| `conda run -n numpy_py311 pytest tests -q` | 430 passed, 8 skipped, 16 warnings |

**결정사항**

| 항목 | 결정 내용 |
|---|---|
| 코드 로직 | 이번 변경은 로직을 바꾸지 않고 설명 텍스트와 들여쓰기만 정리한다. |
| `np.savez` | named argument unpacking을 요구하는 외부 라이브러리 API는 `**dict` 금지 규칙의 예외로 둔다. |
| 테스트 `__init__.py` | 빈 `tests/stage*/__init__.py` 파일 삭제는 별도 작업으로 남긴다. |

## 260619 교육용 노트북 전체 완성 (Phase 6.3 / 7.7)

**완료 항목**
- Phase 6.3 노트북 2개 작성 (stage6-1, stage6-2)
- Phase 7.7 노트북 3개 작성 (stage7-1, stage7-2, stage7-3)
- 총 16개 교육용 노트북 전체 완성

**산출물**

| 파일/산출물 | 내용 |
|---|---|
| `notebooks/stage6/stage6-1_cnn-architecture.ipynb` | im2col 원리, Conv2d/MaxPool2d shape 추적, MLP 파라미터 비교, Dropout train/eval 검증 |
| `notebooks/stage6/stage6-2_cnn-training.ipynb` | 3종 task CNN 학습, CuPy fallback 확인, MLP vs CNN 학습 곡선 비교 |
| `notebooks/stage7/stage7-1_multiclass-experiment.ipynb` | MLP+CNN 10 epoch 비교, 예측 grid, checkpoint 재평가, CLI 명령 |
| `notebooks/stage7/stage7-2_binary-experiment.ipynb` | target 변환 확인, sigmoid threshold, MLP vs CNN 비교, Multiclass 대비 차이 정리 |
| `notebooks/stage7/stage7-3_regression-experiment.ipynb` | R² 학습 곡선, round_clip 후처리, 3종 task 최종 비교 막대그래프, 프레임워크 연계 인터페이스 정리 |

## 260620 프레임워크 공통 Stage-Phase 목차 설계

**완료 항목**
- PROJECT-BOOK-PLAN.md 검토 및 프레임워크 공통 관점에서 Stage-Phase 재설계
- src/ 하위 폴더 기준 Stage 대응 원칙 확정
- Stage 0~7 (8단계) 목차 제안 — Stage 4 (models/) 신규, Stage 7 (실험 결과) 분리

**결정사항**

| 항목 | 결정 내용 |
|---|---|
| Stage = src/ 폴더 | src/ 하위 폴더 1개 = Stage 1개 (config/task/utils 묶음은 Stage 1) |
| Stage 3 처리 방침 | numpy 책에서는 직접 구현, 후속 프레임워크 책에서는 번호·제목 유지하고 내용만 교체 |
| Stage 4 신규 | models/ 전담 (MLP, CNN) — 현 Stage 3에서 분리 |
| Stage 5 | 현 Stage 4 (core/) 번호 이동, experiment.py 삭제에 따른 Phase 제거 |
| Stage 6 | 현 Stage 5 (scripts/) 번호 이동 + Phase 6.5 batch experiment 신규 |
| Stage 7 | 현 Stage 6 (실험 결과) 번호 이동 + MLP/CNN 문서 task별 비교 통합 |
| 실행 범위 | 이번 세션은 설계 확정만. 다음 세션에서 SPEC/TODO 재작성 + 코드 변경 실행 |

## 260620 Stage 3 + Stage 4 통합 재편 (Stage 3~7 → Stage 3~6)

**완료 항목**
- Stage 3(nn + MLP)와 Stage 4(CuPy CNN)를 Stage 3으로 통합 (7 Phase 구성)
- 기존 Stage 5~7을 Stage 4~6으로 재번호화
- PROJECT-SPEC.md, PROJECT-TODO.md Phase 구성 전면 재편
- tests/, notebooks/, docs/ 폴더 및 파일 git mv로 재배치

**산출물**

| 파일/산출물 | 내용 |
|---|---|
| `_core/PROJECT-SPEC.md` | §5.4 Stage 3 Phase 3.1~3.7 재편, §5.5 Stage 4(CNN) 삭제, §5.5~5.7 번호 재조정, §6.4/6.7/6.8 갱신 |
| `_core/PROJECT-TODO.md` | §4 Stage 3 Phase 3.1~3.7 재편, §5 Stage 4 삭제, §5~7 번호 재조정 |
| `tests/stage3/` | test_cnn.py, test_experiment.py 합류 (총 7개 파일) |
| `tests/stage4/`, `tests/stage5/` | 기존 stage5/, stage6/ 재번호화 |
| `notebooks/stage3/` | stage3-5_cnn-architecture.ipynb, stage3-6_cnn-training.ipynb 합류 |
| `notebooks/stage4/`, `notebooks/stage5/`, `notebooks/stage6/` | 기존 stage5~7/ 재번호화 (파일명 포함) |
| `docs/stage3/` | stage3.md 전면 재작성, phase3.6_conv.md·phase3.7_cnn.md 신규 작성 |
| `docs/stage4/`, `docs/stage5/`, `docs/stage6/` | 기존 docs/stage5~7/ 이동 및 파일명·내부 번호 갱신 |
| `docs/index.md` | Stage 4 행 삭제, Stage 5~7 → Stage 4~6 갱신 |

**결정사항**

| 항목 | 결정 내용 |
|---|---|
| Phase 4.2 (CNN-core integration) | 별도 Phase 불필요 — CNN Phase(Phase 3.5) 내 항목으로 흡수 |
| experiment.py 위치 | Stage 4(실행 객체)에 유지. CNN 분기 추가는 Phase 3.5 항목으로 기술 |
| scripts/*.py --model 플래그 | CLI Phase(Stage 5) 귀속 유지 |
| 전체 Stage 수 | 0~6 (7개) |

## 260619 교육용 노트북 체계 구축 (Phase 1.4/2.4/3.6 일부)

**완료 항목**
- PROJECT-SPEC.md §3 범위, §5 진입부, §5.2~5.8 Phase 목록, §6.8 notebooks 구조 갱신
- PROJECT-TODO.md Phase 1.4, 2.4, 3.6, 4.8, 5.5, 6.3, 7.7 섹션 신설
- `notebooks/stage1/stage1-1_config-and-task.ipynb` 작성 (Phase 1.4)
- `notebooks/stage2/stage2-1_mnist-loading.ipynb` 작성 (Phase 2.4)
- `notebooks/stage2/stage2-2_dataset-and-dataloader.ipynb` 작성 (Phase 2.4)
- `notebooks/stage3/stage3-1_activations.ipynb` 작성 (Phase 3.6)

**산출물**

| 파일/산출물 | 내용 |
|---|---|
| `_core/PROJECT-SPEC.md` | §3 노트북 항목 추가, §5 4단계 워크플로우 명시, §5.2~5.8 노트북 Phase 추가, §6.8 notebooks 폴더 구조 신설 |
| `_core/PROJECT-TODO.md` | Phase 1.4/2.4/3.6/4.8/5.5/6.3/7.7 체크박스 항목 추가 |
| `notebooks/stage1/stage1-1_config-and-task.ipynb` | config dict, 3 task spec 비교, target 변환, 분포 시각화 |
| `notebooks/stage2/stage2-1_mnist-loading.ipynb` | load_mnist shape/dtype 확인, 샘플 grid, 픽셀 histogram |
| `notebooks/stage2/stage2-2_dataset-and-dataloader.ipynb` | MNISTDataset 3종 비교, Dataloader 배치/shuffle 검증 |
| `notebooks/stage3/stage3-1_activations.ipynb` | 4종 활성화 함수 그래프, 수치 안정성, 배치 입출력 확인 |

**결정사항**

| 항목 | 결정 내용 |
|---|---|
| 노트북 위치 | 각 Stage 마지막 Phase (1.4/2.4/3.6/4.8/5.5/6.3/7.7) |
| 파일명 규칙 | `stage{N}-{순번}_{kebab-case-keywords}.ipynb` |
| 노트북 총 개수 | 16개 (Stage 1~7), 이번 세션 4개 완료 |
| 실행 환경 | `numpy_py311` 기본, CNN은 CuPy try/except fallback |
