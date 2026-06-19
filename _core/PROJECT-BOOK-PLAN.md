---
tags: [project, docs, book, plan]
created: 2026-06-20
updated: 2026-06-20
---

# 책 목차 기준 Stage-Phase 재구성 계획

## 1. 목적

이 문서는 `docs/` 전체를 하나의 책으로 구성할 때 적용할 Stage-Phase 재구성 계획을 기록한다.
Stage는 Chapter, Phase는 Section에 대응하며, `_core/PROJECT-SPEC.md`, `_core/PROJECT-TODO.md`, `docs/`가 동일한 목차를 사용하도록 정리하는 것을 목표로 한다.

이 문서는 아직 반영되지 않은 계획안이다.
구조 변경을 실행하기 전 내용을 보완하고, 확정된 항목부터 SPEC, TODO, 문서에 순서대로 반영한다.

## 2. 확정 원칙

책 구조를 재구성할 때 적용할 원칙은 다음과 같다.

| 항목 | 원칙 |
|---|---|
| Stage | 책의 Chapter에 대응한다. |
| Phase | Chapter 내부의 Section에 대응한다. |
| Phase 문서 | Phase마다 대표 Markdown 문서 1개를 둔다. |
| H1 제목 | `docs/`의 Section 문서와 `notebooks/`의 H1에는 Phase, Section, Chapter 번호를 넣지 않고 주제 텍스트만 사용한다. |
| TODO | Phase 아래에서 코드, 테스트, 문서, 노트북 작업을 관리한다. |
| 노트북 | 각 Chapter 마지막의 독자용 실습 Section으로 구성한다. |
| Obsidian | `docs/`를 Obsidian에서 직접 탐색하고 읽을 수 있는 Markdown 문서 구조로 유지한다. |
| Jupyter Book | 동일한 `docs/`와 `notebooks/`를 Jupyter Book v2 기반 web book으로 build한다. |
| 목차 | Jupyter Book v2의 `myst.yml`에 Chapter-Section 순서를 명시한다. |
| PDF | Jupyter Book v2의 MyST export와 LaTeX book template을 사용하여 최종 PDF를 생성한다. |
| Chapter 0 | Section 번호를 `0.1`부터 시작한다. |
| 문서 표현 | `구현`, `문서 작성`, `노트북 작성`보다 독자가 학습할 주제를 중심으로 제목을 정한다. |

## 3. 현재 구조의 검토 결과

현재 구조에서 보완할 항목은 다음과 같다.

1. 개발 절차와 책 목차가 같은 Phase 수준에 혼재되어 있다.
2. Stage 3은 하나의 Phase가 여러 문서와 대응하여 Phase-Section 관계가 일치하지 않는다.
3. Stage 6 tutorial은 하나의 Phase가 MLP와 CNN 문서 두 개로 분리되어 있다.
4. Stage 3에서 Stage 4의 `Experiment`를 먼저 참조하여 학습 순서가 역전된다.
5. `experiments/run_*.py`의 batch experiment 역할이 목차와 TODO에 독립적으로 반영되지 않았다.
6. `Phase 0.0`과 TODO의 외부 번호가 함께 사용되어 책의 장절 번호가 직관적이지 않다.
7. `PROJECT-SPEC.md`, `PROJECT-TODO.md`, `docs/index.md`, Stage overview, Phase 문서의 번호와 제목에 이전 Stage 구조가 남아 있다.
8. 노트북 Phase 제목이 독자용 실습이 아니라 작성 작업으로 표현되어 있다.

## 4. 목표 목차

재구성 후 사용할 Chapter와 Section은 다음과 같다.

### 4.1. Chapter 0 환경 구성과 프로젝트 설계

| Section | 제목 |
|---|---|
| 0.1 | conda 실행 환경 |
| 0.2 | 레거시 코드 분석 |
| 0.3 | package 구조와 구현 계획 |
| 0.4 | 테스트 전략 |

Chapter 0은 Jupyter 실행 환경을 준비하는 단계이므로 실습 notebook을 두지 않는다.

### 4.2. Chapter 1 config와 task 규약

| Section | 제목 |
|---|---|
| 1.1 | config 구성 |
| 1.2 | task 규약과 target 변환 |
| 1.3 | 공통 utility |
| 1.4 | Chapter 1 실습 |

Chapter 1의 실습 notebook 제목은 다음과 같다.

1. Config, Task and Utilities

### 4.3. Chapter 2 MNIST data pipeline

| Section | 제목 |
|---|---|
| 2.1 | MNIST raw data loading |
| 2.2 | Dataset과 target 변환 |
| 2.3 | DataLoader와 mini-batch |
| 2.4 | Chapter 2 실습 |

Chapter 2의 실습 notebook 제목은 다음과 같다.

1. MNIST Raw Data Loading
2. Dataset and DataLoader

### 4.4. Chapter 3 nn module과 model

| Section | 제목 |
|---|---|
| 3.1 | activation 함수 |
| 3.2 | loss와 gradient |
| 3.3 | metric 함수 |
| 3.4 | layer와 Module interface |
| 3.5 | MLP model |
| 3.6 | convolution과 pooling layer |
| 3.7 | CNN model |
| 3.8 | Chapter 3 실습 |

Chapter 3은 model interface까지 설명한다.
전체 data, model, trainer 조립과 epoch 학습은 Chapter 4의 `Experiment`에서 다룬다.

Chapter 3의 실습 notebook 제목은 다음과 같다.

1. Activation Functions
2. Losses and Metrics
3. Layer Modules
4. MLP Model
5. CNN Architecture
6. CNN Model

### 4.5. Chapter 4 학습 및 실행 framework

| Section | 제목 |
|---|---|
| 4.1 | optimizer |
| 4.2 | checkpoint |
| 4.3 | Trainer |
| 4.4 | Evaluator |
| 4.5 | Predictor |
| 4.6 | Experiment |
| 4.7 | visualization |
| 4.8 | Chapter 4 실습 |

Chapter 4의 실습 notebook 제목은 다음과 같다.

1. Optimizers
2. Trainer and Evaluator
3. Experiment, Predictor and Checkpoint

### 4.6. Chapter 5 CLI와 experiment 자동화

| Section | 제목 |
|---|---|
| 5.1 | training CLI |
| 5.2 | evaluation CLI |
| 5.3 | prediction CLI |
| 5.4 | visualization CLI |
| 5.5 | batch experiment 실행 |
| 5.6 | Chapter 5 실습 |

Section 5.5는 `experiments/run_all.py`, `run_train.py`, `run_evaluate.py`, `run_predict.py`, `run_visualize.py`의 역할과 실행 흐름을 설명한다.

Chapter 5의 실습 notebook 제목은 다음과 같다.

1. CLI and Batch Experiments

### 4.7. Chapter 6 experiment 결과와 framework 연계

| Section | 제목 |
|---|---|
| 6.1 | 실험 조건과 결과 개요 |
| 6.2 | multiclass MLP-CNN 비교 |
| 6.3 | binary MLP-CNN 비교 |
| 6.4 | regression MLP-CNN 비교 |
| 6.5 | framework interface와 migration |
| 6.6 | Chapter 6 실습 |

Section 6.2부터 6.4까지는 MLP와 CNN 문서를 각각 분리하지 않고 task별 대표 문서 하나에서 두 model을 비교한다.

Chapter 6의 실습 notebook 제목은 다음과 같다.

1. Multiclass Experiment: MLP vs CNN
2. Binary Experiment: MLP vs CNN
3. Regression Experiment: MLP vs CNN

## 5. 출판 및 build 구조

책의 단일 원본은 `docs/`의 Markdown 문서와 `notebooks/`의 Jupyter Notebook으로 구성한다.
같은 원본을 Obsidian, Jupyter Book web build, LaTeX 기반 PDF export에서 함께 사용한다.

### 5.1. Obsidian 문서 구조

`docs/`는 Obsidian vault에서 바로 열람할 수 있는 책 본문으로 유지한다.

1. `docs/index.md`를 책의 최상위 진입점으로 사용한다.
2. `docs/stageN/stageN.md`를 Chapter landing page로 사용한다.
3. `docs/stageN/phaseN.x_*.md`를 Section 본문으로 사용한다.
4. YAML frontmatter, heading, table, code block은 Obsidian과 MyST가 함께 처리할 수 있는 Markdown 범위로 작성한다.
5. 내부 링크는 Obsidian과 MyST가 모두 해석할 수 있는 표준 relative Markdown link를 기준으로 통일한다.
6. 기존 Obsidian wikilink를 유지해야 한다면 Jupyter Book build 전 변환 절차를 별도로 마련한다.

현재 `_core/rules/docs-rules.md`는 Obsidian wikilink 사용을 규정한다.
표준 relative Markdown link로 전환하려면 실행 전에 사용자 승인을 받아 해당 규칙을 함께 변경해야 한다.

### 5.2. Jupyter Book v2 설정과 목차

Jupyter Book v2의 MyST engine을 기준으로 root에 단일 `myst.yml`을 생성한다.
Jupyter Book v1 호환용 `_toc.yml`이나 별도 `toc.yml`은 사용하지 않는다.

`myst.yml`에서 관리할 항목은 다음과 같다.

| 설정 항목 | 책임 |
|---|---|
| project metadata | 책 제목, 저자, repository와 실행 환경 정보 |
| `project.toc` | `docs/index.md`부터 Chapter, Section, 실습 notebook까지의 순서와 계층 |
| site | Jupyter Book web build와 navigation 설정 |
| exports | LaTeX source와 PDF output 설정 |

`myst.yml`의 `project.toc`에는 `docs/stageN/stageN.md`를 Chapter로 배치하고, Phase Markdown 문서와 관련 notebook을 `children`으로 연결한다.

### 5.3. notebook 실습 Section

`notebooks/`의 `.ipynb` 파일은 별도 부록이 아니라 각 Chapter의 마지막 실습 Section에 포함한다.
각 notebook은 이전 notebook의 실행 상태 없이도 독립적인 강의 또는 실습 자료로 사용할 수 있어야 한다.
프로젝트의 `src/`, 지정된 conda 환경, 로컬 MNIST dataset은 공통 실행 조건으로 사용할 수 있다.

1. Chapter 실습 대표 Markdown 문서에서 학습 목표, 선수 Section, notebook 실행 순서를 안내한다.
2. 각 notebook은 `myst.yml`의 `project.toc`에서 해당 실습 Section의 child page로 등록한다.
3. notebook의 첫 Markdown cell에는 Jupyter Book에 표시할 title과 필요한 MyST frontmatter를 둔다.
4. web과 PDF에서 재현할 수 있도록 실행 환경, kernel, dataset, GPU 요구사항을 명시한다.
5. interactive output이 PDF에서 표현되지 않을 경우 static image 또는 table fallback을 제공한다.
6. 이전 notebook에서 정의한 변수, 함수, checkpoint, output file과 kernel state에 의존하지 않는다.
7. 필요한 import, path 설정, random seed, data loading과 output directory 준비를 notebook 내부에서 수행한다.
8. 학습 목표, 선수 지식, 환경 설정, 개념 설명, 단계별 실습, 결과 검증, 정리와 추가 과제를 공통 구성으로 사용한다.
9. 파일명은 `stage{N}-{순번}_{주제}.ipynb` 형식을 사용하고 Chapter마다 순번을 1부터 시작한다.
10. 같은 Chapter의 notebook은 번호 누락과 중복 없이 순차적이고 단계적인 학습 흐름으로 구성한다.
11. 첫 Markdown cell의 H1에는 `Chapter N-M`, `Phase N.M`, `Section N.M`과 같은 구분자나 번호를 넣지 않고 주제 텍스트만 작성한다.
12. 학습 순서와 번호는 파일명과 Jupyter Book TOC에서 관리하고, H1 주제는 해당 파일명과 의미가 일치하도록 작성한다.
13. Chapter 3 notebook은 Chapter 4에서 설명하는 `Experiment`에 의존하지 않고 nn module과 model interface 범위에서 완결한다.

### 5.4. Chapter별 notebook 구성

Chapter 0은 Jupyter 실행 환경을 구성하기 전 단계이므로 별도 notebook을 두지 않는다.
Chapter 1부터 Chapter 6까지 사용할 목표 notebook 파일은 다음과 같다.

| Chapter | 순번 | 목표 파일명 | 주요 학습 내용 |
|---|---:|---|---|
| Chapter 1 | 1 | `stage1-1_config-task-and-utils.ipynb` | config, task 규약, target 변환, 공통 utility |
| Chapter 2 | 1 | `stage2-1_mnist-loading.ipynb` | MNIST raw data loading과 data 확인 |
| Chapter 2 | 2 | `stage2-2_dataset-and-dataloader.ipynb` | Dataset, target 변환, DataLoader |
| Chapter 3 | 1 | `stage3-1_activations.ipynb` | activation 함수와 시각화 |
| Chapter 3 | 2 | `stage3-2_losses-and-metrics.ipynb` | task별 loss, gradient, metric |
| Chapter 3 | 3 | `stage3-3_layers.ipynb` | Module, Linear, activation layer, Sequential |
| Chapter 3 | 4 | `stage3-4_mlp.ipynb` | MLP forward, backward와 parameter update |
| Chapter 3 | 5 | `stage3-5_cnn-architecture.ipynb` | im2col, convolution, pooling과 CNN 구조 |
| Chapter 3 | 6 | `stage3-6_cnn-model.ipynb` | CNN forward, backward와 model interface |
| Chapter 4 | 1 | `stage4-1_optimizers.ipynb` | SGD, Adam과 learning rate 비교 |
| Chapter 4 | 2 | `stage4-2_trainer-and-evaluator.ipynb` | Trainer, Evaluator와 epoch log |
| Chapter 4 | 3 | `stage4-3_experiment.ipynb` | Experiment, Predictor, checkpoint와 visualization |
| Chapter 5 | 1 | `stage5-1_cli-and-batch-experiments.ipynb` | CLI scripts와 batch experiment 실행 |
| Chapter 6 | 1 | `stage6-1_multiclass-experiment.ipynb` | multiclass MLP-CNN 비교 실험 |
| Chapter 6 | 2 | `stage6-2_binary-experiment.ipynb` | binary MLP-CNN 비교 실험 |
| Chapter 6 | 3 | `stage6-3_regression-experiment.ipynb` | regression MLP-CNN 비교 실험 |

기존 16개 notebook의 수는 유지한다.
Stage 3의 `layers`와 `losses-and-metrics` notebook은 목표 목차 순서에 맞게 재번호화한다.
기존 `stage3-6_cnn-training.ipynb`는 `Experiment` 의존성을 제거하고 `stage3-6_cnn-model.ipynb`로 개편한다.
Stage 1 notebook은 공통 utility 실습을 포함하도록 확장하고, Stage 5 notebook은 `experiments/run_*.py`의 batch 실행 흐름을 포함하도록 확장한다.
기존 notebook 내부에 남은 Stage 6, Stage 7 제목은 목표 파일명과 Chapter 번호에 맞게 수정한다.

### 5.5. LaTeX 기반 PDF export

최종 문서는 Jupyter Book v2의 MyST export가 전체 book을 LaTeX로 변환한 뒤 PDF를 생성하는 방식으로 build한다.

1. `myst.yml`에 multi-article book export를 정의한다.
2. LaTeX book template은 `plain_latex_book`을 초기 기준으로 사용한다.
3. Chapter landing page는 LaTeX `chapter`, Phase 문서와 notebook은 `section` 수준으로 매핑한다.
4. LaTeX source와 PDF를 모두 검증할 수 있는 export 구성을 사용한다.
5. 최종 PDF는 `outputs/book/mnist-numpy-from-scratch.pdf`에 저장한다.
6. 한글 font, code block, 수식, 표, 그림, notebook output, page break를 PDF 검증 항목에 포함한다.

## 6. 문서별 변경 방향

### 6.1. PROJECT-SPEC.md

`PROJECT-SPEC.md`에는 책 구조의 기준과 전체 목차를 정의한다.

1. Stage는 Chapter, Phase는 Section이라는 규칙을 명시한다.
2. Phase마다 대표 Markdown 문서 하나를 둔다는 규칙을 추가한다.
3. `코드 구현 -> 테스트 -> 문서 -> 노트북`은 Phase 구성이 아니라 각 Section의 작업 흐름으로 설명한다.
4. Stage 3의 구현과 문서 분리 Phase를 학습 주제별 Section으로 변경한다.
5. Stage 5에 batch experiment Section을 추가한다.
6. Stage 6의 제목과 Phase를 experiment 비교 중심으로 변경한다.

### 6.2. PROJECT-TODO.md

`PROJECT-TODO.md`는 책 목차와 같은 Stage-Phase 제목을 사용하고, 각 Phase의 산출물을 체크박스로 관리한다.

1. Stage 0의 Phase를 `0.1`부터 `0.4`까지로 변경한다.
2. Stage 번호와 TODO H2 번호를 일치시킨다.
3. 구현, 테스트, 문서 작성 항목을 해당 학습 주제 Phase 아래에 함께 배치한다.
4. 노트북 작성 Phase를 독자용 Chapter 실습 Phase로 변경한다.
5. Stage 3의 `test_experiment.py` 책임은 Chapter 4의 Experiment 통합 테스트로 이동한다.
6. Stage 5에 `experiments/run_*.py`와 관련 검증 항목을 추가한다.
7. Stage 6의 task별 MLP-CNN 문서 통합 작업을 반영한다.

### 6.3. docs와 notebooks

`docs/`와 `notebooks/`는 확정된 SPEC과 TODO를 기준으로 동기화한다.

1. `docs/index.md`를 Chapter 0부터 Chapter 6까지의 최상위 목차로 갱신한다.
2. 각 `docs/stageN/stageN.md`가 해당 Chapter의 Section 목록을 제공하도록 수정한다.
3. Phase 문서 파일명과 frontmatter tag의 Stage와 Phase 번호를 일치시킨다.
4. Section 문서와 notebook의 H1에서는 Phase, Section, Chapter 번호를 제거하고 주제 텍스트만 남긴다.
5. 각 실습 Phase에 노트북의 목표와 실행 순서를 안내하는 대표 Markdown 문서를 둔다.
6. Stage 3 실습에서는 Chapter 4에서 설명할 `Experiment`의 선행 사용을 제거한다.
7. Stage 6의 MLP와 CNN tutorial 문서를 task별 비교 문서로 통합한다.
8. 이전 Stage 번호와 존재하지 않는 문서를 가리키는 링크를 제거한다.
9. Obsidian과 MyST가 함께 처리할 수 있는 Markdown 문법과 link 형식으로 통일한다.
10. 각 Chapter와 Section을 `myst.yml`의 Jupyter Book v2 목차와 연결한다.

## 7. 실행 순서

목차 재구성은 다음 순서로 실행한다.

1. 이 계획 문서를 보완하고 목표 목차를 최종 확정한다.
2. `PROJECT-SPEC.md`의 Stage-Phase 구조를 변경한다.
3. `PROJECT-TODO.md`를 같은 구조로 재배치한다.
4. `docs/index.md`와 Stage overview를 갱신한다.
5. Phase 문서를 이동, 통합, 재번호화한다.
6. Section 문서와 notebook의 H1에서 Phase, Section, Chapter 번호를 제거한다.
7. Chapter 실습 문서와 노트북 연결을 정리한다.
8. Obsidian과 MyST의 공통 Markdown 및 link 규약을 적용한다.
9. root에 목차와 export 설정을 포함한 단일 `myst.yml`을 생성한다.
10. Jupyter Book v2 web book을 build하고 목차와 notebook rendering을 검증한다.
11. LaTeX book export를 설정하고 최종 PDF를 생성한다.
12. Obsidian link, Jupyter Book link, 문서 제목과 TOC를 검사한다.
13. 관련 테스트 경로와 문서 참조를 정리한다.
14. 전체 문서 정합성, build 결과와 테스트 결과를 확인한다.

## 8. 완료 기준

재구성 완료 기준은 다음과 같다.

| 기준 | 완료 조건 |
|---|---|
| 목차 일치 | SPEC, TODO, docs index, Stage overview의 Chapter와 Section 제목이 일치한다. |
| 문서 대응 | 모든 Phase가 대표 Markdown 문서 하나와 대응한다. |
| 실습 연결 | 모든 노트북이 해당 Chapter 실습 문서에서 연결된다. |
| Notebook 독립성 | 각 notebook이 이전 notebook의 kernel state와 output 없이 clean kernel에서 실행된다. |
| Notebook 구성 | 모든 notebook이 학습 목표부터 추가 과제까지의 공통 강의 구조를 갖는다. |
| Notebook 번호 | Chapter별 파일 번호가 1부터 시작하고 중복이나 누락 없이 이어진다. |
| H1 제목 | Section 문서와 notebook H1에 Phase, Section, Chapter 번호가 없고 주제 텍스트만 남아 있다. |
| Notebook 제목 | notebook H1의 주제 텍스트가 파일명 및 Jupyter Book TOC 제목과 의미상 일치한다. |
| 학습 순서 | 뒤 Chapter에서 설명할 public interface를 앞 Chapter에서 필수로 사용하지 않는다. |
| 링크 정합성 | 존재하지 않는 Stage, Phase, 문서 링크가 없다. |
| 고립 문서 | `docs/`에 상위 문서에서 연결되지 않은 고립 문서가 없다. |
| Obsidian | `docs/index.md`에서 모든 Chapter와 Section을 탐색할 수 있다. |
| Jupyter Book 설정 | 단일 `myst.yml`이 Chapter-Section-notebook 구조를 정의한다. |
| Web build | Jupyter Book v2 web build가 오류 없이 완료되고 목차 순서가 목표 목차와 일치한다. |
| Notebook | 각 notebook이 해당 Chapter 실습 Section에 표시되고 static output이 정상적으로 rendering된다. |
| LaTeX export | 전체 book의 LaTeX source가 생성되고 Chapter와 Section level이 올바르게 적용된다. |
| PDF | `outputs/book/mnist-numpy-from-scratch.pdf`가 생성되고 한글, 코드, 수식, 표와 그림이 정상 출력된다. |
| 문서 규칙 | `_core/rules/docs-rules.md`를 준수한다. |
| 코드 영향 | 목차 재구성만으로 public API를 변경하지 않는다. |
| 테스트 | 문서 경로 변경 후 전체 pytest 수집과 테스트가 통과한다. |

## 9. 보류 사항

실행 전에 추가로 검토할 항목은 다음과 같다.

1. 통합 후 제거할 기존 Phase 문서의 Git 이력 보존 방식
2. Chapter 실습 대표 문서의 공통 구성 형식
3. `experiments/run_*.py`의 output directory와 checkpoint 재사용 규약
4. 중복된 `tests/stage3/test_experiment.py`와 `tests/stage4/test_experiment.py`의 통합 범위
5. 후속 PyTorch, TensorFlow, JAX 프로젝트에서 유지할 Chapter-Section 구조
6. Obsidian wikilink를 표준 relative Markdown link로 전환할지 여부와 docs rule 변경 승인
7. Jupyter Book v2와 MyST CLI의 설치 환경 및 version 고정 방식
8. LaTeX distribution, 한글 font와 `plain_latex_book` template의 세부 설정

## 10. 참고 자료

Jupyter Book v2와 MyST 설정을 실행할 때 확인할 공식 문서는 다음과 같다.

- [MyST Table of Contents](https://mystmd.org/guide/table-of-contents)
- [MyST Jupyter Notebook](https://mystmd.org/guide/interactive-notebooks)
- [MyST PDF 생성](https://mystmd.org/guide/creating-pdf-documents)
- [MyST export 개요](https://mystmd.org/guide/documents-exports)
