---
tags: [project, docs]
created: YYYY-MM-DD
updated: 2026-06-15
---

# coding-guide.md

이 프로젝트의 코딩 작업 방식을 설명한다.
사용자가 참조하는 문서이다. 에이전트 행동 규칙은 `_core/rules/coding-rules.md` 를 참조한다.

## 1. 목적

`src/` 와 `experiments/` 를 중심으로 코드를 작성·정제·기록한다.

- `notebooks/` 에서 탐색·시각화·프로토타입을 수행한다.
- 재사용 가능한 코드는 `src/` 로 정제한다.
- 실험은 `experiments/` 에 기록하고 결과는 `outputs/` 에 저장한다.

## 2. 자료 흐름

자료는 다음 순서로 이동한다.

```text
data/ → notebooks/ → src/ → scripts/ → experiments/ → outputs/
```

각 단계의 역할은 다음과 같다.

| 단계 | 폴더 | 의미 |
|---|---|---|
| Data | `data/` | 원본 데이터 참조 또는 경로 관리 |
| Exploration | `notebooks/` | 탐색·시각화·프로토타입 |
| Implementation | `src/` | 정제된 코드를 모듈로 구조화 |
| Execution | `scripts/` | `src/core/` 실행 객체를 호출하는 CLI 진입점 |
| Experiment | `experiments/` | 실험 실행 기록 및 메타데이터 |
| Output | `outputs/` | 최종 결과물 저장 |

## 3. 폴더별 역할

### src/

재사용 가능한 모듈·클래스·함수를 둔다.

- 프로토타입 코드는 두지 않는다. 먼저 `notebooks/` 에서 검증한다.
- 파일 첫 줄에 한 줄 한국어 주석으로 역할을 기록한다.
- 패키지 구조가 필요하면 `__init__.py` 를 둔다.

### scripts/

`src/core/` 실행 객체를 호출하는 CLI 진입점 스크립트를 둔다.

- `train.py`, `evaluate.py`, `predict.py`, `visualize.py` 네 파일로 구성한다.
- 내부 구현 모듈을 직접 조립하지 않고 `src/core/` 실행 객체를 참조한다.
- 스크립트 실행 결과는 `experiments/` 에 기록하고 결과 파일은 `outputs/` 에 저장한다.

### notebooks/

탐색·시각화·프로토타입 노트북을 둔다.

- 노트북 파일명 형식: `NNN_설명.ipynb` (예: `001_data-exploration.ipynb`)
- 완성된 로직은 `src/` 로 이동한다.
- 노트북 자체는 테스트 대상이 아니다.

### experiments/

실험을 기록한다.

- 폴더명 형식: `YYYYMMDD_실험명` (예: `20260615_baseline-lr-sweep`)
- 각 실험 폴더에 `README.md` 를 두어 목적, 설정, 결과, 재현 방법을 요약한다.
- 결과 파일은 `outputs/` 에 저장하고 경로를 `README.md` 에 기록한다.

### outputs/

최종 결과물을 저장한다.

- 모델, 그래프, 예측 결과, 내보내기 파일을 둔다.
- 어느 실험에서 생성된 것인지 추적 가능해야 한다.

## 4. 작업 흐름 권장 순서

1. `data/` 에 데이터 경로 또는 설명을 기록한다.
2. `notebooks/` 에서 데이터를 탐색하고 접근 방식을 검증한다.
3. 재사용 로직을 `src/` 로 이동한다.
4. `scripts/` 에 `src/core/` 실행 객체를 호출하는 CLI 스크립트를 작성한다.
5. `experiments/` 에 실험 폴더를 생성하고 `README.md` 를 작성한다.
6. `scripts/` 를 통해 실험을 실행하고 결과를 `outputs/` 에 저장한다.
7. 실험 `README.md` 에 결과 요약과 출력 경로를 기록한다.

## 5. 준수 규칙 및 참조 경로

코드와 문서 작성 시 다음 규칙을 반드시 준수한다.

| 항목 | 경로 |
|---|---|
| 에이전트 코딩 행동 규칙 | `_core/rules/coding-rules.md` |
| Python 코드 규칙 | `_core/rules/python-rules.md` |
| 마크다운 문서 규칙 | `_core/rules/docs-rules.md` |
| 프로젝트 구조 가이드 | `_core/docs/project-guide.md` |
