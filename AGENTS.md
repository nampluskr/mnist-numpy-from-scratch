---
tags: [project, agent-config]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---

# AGENTS.md — {{PROJECT_NAME}}

이 파일은 Codex가 이 프로젝트를 다룰 때 따르는 진입점 지침이다.
상세 규칙은 `_core/` 를 참조한다.

## 1. 프로젝트 개요

- **이름**: {{PROJECT_NAME}}
- **목적**: {{PROJECT_DESCRIPTION}}
- **템플릿**: `project-coding-template`

## 2. 프로젝트 구조

```text
{{PROJECT_SLUG}}/
├── _core/        # 프로젝트 운영 파일
├── data/         # 데이터 (경로 또는 설명)
├── src/          # 소스 코드
├── scripts/      # 실행 스크립트
├── notebooks/    # Jupyter 노트북
├── tests/        # 테스트 코드
├── configs/      # 설정 파일
├── experiments/  # 실험 기록
├── docs/         # 프로젝트 문서
└── outputs/      # 결과물 (모델, 그래프, 예측 등)
```

## 3. 핵심 행동 규칙

- 작업 전 `_core/rules/agent-rules.md` 와 `_core/rules/coding-rules.md` 를 참조한다.
- 마크다운 문서 작성·수정 시 `_core/rules/docs-rules.md` 를 따른다.
- Python 스크립트 작성·수정 시 `_core/rules/python-rules.md` 를 따른다.
- 루트 폴더 구조는 변경하지 않는다.

## 4. 응답 스타일

- 한국어 경어체로 응답한다.
- 변경한 파일과 결과를 간단히 요약한다.
