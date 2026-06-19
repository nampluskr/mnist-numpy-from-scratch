---
tags: [project, templates, docs, notebooks]
created: 2026-06-20
updated: 2026-06-20
---

# 문서와 notebook 템플릿 사용 가이드

## 1. 목적

이 폴더는 책의 Section 문서와 독립 실행형 tutorial notebook을 일정한 형식으로 작성하기 위한 원본 템플릿을 보관한다.
템플릿은 `_core/`의 운영 파일이며 Jupyter Book 목차에는 포함하지 않는다.

## 2. 템플릿 목록

제공하는 템플릿과 용도는 다음과 같다.

| 파일 | 용도 |
|---|---|
| `book-section-template.md` | API 구현, 환경 구성, 설계, 실험 결과를 설명하는 책의 Section 문서 |
| `tutorial-notebook-template.ipynb` | code, graph, 설명과 검증을 포함하는 독립 실행형 tutorial notebook |

## 3. 템플릿 선택

책의 Section 문서는 모두 `book-section-template.md`를 사용한다.
구현된 함수나 클래스를 다루는 경우 API reference를 작성하고, API가 없는 환경·설계·실험 문서는 같은 위치를 구성 절차 또는 실험 규약으로 바꾼다.

실습 notebook은 `tutorial-notebook-template.ipynb`를 사용한다.
Notebook은 이전 notebook의 변수, checkpoint, output과 kernel state에 의존하지 않아야 한다.

## 4. 복사와 이름 변경

Section 문서를 만들 때는 템플릿을 대상 Stage 폴더로 복사한다.

```bash
cp _core/templates/book-section-template.md docs/stageN/phaseN.x_topic.md
```

Tutorial notebook을 만들 때는 Chapter 내부의 학습 순서가 드러나는 이름으로 복사한다.

```bash
cp _core/templates/tutorial-notebook-template.ipynb notebooks/stageN/stageN-M_topic.ipynb
```

파일명 규칙은 다음 기준을 사용한다.

| 대상 | 형식 | 예시 |
|---|---|---|
| Section 문서 | `phase{N}.{M}_{topic}.md` | `phase3.1_activations.md` |
| Tutorial notebook | `stage{N}-{M}_{topic}.ipynb` | `stage3-1_activations.ipynb` |

## 5. Placeholder 교체

템플릿을 복사한 뒤 아래 placeholder를 실제 값으로 바꾼다.

| Placeholder | 의미 |
|---|---|
| `{{STAGE_TAG}}` | Stage tag, 예: `stage3` |
| `{{TOPIC_TAG}}` | 주제 tag, 예: `activations` |
| `{{CREATED_DATE}}` | 최초 작성일, `YYYY-MM-DD` |
| `{{UPDATED_DATE}}` | 최종 수정일, `YYYY-MM-DD` |
| `{{TOPIC_TITLE}}` | 번호 없는 H1 주제 제목 |
| `{{SECTION_PURPOSE}}` | Section에서 해결할 문제와 범위 |
| `{{PREREQUISITES}}` | 독자가 미리 알아야 할 내용 |
| `{{EXPECTED_TIME}}` | Notebook의 예상 학습 시간 |
| `{{API_NAME}}` | 설명할 함수 또는 클래스 이름 |
| `{{NEXT_SECTION}}` | 다음 학습 주제 |

복사본에 placeholder가 남아 있는지는 다음 명령으로 확인한다.

```bash
rg '\{\{[A-Z_]+\}\}' docs notebooks
```

## 6. Section 문서 작성 기준

Section 문서에서 반드시 작성할 항목은 다음과 같다.

1. 초보자가 이해할 수 있는 문제 배경과 학습 목표
2. 이 구현이나 구성이 필요한 이유
3. 핵심 개념과 data flow
4. public API 또는 구성·실험 규약
5. 구현을 단계별로 읽는 해설
6. 실행 가능한 사용 예제와 예상 결과
7. 선택한 설계와 고려한 대안
8. 자주 발생하는 문제와 해결 방법
9. 테스트 또는 재현 가능한 검증 절차
10. 핵심 요약과 다음 Section 안내

API를 설명할 때는 signature, parameter, type, shape, dtype, default, return, attribute, side effect와 오류 조건을 빠뜨리지 않는다.
H1에는 Phase, Section, Chapter 번호를 넣지 않고 주제 텍스트만 작성한다.

## 7. Tutorial notebook 작성 기준

Notebook의 각 실습 단위는 아래 순서를 유지한다.

1. Markdown으로 개념, 목적과 예상 결과 설명
2. Code cell에서 계산 또는 API 호출
3. Code cell에서 graph, image 또는 table 출력
4. Markdown으로 실제 결과와 설계 이유 해석
5. Assertion으로 핵심 결과 검증

모든 code cell은 기본 상태에서 실행되어야 한다.
추가 과제는 실행 가능한 예시와 assertion을 제공하거나 Markdown 질문으로 작성하며, 빈 TODO cell이나 의도적인 예외를 남기지 않는다.

## 8. 완료 체크리스트

템플릿을 사용한 문서와 notebook은 아래 기준을 모두 만족해야 한다.

- [ ] H1에 Phase, Section, Chapter 번호가 없다.
- [ ] Markdown 본문은 한국어 서술체이고 code identifier와 comment는 영어이다.
- [ ] 모든 table과 list 앞에 도입 문장이 있다.
- [ ] public API의 입력, 출력과 사용 예가 설명되어 있다.
- [ ] 구현 및 설계 이유가 초보자 관점에서 설명되어 있다.
- [ ] Notebook이 clean kernel에서 독립적으로 실행된다.
- [ ] Notebook에 code, graph 또는 table, 결과 해석 Markdown이 포함되어 있다.
- [ ] Assertion 기반 결과 검증이 포함되어 있다.
- [ ] 복사본에 `{{PLACEHOLDER}}`가 남아 있지 않다.
- [ ] Jupyter Book web build와 PDF export에서 정상적으로 rendering된다.
