---
tags: [project, rules]
created: 2026-06-20
updated: 2026-06-20
---

# template-rules.md

이 프로젝트에서 `docs/`와 `notebooks/`를 작성할 때 따르는 템플릿 사용 규칙이다.
템플릿 원본은 `_core/templates/`에 보관하며, 복사본을 만들어 각 Phase 문서와 노트북을 작성한다.

## 1. 템플릿 목록

제공하는 템플릿과 용도는 다음과 같다.

| 파일 | 용도 |
|---|---|
| `docs-template.md` | 초보자 대상 책 수준 설명 문서. API 구현, 개념, 설계 이유를 포함한다 |
| `notebooks-template.ipynb` | 독립 실행형 실습 노트북. 코드, 그래프, assertion으로 개념을 검증한다 |

## 2. 복사와 파일명

Section 문서는 대상 Stage 폴더로 복사한다.

```bash
cp _core/templates/docs-template.md docs/stageN/phaseN.M_topic.md
```

Tutorial 노트북은 학습 순서가 드러나는 이름으로 복사한다.

```bash
cp _core/templates/notebooks-template.ipynb notebooks/stageN/stageN-M_topic.ipynb
```

파일명 규칙은 다음과 같다.

| 대상 | 형식 | 예시 |
|---|---|---|
| Section 문서 | `phase{N}.{M}_{topic}.md` | `phase3.1_activations.md` |
| Tutorial 노트북 | `stage{N}-{M}_{topic}.ipynb` | `stage3-1_activations.ipynb` |

## 3. Placeholder 교체

템플릿을 복사한 뒤 아래 placeholder를 실제 값으로 교체한다.

| Placeholder | 의미 | 예시 |
|---|---|---|
| `{{STAGE_TAG}}` | Stage 태그 | `stage3` |
| `{{TOPIC_TAG}}` | 주제 태그 | `activations` |
| `{{CREATED_DATE}}` | 최초 작성일 | `2026-06-20` |
| `{{UPDATED_DATE}}` | 최종 수정일 | `2026-06-20` |
| `{{TOPIC_TITLE}}` | H1 주제 제목 (번호 없음) | `Phase 3.1 activation 구현` |
| `{{SECTION_PURPOSE}}` | 이 문서/노트북이 다루는 범위 | `4종 활성화 함수` |
| `{{CONCEPT_1}}`, `{{CONCEPT_2}}` | 개념 섹션 소제목 | `sigmoid`, `softmax` |
| `{{FORMULA}}` | LaTeX 수식 | `\sigma(x) = \frac{1}{1+e^{-x}}` |
| `{{API_NAME}}` | 설명할 함수 또는 클래스 이름 | `sigmoid` |
| `{{MODULE_PATH}}` | import 경로 | `src.nn.activations` |
| `{{ENV_NAME}}` | conda 환경명 | `numpy_py311` |
| `{{STAGE_DIR}}` | 테스트 폴더명 | `stage3` |
| `{{TOPIC}}` | 테스트 파일 주제명 | `activations` |
| `{{NEXT_PHASE_LINK}}` | 다음 Phase Obsidian 링크 | `phase3.2_losses` |
| `{{PRACTICE_TITLE_1}}` | 노트북 첫 번째 실습 제목 | `4종 함수 그래프` |
| `{{PLOT_TITLE}}` | 그래프 제목 | `Activation Functions` |

복사본에 placeholder가 남아 있지 않은지 다음 명령으로 확인한다.

```bash
grep -r '{{' docs/ notebooks/
```

## 4. docs-template.md 작성 기준

Section 문서는 6개 섹션으로 구성하며 순서를 바꾸지 않는다.

| 섹션 | 제목 | 핵심 내용 |
|---|---|---|
| 1 | 개요 | 책임 서술 2~4문장 + **목표** 블록 |
| 2 | 개념 | 수식·용어·원리를 코드 없이 초보자 수준으로 상세 설명 |
| 3 | 구현 | API 표 + 코드 해설. H3 소섹션은 Phase 성격에 맞게 자유 구성 |
| 4 | 사용법 | 최소 실행 예제 + 프로젝트 통합 예제 |
| 5 | 테스트 | 실행 명령 + 테스트 클래스·항목 수 표 |
| 6 | 요약 | 2~4문장 마무리 + 다음 Phase 링크 |

섹션별 세부 규칙은 다음과 같다.

- **개요**: 서브섹션 없이 본문 뒤 바로 `**목표**` bold 제목과 불릿 목록을 작성한다.
- **개념**: 수식이 있는 Phase는 $...$ 인라인 또는 $$...$$ 블록으로 수식을 먼저 제시하고 한국어 해설을 붙인다. 수식이 없는 Phase는 동작 원리를 서술체로 충분히 설명한다.
- **구현**: 비직관적 구현(수치 안정성, shape 변환, backend 선택 등)은 이유를 상세히 해설한다.
- **사용법**: import부터 출력까지 그대로 실행할 수 있는 코드를 제공한다.
- **테스트**: `conda run -n {env} pytest ...` 형식의 실행 가능한 명령을 작성한다.

## 5. notebooks-template.ipynb 작성 기준

노트북은 다음 구조를 따른다.

| 위치 | 내용 |
|---|---|
| Intro cell | H1 제목 + 설명 1~2문장 + **목표** 불릿 (섹션 번호 없음) |
| 0. 환경 설정 | 고정 3셀 구조 (아래 참고) |
| 1. 개요 | 필요성과 핵심 개념 요약. 표 또는 수식 포함 |
| 2~N. 주제별 실습 | Phase 성격에 따라 자유 구성 |
| 요약 | 정리 표 + 핵심 설계 원칙 불릿 |

환경 설정 3셀 구조는 순서와 역할이 고정되어 있다.

| 셀 | 역할 | jupyter book 태그 |
|---|---|---|
| 셀 1 | `os`, `sys` import 및 `sys.path` 설정 | `remove-cell` |
| 셀 2 | 외부 라이브러리 import (`numpy`, `matplotlib` 등) | 없음 |
| 셀 3 | `src/` 사용자 구현 import | 없음 |

각 실습 단위는 다음 순서를 유지한다.

1. Markdown: 이 실습의 목적과 예상 결과 설명
2. Code: 계산 또는 API 호출
3. Code: 그래프 또는 출력
4. Code: assertion으로 핵심 결과 검증

## 6. 공통 규칙

- 노트북은 이전 노트북의 변수, checkpoint, kernel state에 의존하지 않는다. 단독 실행이 가능해야 한다.
- H1 제목에 Phase, Stage, Chapter 번호를 넣지 않는다. 주제 텍스트만 작성한다.
- 모든 표와 목록 앞에 도입 문장을 1문장 이상 작성한다.
- 문서 본문은 한국어 서술체(`~이다`, `~한다`)로 작성한다. 코드 식별자와 주석은 영어로 작성한다.
- `docs-rules.md`, `python-rules.md` 규칙을 함께 준수한다.
- 노트북의 공통 강의 구조는 학습 목표, 선수 지식, 환경 설정, 개념 설명, 단계별 실습, 결과 검증, 정리, 추가 과제 순으로 구성한다.
- 노트북 내부에 conda 환경명, kernel, dataset 경로, GPU 요구사항을 명시한다.
- interactive output이 PDF에서 표현되지 않을 경우 static image 또는 table로 대체한다.

## 7. 완료 체크리스트

템플릿으로 작성한 문서와 노트북은 아래 항목을 모두 만족해야 한다.

- [ ] placeholder가 남아 있지 않다.
- [ ] H1에 번호가 없다.
- [ ] 개요 섹션에 **목표** 블록이 있다.
- [ ] 개념 섹션에 수식 또는 동작 원리 서술이 있다.
- [ ] 사용법 섹션에 실행 가능한 코드 예제가 있다.
- [ ] 테스트 섹션에 `conda run -n ...` 명령이 있다.
- [ ] 노트북이 clean kernel에서 단독으로 실행된다.
- [ ] 노트북 환경 설정 셀 1에 `remove-cell` 태그가 있다.
- [ ] 노트북 각 실습 단위에 assertion이 있다.
- [ ] 노트북 내부에 conda 환경명, dataset 경로, GPU 요구사항이 명시되어 있다.
