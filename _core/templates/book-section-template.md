---
tags: [docs, "{{STAGE_TAG}}", "{{TOPIC_TAG}}"]
created: "{{CREATED_DATE}}"
updated: "{{UPDATED_DATE}}"
---

# {{TOPIC_TITLE}}

## 1. 개요

{{SECTION_PURPOSE}}
이 Section에서 다루는 구현 또는 구성의 책임과 전체 학습 흐름을 초보자가 이해할 수 있는 문장으로 설명한다.

## 2. 학습 목표와 선수 지식

이 Section을 마치면 할 수 있어야 하는 일은 다음과 같다.

1. 핵심 개념을 자신의 말로 설명한다.
2. public API의 입력과 출력을 설명한다.
3. 최소 사용 예제를 직접 실행한다.
4. 구현의 주요 설계 결정을 설명한다.
5. 테스트 또는 재현 절차로 결과를 검증한다.

이 Section을 읽기 전에 필요한 선수 지식은 다음과 같다.

- {{PREREQUISITES}}

## 3. 왜 필요한가

이 구현 또는 구성이 해결하는 문제를 먼저 설명한다.
단순히 코드를 나열하지 않고, 해당 기능이 없을 때 발생하는 중복, 오류 또는 학습상의 어려움을 구체적인 예로 제시한다.

## 4. 핵심 개념과 동작 흐름

입력부터 출력까지의 data flow와 각 구성요소의 책임을 순서대로 설명한다.
필요한 경우 shape, dtype, backend와 상태 변화를 함께 표시한다.

```text
input -> preprocessing -> {{API_NAME}} -> output -> validation
```

핵심 용어는 다음 표와 같이 정리한다.

| 용어 | 의미 | 이 프로젝트에서의 역할 |
|---|---|---|
| 개념 1 | 용어의 일반적인 의미 | 현재 구현에서 담당하는 책임 |
| 개념 2 | 용어의 일반적인 의미 | 다른 구성요소와의 연결 관계 |

## 5. API reference 또는 구성·실험 규약

구현된 함수나 클래스를 다루는 경우 이 Section에 public API를 작성한다.
환경 구성, 설계 또는 실험 결과 문서라면 제목을 유지하고 API 대신 명령, 설정 key, 입력 자료, 산출물과 재현 조건을 같은 형식으로 설명한다.

### 5.1. {{API_NAME}}

API signature는 다음과 같다.

```python
{{API_NAME}}(...)
```

이 API는 입력을 받아 수행하는 책임과 반환하는 결과를 한 문단으로 설명한다.

Parameter는 다음과 같다.

| 이름 | Type | Shape·dtype | Default | 설명 |
|---|---|---|---|---|
| `parameter` | `type` | `(N, D)`, `float32` | 없음 | 입력값의 의미와 허용 범위 |

반환값은 다음과 같다.

| Type | Shape·dtype | 설명 |
|---|---|---|
| `return_type` | `(N, K)`, `float32` | 반환값의 의미와 후속 사용 위치 |

Class인 경우 주요 attribute와 method는 다음과 같다.

| 이름 | Type | 설명 |
|---|---|---|
| `attribute` | `type` | 저장하는 상태와 변경 시점 |
| `method()` | method | 수행하는 동작과 상태 변화 |

호출 전제, side effect와 오류 조건은 다음과 같다.

- 입력 shape와 dtype 조건
- 내부 상태를 변경하는지 여부
- file 또는 device 접근 여부
- 잘못된 입력에서 발생하는 오류

최소 사용 예는 다음과 같다.

```python
result = {{API_NAME}}(...)
print(result)
```

예상 결과는 다음과 같다.

```text
예상되는 type, shape 또는 출력값
```

## 6. 구현 해설

구현은 독자가 source code를 따라갈 수 있도록 실행 순서대로 설명한다.

### 6.1. 입력 준비

입력을 검증하고 내부 계산에 필요한 형태로 변환하는 이유를 설명한다.

### 6.2. 핵심 계산

핵심 알고리즘과 수식을 코드의 변수명에 연결하여 설명한다.
수치 안정성, memory layout 또는 backend 차이가 있다면 선택 이유를 포함한다.

### 6.3. 출력과 상태 관리

반환값을 만드는 과정과 저장되는 상태가 이후 forward, backward 또는 실행 단계에서 어떻게 사용되는지 설명한다.

## 7. 사용 예제와 예상 결과

### 7.1. 최소 예제

가장 적은 입력으로 API의 기본 동작을 확인한다.

```python
# Replace with a minimal executable example.
```

실행 후 확인할 결과와 그 의미를 설명한다.

### 7.2. 프로젝트 통합 예제

다른 module, Dataset, model 또는 core 객체와 연결하는 실제 사용 흐름을 보여준다.

```python
# Replace with a project integration example.
```

예제의 입력, 중간 결과와 최종 출력이 설계 의도와 일치하는 이유를 설명한다.

## 8. 설계 결정과 대안

현재 구현에서 선택한 설계와 이유는 다음과 같다.

| 결정 | 선택한 방식 | 대안 | 선택 이유 |
|---|---|---|---|
| 설계 항목 | 현재 방식 | 고려한 다른 방식 | 학습 목적, 단순성, 성능 또는 호환성 관점의 이유 |

초보자가 흔히 떠올릴 수 있는 다른 구현이 왜 현재 프로젝트의 목적과 맞지 않는지도 함께 설명한다.

## 9. 주의사항과 문제 해결

자주 발생하는 문제와 확인 방법은 다음과 같다.

| 증상 | 원인 | 확인 방법 | 해결 방법 |
|---|---|---|---|
| 오류 또는 잘못된 결과 | 가능한 원인 | shape, dtype, path 또는 state 확인 | 수정 방법 |

## 10. 테스트와 검증

이 Section의 구현을 검증하는 테스트 대상과 성공 기준을 설명한다.

```bash
conda run -n numpy_py311 pytest tests/stageN/test_target.py -v
```

테스트가 확인하는 동작은 다음과 같다.

1. 정상 입력의 type, shape와 값
2. 경계 조건 또는 잘못된 입력
3. 상태 변경과 side effect
4. 다른 구성요소와의 통합 동작

## 11. 요약과 다음 Section

이 Section의 핵심 내용을 다음 문장으로 요약한다.

1. 해결한 문제
2. public API와 data flow
3. 중요한 설계 결정
4. 검증 방법

다음 Section에서는 {{NEXT_SECTION}}을 다룬다.
