---
tags: [project, commands]
created: 2026-06-15
updated: 2026-06-15
---

# 커스텀 명령어: `project-init` / `프로젝트 초기화`

`project-init 실행` 또는 `@project-init.md 실행`으로 호출한다.
프로젝트 시작 시 또는 진행 중 대화형으로 프로젝트 정보를 수집하여
`_core/docs/project-spec.md` → `_core/docs/project-todo.md` → `README.md` 순으로 작성하거나 갱신한다.

## 1. 실행 모드 판단

명령어 실행 시 먼저 `_core/docs/project-spec.md` 존재 여부를 확인한다.

- **신규 모드**: 파일이 없거나 플레이스홀더(`{{...}}`) 상태인 경우 → 전체 항목을 새로 수집하여 작성
- **업데이트 모드**: 실제 내용이 있는 경우 → 현재 값을 보여주며 변경할 항목만 수집하여 갱신

## 2. 질문 절차

아래 순서대로 사용자에게 질문한다.
업데이트 모드에서는 각 질문 앞에 `[현재: ...]` 형태로 기존 값을 먼저 보여준다.
사용자가 빈 입력(Enter)을 하면 기존 값을 유지한다.

### Step 1. 기본 정보

```
1. 프로젝트명을 입력하세요.
2. 프로젝트를 한 줄로 설명해 주세요.
3. Subject code를 선택하세요.
   MATH, PHYS, NA, ML, CV, NLP, CS, DEV, MISC
   (복수 선택 가능, 쉼표로 구분)
```

### Step 2. 프로젝트 내용

```
4. 이 프로젝트의 목적은 무엇인가요?
5. 배경이나 문제 상황을 설명해 주세요.
6. 수행할 작업의 범위를 항목별로 입력해 주세요. (여러 줄 가능, 완료 시 빈 줄 입력)
7. 제약 사항이 있으면 입력해 주세요. (없으면 Enter)
```

### Step 3. 진행 단계

수집한 내용을 바탕으로 AI가 먼저 Stage - Phase 수준의 초안을 작성하여 제시한다.

```
수집한 내용을 바탕으로 진행 단계 초안을 작성했습니다.

Stage 1 {Stage명}
  Phase 1.1 {Phase명}
  Phase 1.2 {Phase명}
Stage 2 {Stage명}
  Phase 2.1 {Phase명}
  ...

수정할 사항이 있으면 말씀해 주세요. (없으면 "확인" 입력)
```

사용자 피드백을 반영하여 수정하고, "확인" 입력 시 다음 단계로 넘어간다.
진행 단계 확정 후 AI가 각 Phase에 맞는 Task를 직접 작성하여 `project-todo.md` 초안을 생성한다.

## 3. 확인 및 수정

모든 항목 수집 후 요약을 보여주고 확인을 요청한다.

```
수집한 내용을 요약합니다.
프로젝트명: ...
설명: ...
Subject code: ...
목적: ...
배경: ...
범위: ...
제약 사항: ...
진행 단계:
  Stage 1 ...
    Phase 1.1 ...
      Task: ...

이 내용으로 파일을 작성합니다. 수정할 항목이 있으면 항목 번호 또는 이름을 말씀해 주세요.
확인이면 "확인" 또는 "진행"을 입력하세요.
```

## 4. 파일 작성 규칙

### project-spec.md

`_core/rules/docs-rules.md` 규칙을 준수하여 작성한다.

```markdown
---
tags: [project, spec]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---

# {프로젝트명}

{한 줄 설명}

## 1. 목적

{수집한 목적}

## 2. 배경

{수집한 배경}

## 3. 범위

{수집한 범위 항목 불릿}

## 4. 제약 사항

{수집한 제약 사항 불릿}

## 5. 진행 단계

### Stage 1. {Stage명}

- Phase 1.1 {Phase명}
- Phase 1.2 {Phase명}

### Stage 2. {Stage명}

- Phase 2.1 {Phase명}
```

### project-todo.md

```markdown
---
tags: [project, todo]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---

# 작업 목록

Stage - Phase - Task 단위로 체크박스를 관리한다.

## Stage 1. {Stage명}

### Phase 1.1. {Phase명}

- [ ] Task 1
- [ ] Task 2

### Phase 1.2. {Phase명}

- [ ] Task 1

## Stage 2. {Stage명}

### Phase 2.1. {Phase명}

- [ ] Task 1
```

**업데이트 모드 주의사항**:
- 완료된 체크박스(`- [x]`)는 절대 변경하지 않는다.
- 기존 Phase에 Task를 추가할 때는 미완료 항목 뒤에 이어 붙인다.
- 새로운 Stage/Phase는 기존 내용 뒤에 추가한다.

### README.md

신규 모드에서는 프로젝트명, 설명, 태그, 개요 섹션을 반영한다.
업데이트 모드에서는 프로젝트명, 설명, 태그만 갱신한다.

## 5. 완료 보고

```
완료되었습니다.

작성된 파일:
- _core/docs/project-spec.md: 목적 / 배경 / 범위 / 제약 / {N}개 Stage
- _core/docs/project-todo.md: {N}개 Stage, {M}개 Phase, {K}개 Task
- README.md: 프로젝트명 / 설명 / 태그 갱신
```
