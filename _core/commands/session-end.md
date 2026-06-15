---
tags: [project, commands]
created: 2026-06-15
updated: 2026-06-15
---

# 커스텀 명령어: `session-end` / `세션 종료`

`session-end 실행` 또는 `@session-end.md 실행`으로 호출한다.
세션 종료 전 완료 Task 확인, `project-todo.md` 업데이트, `project-log.md` 갱신, 세션 핸드오프 문서 작성을 순서대로 수행한다.

## 1. 실행 절차

### Step 1. 작업 결과 확인

1. `git status` 로 이번 세션에서 변경된 파일 목록을 확인한다.
2. 완료된 Task 를 사용자에게 확인한다.

### Step 2. project-todo.md 업데이트

`_core/docs/project-todo.md` 에서 사용자가 확인한 완료 항목을 `[ ]` → `[x]` 로 변경한다.
완료되지 않은 항목은 그대로 유지한다.

### Step 3. project-log.md 갱신

`_core/docs/project-log.md` 에 이번 세션에서 완료된 항목을 아래 형식으로 추가한다.

```markdown
## YYMMDD {세션 제목}

**완료 항목**
- {완료 항목1}
- {완료 항목2}

**산출물**

| 파일/산출물 | 내용 |
|---|---|
| `...` | ... |

**결정사항**

| 항목 | 결정 내용 |
|---|---|
| ... | ... |
```

산출물 또는 결정사항이 없는 경우 해당 항목은 생략한다.

### Step 4. session-handoff 문서 작성

`session-handoff` 절차를 실행하여 `_core/sessions/` 에 핸드오프 문서를 저장한다.

파일 저장 전에 반드시 `date +%y%m%d-%H%M%S` 를 실행하여 실제 현재 시각을 확인하고 그 값을 파일명에 사용한다.
플레이스홀더(`HHMMSS`, `000000` 등)를 그대로 쓰는 것을 금지한다.

### Step 5. 종료 브리핑 출력

```
== 세션 종료 브리핑 ==

[ 완료된 작업 ]
- [x] Task 1
- [x] Task 2

[ 미완료 / 미결 ]
- [ ] Task 3
- 미결 사항: ...          ← 없으면 이 항목 생략

[ 저장된 파일 ]
- project-todo.md 업데이트 완료
- project-log.md 갱신 완료
- 핸드오프: _core/sessions/{파일명}

[ 다음 세션 시작 지시문 ]
"session-start 실행 후 {다음 Task} 를 이어서 진행해 주세요."
```
