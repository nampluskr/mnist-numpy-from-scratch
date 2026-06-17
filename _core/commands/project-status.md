---
tags: [project, commands]
created: 2026-06-15
updated: 2026-06-15
---

# 커스텀 명령어: `project-status` / `프로젝트 상태`

`project-status 실행` 또는 `@project-status.md 실행`으로 호출한다.
`project-todo.md` 진행률, 직전 핸드오프의 미결 사항, `project-log.md` 최근 항목을 출력한다.
어떤 파일도 수정하지 않는 읽기 전용 명령어이다.

## 1. 실행 절차

### Step 1. project-todo.md 진행률 집계

`_core/docs/project-todo.md` 를 읽어 아래 항목을 집계한다.

- 전체 Task 수, 완료(`[x]`) 수, 미완료(`[ ]`) 수
- 미완료 항목이 있는 첫 번째 Phase
- 현재 Phase 의 미완료 Task 목록 (최대 5개 - 초과 시 "외 N개" 표시)

### Step 2. 미결 사항 확인

`_core/sessions/` 에서 파일명 기준 가장 최신 핸드오프 파일을 찾아 "미결 사항" 섹션을 추출한다.
해당 섹션이 없거나 파일이 존재하지 않으면 "없음" 으로 표시한다.

### Step 3. 최근 세션 로그 확인

`_core/docs/project-log.md` 에서 가장 최근 세션 항목 1개를 읽어 출력한다.
파일이 없거나 항목이 없으면 해당 항목을 생략한다.

## 2. 출력 형식

```
== 프로젝트 현황 ==

[ 진행률 ]
전체 Task: {N}개  (완료 {K}개 / 미완료 {M}개)
현재 위치: Stage {N} {Stage명} > Phase {N.N} {Phase명}

남은 작업:
- [ ] Task A
- [ ] Task B
  (외 {N}개)

[ 미결 사항 ]
- ...               ← 없으면 "없음"

[ 최근 세션 ]
{날짜} - {세션 제목}: {완료 항목 요약}
```
