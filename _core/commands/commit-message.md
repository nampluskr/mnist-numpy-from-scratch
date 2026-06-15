---
tags: [project, commands]
created: 2026-06-15
updated: 2026-06-15
---

# 커스텀 명령어: `commit-message` / `커밋 메시지`

`commit-message 실행` 또는 `@commit-message.md 실행`으로 호출한다.
커밋은 사용자가 직접 수행한다. AI는 메시지만 제안한다.

## 1. 실행 절차

1. `git status` 로 변경 파일 목록을 확인한다.
2. `git diff HEAD --stat` 로 변경 규모를 파악한다.
3. 변경 규모가 크면 주요 파일만 `git diff HEAD -- {파일}` 로 선택 확인한다.
4. 커밋 메시지를 제안한다. 변경 성격이 다른 파일이 혼재하면 분리 커밋도 함께 제안한다.

## 2. 메시지 형식

```
{type}({scope}): {subject}
```

| 항목 | 설명 |
|---|---|
| `type` | 변경 유형 (아래 표 참조) |
| `scope` | 변경 범위 (폴더명, 파일명, 기능명 등) |
| `subject` | 변경 내용 요약 (명령형, 50자 이내) |

| type | 사용 시점 |
|---|---|
| `feat` | 새 기능 추가 |
| `fix` | 버그 수정 |
| `docs` | 문서 추가·수정 |
| `chore` | 설정 또는 운영 변경 |
| `refactor` | 코드 리팩토링 |
| `style` | 포맷·공백 등 기능 변경 없는 수정 |
| `test` | 테스트 추가·수정 |

## 3. 제안 형식

### 단일 커밋인 경우

```
{type}({scope}): {subject}
```

### 분리 커밋이 필요한 경우

```
커밋 1: {type}({scope}): {subject}
커밋 2: {type}({scope}): {subject}
```
