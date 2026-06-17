---
tags: [project, commands]
created: 2026-06-15
updated: 2026-06-18
---

# 커스텀 명령어: `project-update` / `프로젝트 업데이트`

`project-update 실행` 또는 `@project-update.md 실행`으로 호출한다.
자유 형식으로 입력된 수정 내용을 파악하여
`_core/PROJECT-SPEC.md` / `_core/PROJECT-TODO.md` / `README.md` 중 해당 파일만 선택적으로 갱신한다.

`project-init` 이 순서대로 정보를 수집하는 초기 설정 명령어라면,
`project-update` 는 진행 중 언제든 자유롭게 수정 사항을 반영하는 명령어이다.

## 1. 입력 없이 호출한 경우

수정 내용 없이 명령어만 호출하면 아래 안내를 출력한다.

```
어떤 내용을 수정할까요?
PROJECT-SPEC.md, PROJECT-TODO.md, README.md 중 변경할 내용을 자유롭게 말씀해 주세요.

예시:
- "목적을 바꿨어. 이제 딥러닝 모델 비교가 메인이야"
- "Phase 2.1에 Task 하나 추가해줘 - 데이터 전처리 스크립트 작성"
- "제약사항에 'GPU 없는 환경' 추가하고 Stage 3 삭제해줘"
```

## 2. 실행 절차

### Step 1. 의도 파악

입력을 분석하여 수정 대상 파일과 섹션을 식별한다.

| 수정 대상 | 대상 파일 |
|---|---|
| 목적 / 배경 / 범위 / 제약 사항 / 진행 단계 (Stage·Phase) | `_core/PROJECT-SPEC.md` |
| Task 추가·삭제 / Stage·Phase 신규 추가 | `_core/PROJECT-TODO.md` |
| 프로젝트명 / 한 줄 설명 / Subject 태그 | `README.md` |

하나의 입력으로 여러 파일에 걸친 수정도 처리한다.
Stage·Phase 신규 추가는 `PROJECT-SPEC.md` 단계 섹션과 `PROJECT-TODO.md` 를 함께 갱신한다.

### Step 2. 현재 값 확인 및 변경 내용 제시

수정 전 해당 섹션의 현재 값을 읽어 표시하고, 변경 내용을 제시한다.

```
[ 현재 목적 ]
{기존 내용}

아래 내용으로 변경합니다.
{변경 내용}
```

- **삭제·덮어쓰기·구조 변경** (섹션 내용 교체, Stage 삭제 등): 변경 전 확인 요청
- **단순 추가** (Task 추가, 범위 항목 추가 등): 확인 없이 바로 반영

### Step 3. 파일 갱신

확인된 수정 사항을 해당 파일에 적용한다.

- 완료된 체크박스(`- [x]`)는 절대 변경하지 않는다.
- frontmatter `updated` 필드를 오늘 날짜로 자동 갱신한다.
- `_core/rules/docs-rules.md` 규칙을 준수하여 작성한다.

**PROJECT-TODO.md 특별 규칙**:
- Task 추가 시 Phase 가 지정되면 해당 Phase 끝에 삽입한다.
- Phase 가 지정되지 않으면 미완료 항목이 있는 첫 번째 Phase 끝에 삽입한다.
- Task 완료 체크(`- [x]`)는 `project-update` 로 수행하지 않는다.

### Step 4. 완료 보고

```
완료되었습니다.

변경된 파일:
- _core/PROJECT-SPEC.md: {변경 섹션} 수정
- _core/PROJECT-TODO.md: Phase X.X 에 Task 추가
```

변경된 파일이 없는 경우 그 이유를 설명한다.
