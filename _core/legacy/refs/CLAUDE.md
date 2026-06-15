# 워크스페이스 가이드

이 파일은 AI CLI 가 워크스페이스에서 작업할 때 따르는 운영 지침이다.
프로젝트 내용 및 진행 사항은 `_project/PROJECT.md` 와 `_project/PROJECT-TODO.md` 를 참조한다.
완료된 Task 의 상세 진행 이력은 `_project/PROJECT-HISTORY.md` 를 참조한다.

다른 AI CLI 사용 시 이 파일을 복사하여 해당 CLI 의 지침 파일명으로 rename 한다.
- Claude: CLAUDE.md
- Codex: AGENTS.md
- Gemini: GEMINI.md
- RooCode: ROO.md

> 생성일시: 260605-195326
> 수정일시: 260607-190719
> 주제: Development Environment and Tools

**목차**

1. [문서 작성 규칙](#1-문서-작성-규칙)
2. [AI CLI 응답 스타일](#2-ai-cli-응답-스타일)
3. [프로젝트 구조](#3-프로젝트-구조)
4. [폴더 규칙](#4-폴더-규칙)
5. [네이밍 규칙](#5-네이밍-규칙)
6. [코딩 컨벤션](#6-코딩-컨벤션)
7. [작업 방식](#7-작업-방식)
8. [Jupyter Book v2](#8-jupyter-book-v2)
9. [제한 사항](#9-제한-사항)
10. [커스텀 명령어](#10-커스텀-명령어)

## 1. 문서 작성 규칙

문서 작성은 이 워크스페이스에서 수행하는 모든 작업의 핵심이다. 아래 규칙은 절대 원칙으로, 이 워크스페이스에서 생성하거나 수정하는 모든 문서에 예외 없이 적용한다.

### 1.1. 문체 규칙

본문에서 사용하는 문체와 언어 형식에 관한 규칙이다.

- 서술체(`~이다`, `~한다`) 사용 — 경어체(`~입니다`, `~합니다`)는 문서 본문에 사용 불가
- 기술 문서 또는 학술 문서 형식 — 구어체, 비격식 표현, 감탄사 사용 불가

### 1.2. 형식 규칙

문서의 시각적 형식과 강조 표현에 관한 규칙이다.

- 이모지 사용 절대 금지
- 경어체, 불필요한 볼드·이탤릭 강조 남용 금지

### 1.3. 항목 작성 규칙

목록 및 항목 작성 시 따르는 규칙이다.

- 항목은 반드시 개조식으로 작성하며, 핵심 키워드를 중심으로 의미와 맥락이 명확히 전달되도록 기술하고 단순 명사 나열이나 지나치게 압축된 표현 금지
- 항목명이 필요한 경우에만 `항목명: 개조식 설명` 형태로 사용
- 테이블·숫자/불릿 목록 바로 앞에 도입 문장 1문장 이상 삽입 — 나열 항목의 맥락과 존재 이유를 먼저 제시

### 1.4. 세부 규칙 참조

헤더 계층, 수식, 이미지, 테이블, Jupyter Book(MyST) 등 세부 형식 규칙은 별도 파일에서 관리한다.

- 문서 작성 세부 규칙: `_project/rules/markdown-style.md`
- 주제 분류 체계: `_project/rules/subjects.md`

## 2. AI CLI 응답 스타일

AI CLI 가 사용자에게 응답할 때 다음 스타일 규칙을 따른다.
이 규칙은 §1 문서 작성 규칙과 별개로 적용되며, AI CLI 의 대화 응답에만 해당한다.

### 2.1. 응답 문체

대화 응답에서 사용하는 문체 규칙이다.

- `~합니다`, `~입니다` 경어체 응답 — 서술체(`~이다`, `~한다`)는 문서 본문 전용

### 2.2. 응답 형식

응답의 형식과 구조에 관한 규칙이다.

- 응답 텍스트에 이모지 사용 불가
- 불필요한 사과, 서두, 재확인 없이 요청 즉시 처리

### 2.3. 날짜·시각 처리

문서의 날짜·시각 값을 기입할 때 반드시 따르는 규칙이다.

- 날짜·시각이 필요한 경우 반드시 `date +"%y%m%d-%H%M%S"` 명령을 실행하여 실제 값을 확인한 후 기입 — 임의 값 사용 금지
- 파일을 수정할 때마다 해당 파일의 `수정일시` 를 현재 시각으로 갱신

## 3. 프로젝트 구조

프로젝트 루트의 기본 디렉토리 구조와 각 파일의 역할을 정의한다. 각 폴더의 상세 규칙은 §4 폴더 규칙에서 기술한다.

### 3.1. 디렉토리 구조

프로젝트 루트의 기본 구조는 다음과 같다.

```
project-root/
├── README.md                   # 프로젝트 외부 소개
├── CLAUDE.md                   # AI CLI 워크스페이스 운영 지침 (이 파일)
├── AGENTS.md                   # Codex 용 AI CLI 워크스페이스 운영 지침
│
├── _project/                   # 프로젝트 운영 관리
│   ├── PROJECT.md              # 프로젝트 내부 명세
│   ├── PROJECT-TODO.md         # Stage-Phase-Task 진행 관리
│   ├── PROJECT-HISTORY.md      # 완료 Task 상세 기록
│   ├── commands/               # 커스텀 명령어 프롬프트
│   ├── docs/                   # 운영 문서
│   ├── rules/                  # 공통 규칙
│   └── sessions/               # 세션 핸드오프 문서
├── _assets/                    # 바이너리 읽기 전용 입력 보관소
│
├── legacy/                     # 참고·보완용 레거시 문서 및 코드
├── scripts/                    # 실행 스크립트 및 Python 파일
├── outputs/                    # 프로젝트 중간/최종 결과 저장
└── docs/                       # 최종 산출물 (Jupyter Book v2)
    ├── myst.yml
    └── contents/
        └── chap01/
            └── images/
```

### 3.2. 핵심 파일 설명

프로젝트의 핵심 파일 역할은 다음과 같다.

| 파일 | 역할 |
|------|------|
| `README.md` | 외부 공개용 프로젝트 소개 문서 |
| `CLAUDE.md` | Claude Code 용 AI CLI 운영 지침 (이 파일) |
| `AGENTS.md` | Codex 용 AI CLI 운영 지침 — CLAUDE.md 와 동일 내용 유지 |
| `_project/PROJECT.md` | 프로젝트의 목적, 배경, 범위, 제약 사항, 진행 단계를 정의하는 내부 명세 |
| `_project/PROJECT-TODO.md` | Stage-Phase-Task 단위의 진행 현황 관리 문서 |

두 파일은 항상 동일한 내용을 유지한다. 공통 지침을 외부 파일에 두고 참조만 하는 방식도 검토했으나, AI CLI 는 CLAUDE.md 안에서 참조하는 외부 파일을 자동 로드하지 않으므로 각 파일에 직접 유지하는 현행 방식을 채택한다.

## 4. 폴더 규칙

### 4.1. 접두사 규칙

폴더명 접두사로 내부 운영 폴더와 프로젝트 산출물 폴더를 구분한다. 접두사 규칙은 다음과 같다.

| 접두사 | 의미 | 예시 |
|--------|------|------|
| `_` | 내부 운영/보조 폴더 (프로젝트 산출물 아님) | `_project/`, `_assets/` |
| 없음 | 프로젝트 산출물 또는 표준 폴더 | `docs/`, `scripts/`, `outputs/` |

### 4.2. 폴더 역할

각 폴더의 역할과 사용 기준은 다음과 같다. `_` 접두사 폴더는 운영 보조용이며 프로젝트 산출물로 취급하지 않는다.

| 폴더 | 역할 | 비고 |
|------|------|------|
| `_project/` | 프로젝트 운영 문서 및 세션 기록 | |
| `_project/commands/` | 커스텀 명령어 정의 파일 | `{파일명} 실행` 또는 `@{파일명.md} 실행` 형식으로 호출 |
| `_project/rules/` | 공통 규칙 파일 | 문서 작성 스타일 등 |
| `_assets/` | 읽기 전용 입력 보관소 | 참조만 가능, 수정/가공 금지 |
| `legacy/` | 이전 버전 문서 및 리팩토링 대상 레거시 코드 | 수정·가공 허용 (`_assets/` 와 달리 편집 가능) |
| `scripts/` | 실행 스크립트 및 Python 파일 | 문서 생성 스크립트 포함 |
| `outputs/` | 프로젝트 산출물 | 하위 구조는 프로젝트별 추가 |
| `docs/` | Jupyter Book v2 배포 가능한 최종 산출물 | |

### 4.3. 데이터 흐름

입력 자료에서 최종 문서까지의 처리 흐름은 다음과 같다. `_assets/` 는 원본 보관소이므로 이 흐름에서 수정 없이 읽기만 한다.

```
_assets/          →      scripts/       →      outputs/
(읽기 전용 입력)           (처리/실행)           (중간/최종 산출물)
                                                    ↓
                                             docs/contents/
                                             (최종 문서)
```

## 5. 네이밍 규칙

파일과 폴더 이름은 일관된 규칙을 따른다. 세부 규칙은 `_project/rules/markdown-style.md §3` 을 참조한다.

- 폴더명: 소문자, 하이픈 구분 (예: `chap01`, `session-handoff`)
- 파일명: 단어 구분에 하이픈(`-`), 접두사·접미사 연결에 언더스코어(`_`)
- 루트 레벨 MD 파일: 대문자 (예: `README.md`, `CLAUDE.md`)
- 프로젝트 관리 MD 파일: 대문자 — 위치에 관계없이 적용 (예: `_project/PROJECT.md`, `_project/PROJECT-TODO.md`)
- Python 파일: 소문자, 언더스코어 구분 (예: `train_model.py`)
- 챕터 폴더: `chap01`, `chap02`, ... (두 자리 숫자, 0 패딩)
- 세션 핸드오프 파일: `YYMMDD-HHMMSS_session-handoff.md`

## 6. 코딩 컨벤션

이 워크스페이스의 모든 코드에 다음 컨벤션을 준수한다. 세부 규칙은 언어별 파일을 참조한다. Python: `_project/rules/python-style.md`, Jupyter Notebook: `_project/rules/notebook-style.md`.

- 언어: Python 우선
- 경로 표기: `os.path` 방식 사용 (`pathlib.Path` 사용 금지)
- 코드 내 주석: 영어로 작성
- 들여쓰기: 스페이스 4칸
- 인코딩: UTF-8

## 7. 작업 방식

세션 시작부터 종료까지 다음 순서와 원칙을 따른다. 각 단계는 프로젝트 문서의 일관성과 추적 가능성을 유지하기 위한 것이다.

### 7.1. 세션 시작

작업을 시작하기 전에 수행하는 절차이다.

- 작업 시작 전 `_project/PROJECT.md`, `_project/PROJECT-TODO.md` 읽기 — 목적·범위·진행 상황 파악

### 7.2. 작업 중

작업 수행 중 지켜야 하는 원칙이다.

- `_assets/` 파일: 참조 전용 — 수정·가공·삭제 금지
- 생성된 중간·최종 산출물을 `outputs/` 에 저장
- 문서 포함 이미지는 `docs/contents/chapXX/images/` 에 저장

### 7.3. 세션 종료

작업 완료 후 수행하는 정리 절차이다.

- 작업 완료 후 `_project/PROJECT-TODO.md` 해당 Task 체크박스 완료 상태로 업데이트
- 필요한 경우 `session-end 실행` 으로 세션 종료 절차 수행

## 8. Jupyter Book v2

최종 문서는 Jupyter Book v2 형식으로 `docs/` 에 관리한다. 문서 형식 세부 규칙은 `_project/rules/markdown-style.md §9` 를 참조한다.

- 설정 파일: `docs/myst.yml`
- 문서 내용: `docs/contents/chapXX/`
- 이미지: `docs/contents/chapXX/images/`

### 8.1. 빌드 규칙

빌드 결과물 관리와 실행 주체에 관한 규칙이다.

- 빌드 결과물: `docs/_build/` — `.gitignore` 등록, AI CLI 직접 생성 금지
- 빌드 실행: 사용자가 직접 수행

## 9. 제한 사항

워크스페이스 무결성 유지와 사용자 작업 보호를 위해 다음 사항을 반드시 준수한다.

### 9.1. 파일 보호

원본 자료와 핵심 파일을 보호하기 위한 제한 사항이다.

- `_assets/` 내 파일 수정 및 삭제 금지
- 루트 필수 파일 (`README.md`, `CLAUDE.md`, `AGENTS.md`) 및 프로젝트 관리 파일 (`_project/PROJECT.md`, `_project/PROJECT-TODO.md`) 임의 삭제 금지

### 9.2. 빌드 및 git 제한

빌드 결과물 생성과 git 작업에 관한 제한 사항이다.

- `docs/_build/` 직접 생성 금지 — 빌드는 사용자가 직접 실행
- git commit, push 수행 금지 — 사용자가 VSCode UI 로 직접 수행

## 10. 커스텀 명령어

커스텀 명령어는 `_project/commands/` 에 저장된 Markdown 파일로 정의된다.

### 10.1. 호출 방법

커스텀 명령어를 호출하는 세 가지 방법은 다음과 같다.

- 방법 1: `{파일명 (확장자 제외)} 실행`
- 방법 2: `@{파일명.md} 실행`
- 방법 3: `{한국어 명령} 실행` — §10.2 테이블의 한국어 명령 열 참조

호출 예시:

```
project-init 실행
@project-init.md 실행
프로젝트 초기화 실행
```

### 10.2. 등록 명령어 목록

현재 등록된 커스텀 명령어 목록은 다음과 같다.

| 명령어 | 한국어 명령 | 파일 | 수행 내용 |
|--------|------------|------|-----------|
| `session-start` | `세션 시작` | `commands/session-start.md` | 세션 시작 브리핑 — `_project/PROJECT.md` / `_project/PROJECT-TODO.md` / 직전 핸드오프 읽기 후 현황 요약 출력 |
| `session-end` | `세션 종료` | `commands/session-end.md` | 세션 종료 절차 — PROJECT-TODO.md 업데이트, PROJECT-HISTORY.md 갱신, session-handoff 실행 |
| `session-handoff` | `세션 핸드오프` | `commands/session-handoff.md` | 세션 핸드오프 문서 작성 및 `_project/sessions/` 에 저장 |
| `project-init` | `프로젝트 초기화` | `commands/project-init.md` | 대화형으로 프로젝트 정보 수집 후 `_project/PROJECT.md` → `_project/PROJECT-TODO.md` → `README.md` 작성/갱신 |
| `project-update` | `프로젝트 업데이트` | `commands/project-update.md` | 자유 형식으로 수정 내용 입력 → `_project/PROJECT.md` / `_project/PROJECT-TODO.md` / `README.md` 선택적 갱신 |
| `project-status` | `프로젝트 상태` | `commands/project-status.md` | PROJECT-TODO.md 진행률, 미결 사항, PROJECT-HISTORY.md 최근 Task 요약 출력 (읽기 전용) |
| `commit-message` | `커밋 메시지` | `commands/commit-message.md` | git 변경사항 분석 후 커밋 메시지 제안 |

### 10.3. 명령어 추가

새로운 커스텀 명령어를 추가하는 방법이다.

- 정의 파일 위치: `_project/commands/`
- 명령어 추가: `_project/commands/` 에 Markdown 파일 추가 시 즉시 사용 가능

> 다른 AI CLI 사용 시에도 동일하게 적용된다.
> `_project/commands/` 폴더의 파일명과 내용을 참조하여 동일한 방식으로 명령어를 처리한다.
