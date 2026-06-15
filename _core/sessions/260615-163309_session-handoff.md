# mnist-numpy-from-scratch 세션 핸드오프

> 작성일시: 260615-163309
> 세션 목적: 워크스페이스 초기화 — legacy 자료를 현재 _core/docs/ 구조에 반영

## 1. 세션 핵심 요약

이 세션에서는 이전 프로젝트(legacy)에서 작성된 설계 문서와 코드를 현재 워크스페이스 구조에 반영하는 초기화 작업을 수행했다. 코드 구현은 진행하지 않았고, 운영 문서 정비만 완료한 상태이다.

## 2. 사용자 요청 및 의도

| 요청 내용 | 배경 목적 |
|---|---|
| 현재 워크스페이스 분석 | 프로젝트 상태 파악 및 작업 시작 준비 |
| `_core/legacy/` 내용을 현재 워크스페이스에 반영 | legacy 폴더는 수정 없이 참조 전용으로 유지하고, 실제 운영 문서에 내용 이식 |
| `project-todo.md` 초기화 | 새로 시작하는 워크스페이스이므로 모든 Task를 미완료 상태로 리셋 |
| `project-log.md` 반영 및 세션 핸드오프 실행 | 세션 이력 기록 후 다음 세션 인계 |

## 3. 확정된 결정사항

| 항목 | 확정 내용 | 비고 |
|---|---|---|
| 프로젝트명 | Deep Learning from Scratch with Numpy/Cupy | |
| 슬러그 | mnist-numpy-from-scratch | |
| `_core/legacy/` 역할 | 수정 금지, 참조 전용 | src 3개 파일 + refs 5개 파일 |
| `project-spec.md` | `_core/legacy/refs/PROJECT.md` 전체 내용 반영 완료 | §6 확정 구조 포함 |
| `project-todo.md` | 전체 Task 미완료(`[ ]`)로 초기화 완료 | Stage 0 포함 |
| CLAUDE.md / project-guide.md | 플레이스홀더 채움 완료 | 날짜, 프로젝트명, 목적 |
| 데이터 경로 | `/mnt/d/datasets/mnist/` | gz 파일 4개 로컬 보관 |
| 기준 구현 역할 | numpy → pytorch → tensorflow → jax 시리즈의 첫 번째 | 모듈명·함수명 통일 필요 |

## 4. 미결 사항

| # | 항목 | 현재 상태 | 결정 필요 내용 |
|---|---|---|---|
| 1 | `_core/docs/subject-guide.md` 내용 | 미작성 | 주제 분류 체계 필요 시 작성 |
| 2 | `requirements.txt` 보완 | `numpy`만 등록됨 | `pytest`, `matplotlib` 등 추가 필요 |

## 5. 다음 작업 목록

다음 세션은 Stage 1 구현을 시작한다. TDD 원칙에 따라 테스트를 먼저 작성하고 구현한다.

| 우선순위 | 작업 | 관련 파일 |
|---|---|---|
| 1 | `src/config.py` 코드 작성 | `src/config.py` |
| 2 | `tests/test_config.py` 테스트 작성 및 실행 | `tests/test_config.py` |
| 3 | `src/task.py` 코드 작성 | `src/task.py` |
| 4 | `tests/test_task.py` 테스트 작성 및 실행 | `tests/test_task.py` |
| 5 | `src/utils/batching.py`, `random.py`, `io.py` 코드 작성 | `src/utils/` |

## 6. 다음 세션 시작 지시문

아래는 이전 세션의 컨텍스트입니다.
이 내용을 기반으로 Stage 1 Phase 1.1 `src/config.py` 구현부터 진행해 주세요.

작업 전 확인 파일:
- 프로젝트 명세: `_core/docs/project-spec.md` §6.6 (공통 함수명과 입력·출력 규약)
- 진행 현황: `_core/docs/project-todo.md` Stage 1
- 레거시 참조: `_core/legacy/src/mnist-multiclass-mlp.py`
- 핸드오프: `_core/sessions/260615-163309_session-handoff.md`

`get_default_config()` 함수가 반환해야 할 기본값은 다음과 같다.

- `dataset_dir`: `/mnt/d/datasets/mnist`
- `seed`: `42`
- `batch_size`: `64`
- `num_epochs`: `10`
- `task`: `"multiclass"`
- `split`: `"train"`
