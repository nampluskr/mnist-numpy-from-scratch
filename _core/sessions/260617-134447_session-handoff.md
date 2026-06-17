# mnist-numpy-from-scratch 세션 핸드오프

> 작성일시: 260617-134447
> 세션 목적: Phase 1.1 config 구현 및 프로젝트 환경 확정
> 이전 핸드오프: 260615-163309_session-handoff.md

## 1. 세션 핵심 요약

Phase 1.1을 완료하고, 프로젝트 구조와 실행 환경에 관한 여러 결정사항을 확정했다.
코드 구현보다 환경 설계 결정이 많았던 세션이다.

## 2. 사용자 요청 및 의도

| 요청 내용 | 배경 목적 |
|---|---|
| Phase 1.1 task별 승인 후 진행 | 단계별 검토하며 구현 |
| tests 폴더를 docs와 같이 stage 폴더 구조로 변경 | 문서-테스트 대응 관계 명시 |
| stage{0N} → stage{N} 폴더명 변경 | 불필요한 0 패딩 제거 |
| pyproject.toml 삭제 | 루트 파일 최소화 |
| tests/__init__.py 삭제 및 지침 반영 | pytest 표준 구조 적용 |
| numpy_env 생성 (Python 3.11) | 시리즈 공통 Python 버전 기준 확립 |
| jupyterlab / ipykernel 설치 | 노트북 기반 중간·최종 결과 정리 |

## 3. 확정된 결정사항

| 항목 | 확정 내용 | 비고 |
|---|---|---|
| Python 버전 | 3.11 | pytorch/tensorflow/jax/cupy 시리즈 공통 기준 |
| 실행 환경 | `numpy_env` (conda) | numpy 2.4.6, pytest 9.1.0, matplotlib 3.11.0, jupyterlab 4.5.8 |
| Jupyter 커널 | `Python (numpy_env)` | `~/.local/share/jupyter/kernels/numpy_env` 등록 완료 |
| tests 폴더 구조 | `tests/stage{N}/` — docs와 동일한 stage 폴더 체계 | stage01만 먼저 생성, 이후 단계에서 순차 추가 |
| `__init__.py` | `tests/` 및 하위 폴더 전체 생성 금지 | `coding-rules.md` §8에 규칙 추가 |
| pytest 경로 설정 | `tests/conftest.py`의 `sys.path.insert`로 처리 | `pyproject.toml` 삭제, 루트 파일 최소화 |
| 폴더명 규칙 | `docs/stage{N}/`, `tests/stage{N}/` (0 패딩 없음) | stage0~stage7 |
| docs 태그 | frontmatter `tags: [stage{N}, ...]` | 0 패딩 없음으로 통일 |

## 4. 미결 사항

| # | 항목 | 현재 상태 | 결정 필요 내용 |
|---|---|---|---|
| 1 | `project-todo.md`의 Phase 1.2 이후 테스트 경로 | `tests/test_task.py` 등 stage 폴더 미반영 | `tests/stage1/test_task.py` 등으로 일괄 수정 필요 |
| 2 | `_core/docs/subject-guide.md` 내용 | 미작성 | 주제 분류 체계 필요 시 작성 |

## 5. 다음 작업 목록

| 우선순위 | 작업 | 관련 파일 |
|---|---|---|
| 1 | `project-todo.md` Phase 1.2~1.3 테스트 경로 수정 | `_core/docs/project-todo.md` |
| 2 | `src/task.py` 구현 | `src/task.py` |
| 3 | `tests/stage1/test_task.py` 작성 및 실행 | `tests/stage1/test_task.py` |
| 4 | `docs/stage1/phase1.2_task.md` 작성 | `docs/stage1/phase1.2_task.md` |
| 5 | Phase 1.3 utils 구현 시작 | `src/utils/` |

## 6. 다음 세션 시작 지시문

아래는 이전 세션의 컨텍스트입니다.
이 내용을 기반으로 Phase 1.2 `src/task.py` 구현부터 진행해 주세요.

작업 전 확인 파일:
- 프로젝트 명세: `_core/docs/project-spec.md` §6.5 (task별 차이), §6.6 (공통 함수명과 입출력 규약)
- 레거시 참조: `_core/legacy/src/`의 3개 파일 (multiclass, binary, regression)
- 진행 현황: `_core/docs/project-todo.md` Phase 1.2
- 핸드오프: `_core/sessions/260617-134447_session-handoff.md`

`get_task_spec(task)` 반환 dict 최소 포함 키:

- `task`: `"multiclass"` / `"binary"` / `"regression"`
- `output_dim`: `10` / `1` / `1`
- `target_dtype`: `float32`
- `prediction_mode`: `"argmax"` / `"threshold"` / `"round_clip"`

`transform_targets(labels, task)` 변환 규약:

- `"multiclass"`: one-hot, shape `(N, 10)`, `float32`
- `"binary"`: 홀수=1/짝수=0, shape `(N, 1)`, `float32`
- `"regression"`: `label / 9.0`, shape `(N, 1)`, `float32`

테스트 실행 환경: `conda run -n numpy_env pytest tests/stage1/test_task.py -v`
