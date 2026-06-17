# mnist-numpy-from-scratch 세션 핸드오프

> 작성일시: 260617-135750
> 세션 목적: PROJECT-TODO.md 경로 수정 및 Stage 1 Phase 1.2~1.3 구현 완료
> 이전 핸드오프: 260617-134447_session-handoff.md

## 1. 세션 핵심 요약

이전 세션에서 미결로 남긴 `PROJECT-TODO.md` 테스트 경로 수정을 완료하고, Phase 1.2 `task.py`와 Phase 1.3 `utils/` 3개 파일을 TDD로 구현했다.
Stage 1 전체 38개 테스트가 통과하여 Stage 1이 완료되었다.

## 2. 사용자 요청 및 의도

| 요청 내용 | 배경 목적 |
|---|---|
| PROJECT-TODO.md 테스트 경로 수정 | 이전 세션 확정: `tests/stage{N}/` 체계 반영 |
| Phase 1.2 진행 | `src/task.py` TDD 구현 |
| 문서 작성 | `docs/stage1/phase1.2_task.md` |
| Phase 1.3 진행 | `src/utils/` 3개 파일 TDD 구현 |
| 세션 핸드오프 작성 | 다음 세션 인계 |

## 3. 확정된 결정사항

| 항목 | 확정 내용 | 비고 |
|---|---|---|
| 테스트 경로 체계 | `tests/stage{N}/test_*.py` — docs와 동일한 stage 폴더 체계 | Phase 1.2~6.2 전체 반영 완료 |
| `get_task_spec` 반환 키 | `task`, `output_dim`, `target_dtype`, `prediction_mode` | 4개 최소 키 확정 |
| `transform_targets` 변환 | multiclass→one-hot (N,10), binary→홀수1/짝수0 (N,1), regression→label/9.0 (N,1) | 모두 float32 |
| `get_batches` 인터페이스 | `get_batches(*arrays, batch_size, shuffle=False)` — 단일 배열이면 ndarray, 복수이면 tuple | |
| `save_params` / `load_params` | numpy `.npz` 기반, dict 저장·로드 | checkpoints(Phase 4.1)에서 활용 예정 |

## 4. 미결 사항

미결 사항 없음.

## 5. 다음 작업 목록

| 우선순위 | 작업 | 관련 파일 |
|---|---|---|
| 1 | `src/data/mnist.py` 구현 | `src/data/mnist.py` |
| 2 | `tests/stage2/test_mnist.py` 작성 및 실행 | `tests/stage2/test_mnist.py` |
| 3 | `docs/stage2/phase2.1_mnist.md` 작성 | `docs/stage2/phase2.1_mnist.md` |

## 6. 다음 세션 시작 지시문

아래는 이전 세션의 컨텍스트입니다.
이 내용을 기반으로 Phase 2.1 `src/data/mnist.py` 구현부터 진행해 주세요.

작업 전 확인 파일:
- 프로젝트 명세: `_core/PROJECT-SPEC.md` §6.1 (데이터셋 기준), §6.6 (공통 함수명과 입출력 규약)
- 레거시 참조: `_core/legacy/src/`의 3개 파일 (데이터 로딩 부분)
- 진행 현황: `_core/PROJECT-TODO.md` Phase 2.1
- 핸드오프: `_core/sessions/260617-135750_session-handoff.md`

`load_mnist(split)` 구현 규약:
- 입력: `split: str` — `"train"` 또는 `"test"`
- 출력: `(images, labels)` tuple
  - `images`: shape `(N, 28, 28)`, dtype `uint8` (정규화 없는 원본)
  - `labels`: shape `(N,)`, dtype `uint8`
- 데이터 경로: `config.py`의 `get_default_config()["dataset_dir"]` 사용
- 로컬 gz 파일 4개에서 직접 읽으며 다운로드 기능은 구현하지 않음

테스트 실행 환경: `conda run -n numpy_env pytest tests/stage2/test_mnist.py -v`
