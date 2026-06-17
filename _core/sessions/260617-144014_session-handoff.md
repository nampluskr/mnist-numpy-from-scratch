# mnist-numpy-from-scratch 세션 핸드오프

> 작성일시: 260617-144014
> 세션 목적: Phase 2.2 MnistDataset 구현 및 테스트 완료
> 이전 핸드오프: 260617-143421_session-handoff.md

## 1. 세션 핵심 요약

Phase 2.2를 TDD 순서로 완료했다.
`test_dataset.py` 28개 테스트를 먼저 작성하여 ImportError로 실패 확인 후, `MnistDataset` 클래스를 `src/data/mnist.py`에 추가하여 전체 통과시켰다.
`project-todo.md` Phase 2.2 체크박스를 완료 처리했다.

## 2. 사용자 요청 및 의도

| 요청 내용 | 배경 목적 |
|---|---|
| Phase 2.2만 진행 | Stage 2 순차 진행 — Phase 2.1 완료 상태에서 MnistDataset 단독 구현 |

## 3. 확정된 결정사항

| 항목 | 확정 내용 | 비고 |
|---|---|---|
| `MnistDataset` 위치 | `src/data/mnist.py` 하단에 추가 | `load_mnist` 위에 두지 않음 |
| task 변환 위임 | `task.py`의 `transform_targets` 재사용 | 중복 로직 없음 |
| `task_spec` 보관 | `__init__`에서 `get_task_spec(task)` 결과를 `self.task_spec`에 저장 | |
| `__getitem__` 반환값 | `(self.images[idx], self.targets[idx])` — 단일 샘플 tuple | |
| 테스트 synthetic 데이터 | n=20, labels=0..9 반복, `tmp_path` fixture | 실제 MNIST 의존 최소화 |

## 4. 미결 사항

미결 사항 없음.

## 5. 다음 작업 목록

| 우선순위 | 작업 | 관련 파일 |
|---|---|---|
| 1 | `DataLoader` 클래스 구현 | `src/data/dataloader.py` |
| 2 | `tests/stage2/test_dataloader.py` 작성 및 실행 | `tests/stage2/test_dataloader.py` |
| 3 | `docs/stage2/phase2.3_dataloader.md` 작성 | `docs/stage2/phase2.3_dataloader.md` |

## 6. 다음 세션 시작 지시문

아래는 이전 세션의 컨텍스트입니다.
이 내용을 기반으로 Phase 2.3 `DataLoader` 구현부터 진행해 주세요.

작업 전 확인 파일:
- 프로젝트 명세: `_core/docs/project-spec.md` §6.2, §6.6
- 진행 현황: `_core/docs/project-todo.md` Phase 2.3
- 핸드오프: `_core/sessions/260617-144014_session-handoff.md`

`DataLoader` 구현 규약:
- 위치: `src/data/dataloader.py` (신규 파일)
- 입력: `dataset`, `batch_size: int`, `shuffle: bool = False`
- `__iter__`: indices 배열로 슬라이싱하여 `(images_batch, targets_batch)` yield
- `__len__`: `ceil(len(dataset) / batch_size)` 반환
- `__len__`과 `__getitem__`을 구현한 Dataset이면 모두 수용 (MNIST 전용 아님)

테스트 실행 환경: `conda run -n numpy_env pytest tests/stage2/ -v`
