# mnist-numpy-from-scratch 세션 핸드오프

> 작성일시: 260617-144907
> 세션 목적: Phase 2.3 DataLoader 구현 및 session-end.md 커밋/푸시 절차 추가
> 이전 핸드오프: 260617-144014_session-handoff.md

## 1. 세션 핵심 요약

Phase 2.3을 TDD 순서로 완료했다.
`test_dataloader.py` 13개 테스트를 먼저 작성하여 ImportError로 실패 확인 후, `DataLoader` 클래스를 `src/data/dataloader.py`에 구현하여 전체 통과시켰다.
Stage 2 전체(54개 테스트)를 재실행하여 이상 없음을 확인했다.
추가로 `session-end.md`에 Step 6(사용자 승인 후 커밋/푸시) 절차를 추가했다.

## 2. 사용자 요청 및 의도

| 요청 내용 | 배경 목적 |
|---|---|
| Phase 2.3만 진행 | Stage 2 순차 진행 — Phase 2.2 완료 상태에서 DataLoader 단독 구현 |
| session-end.md에 커밋/푸시 절차 추가 | 세션 종료 시 문서 작업 후 승인을 받아 커밋·푸시까지 일괄 처리 |

## 3. 확정된 결정사항

| 항목 | 확정 내용 | 비고 |
|---|---|---|
| `DataLoader` 위치 | `src/data/dataloader.py` (신규 파일) | |
| index 방식 | `np.random.permutation` / `np.arange`로 index 배열 생성 후 슬라이싱 | Dataset 원본 배열 직접 셔플 안 함 |
| 배치 조립 | `np.stack([dataset[i][0] for i in batch_idx])` | MNIST 전용 아님, 범용 |
| 마지막 배치 | drop 없음, 그대로 yield | 호출부에서 필요 시 처리 |
| 테스트 데이터 | `ToyDataset(n=20, feature_dim=4)` synthetic — MNIST 의존 없음 | |

## 4. 미결 사항

미결 사항 없음.

## 5. 다음 작업 목록

| 우선순위 | 작업 | 관련 파일 |
|---|---|---|
| 1 | `MLP` 클래스 구현 (forward, backward, parameter update) | `src/models/mlp.py` |
| 2 | `tests/stage3/test_mlp.py` 작성 및 실행 | `tests/stage3/test_mlp.py` |
| 3 | `docs/stage3/phase3.1_mlp.md` 작성 | `docs/stage3/phase3.1_mlp.md` |

## 6. 다음 세션 시작 지시문

아래는 이전 세션의 컨텍스트입니다.
이 내용을 기반으로 Phase 3.1 `MLP` 구현부터 진행해 주세요.

작업 전 확인 파일:
- 프로젝트 명세: `_core/docs/project-spec.md` §6.2, §6.5, §6.6
- 진행 현황: `_core/docs/project-todo.md` Phase 3.1
- 핸드오프: `_core/sessions/260617-144907_session-handoff.md`

`MLP` 구현 규약:
- 위치: `src/models/mlp.py` (신규 파일)
- 구조: `784 → 256 → 128 → output_dim`, hidden activation `sigmoid`
- output activation은 task별 (`softmax` / `sigmoid` / `identity`) — `get_task_spec(task)`에서 읽음
- `forward(x)` → `(N, output_dim)` prediction 배열 반환
- manual backward + SGD update 포함
- NumPy 전용 (CuPy는 Phase 6.1)

테스트 실행 환경: `conda run -n numpy_env pytest tests/stage3/ -v`
