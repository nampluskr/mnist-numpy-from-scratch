# mnist-numpy-from-scratch 세션 핸드오프

> 작성일시: 260617-140955
> 세션 목적: Phase 2.1 src/data/mnist.py TDD 구현 완료
> 이전 핸드오프: 260617-135750_session-handoff.md

## 1. 세션 핵심 요약

Phase 2.1 `src/data/mnist.py`를 TDD로 구현하고 전체 51개 테스트가 통과하여 Stage 2가 완료되었다.
단위 테스트는 합성 gz 파일 기반으로 실 MNIST 파일에 의존하지 않으며, 실데이터 테스트 4개는 skipif로 보호되어 있다.

## 2. 사용자 요청 및 의도

| 요청 내용 | 배경 목적 |
|---|---|
| Phase 2.1 진행 | `src/data/mnist.py` TDD 구현 |
| 세션 핸드오프 및 커밋·푸시 | 다음 세션 인계 및 원격 동기화 |

## 3. 확정된 결정사항

| 항목 | 확정 내용 | 비고 |
|---|---|---|
| `load_mnist` 시그니처 | `load_mnist(split, dataset_dir=None)` — `dataset_dir` 선택 파라미터 추가 | spec 기본 `load_mnist(split)` 호환 유지, 테스트 주입용 |
| 반환값 | `(images, labels)` — images `(N,28,28)` uint8, labels `(N,)` uint8 원본 배열 | 정규화·변환 없음 |
| 오류 처리 | 잘못된 split → `ValueError`, 파일 없음 → `FileNotFoundError` | |
| 테스트 경로 | `tests/stage2/test_mnist.py` | stage 폴더 체계 유지 |
| 실데이터 테스트 | `@pytest.mark.skipif(not os.path.exists(MNIST_DIR), ...)` 로 보호 | 4개 |

## 4. 미결 사항

미결 사항 없음.

## 5. 다음 작업 목록

| 우선순위 | 작업 | 관련 파일 |
|---|---|---|
| 1 | `src/models/mlp.py` 구현 | `src/models/mlp.py` |
| 2 | `tests/stage3/test_mlp.py` 작성 및 실행 | `tests/stage3/test_mlp.py` |
| 3 | `docs/stage3/phase3.1_mlp.md` 작성 | `docs/stage3/phase3.1_mlp.md` |

## 6. 다음 세션 시작 지시문

아래는 이전 세션의 컨텍스트입니다.
이 내용을 기반으로 Phase 3.1 `src/models/mlp.py` 구현부터 진행해 주세요.

작업 전 확인 파일:
- 프로젝트 명세: `_core/docs/project-spec.md` §6.2 (src 구조), §6.5 (레거시 매핑), §6.6 (공통 함수명과 입출력 규약)
- 레거시 참조: `_core/legacy/src/`의 3개 파일 (모델 구조, forward, backward 부분)
- 진행 현황: `_core/docs/project-todo.md` Phase 3.1, Phase 3.2
- 핸드오프: `_core/sessions/260617-140955_session-handoff.md`

`MLP` 구현 규약:
- 구조: `784 -> 256 -> 128 -> output_dim` (hidden activation `sigmoid`, output activation은 task에 따라 다름)
- `MLP.forward(x)` — 입력 `(N, 784)` float32, 출력 `(N, output_dim)` float32
- `Trainer.fit`에서 호출하는 backward·update는 Phase 3.2에서 layers/activations/losses로 분리 예정
- `get_task_spec(task)`의 output_dim, output activation 정보를 참조하여 MLP 생성

테스트 실행 환경: `conda run -n numpy_env pytest tests/stage3/test_mlp.py -v`
