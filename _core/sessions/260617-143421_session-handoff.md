# mnist-numpy-from-scratch 세션 핸드오프

> 작성일시: 260617-143421
> 세션 목적: Dataset/Dataloader 설계 확정 및 project-spec·todo 반영
> 이전 핸드오프: 260617-140955_session-handoff.md

## 1. 세션 핵심 요약

Dataset/Dataloader 클래스 도입 방향을 설계 토론으로 확정하고 PROJECT-SPEC.md와 PROJECT-TODO.md에 반영했다.
구현은 진행하지 않았으며, Stage 2를 3개 Phase로 재정의하고 Trainer/Evaluator 인터페이스 변경 사항을 spec에 명시했다.

## 2. 사용자 요청 및 의도

| 요청 내용 | 배경 목적 |
|---|---|
| Dataset/Dataloader 클래스 도입 검토 | PyTorch 시리즈와 동일한 인터페이스 구조 확보 |
| Dataset/Dataloader 분리 여부 결정 | 범용 Dataloader + 데이터셋 전용 XXXDataset 패턴 채택 |
| Task 처리 위치 결정 | Dataset 내부에서 task 변환 담당, task.py는 규약(spec)만 유지 |
| PROJECT-SPEC.md / PROJECT-TODO.md 갱신 | 구현 전 사용자 검토 완료 |

## 3. 확정된 결정사항

| 항목 | 확정 내용 | 비고 |
|---|---|---|
| Dataloader 위치 | `src/data/dataloader.py` — 범용, Dataset 프로토콜만 요구 | `__len__` + `__getitem__` |
| Dataset 패턴 | ` MNISTDataset` — MNIST 전용, 향후 `FashionMnistDataset`, `Cifar10Dataset` 확장 | |
| task 변환 위치 | 각 Dataset 클래스 내부 — `task.py`는 output_dim·loss·metric·prediction_mode 규약만 유지 | `transform_targets`는 Dataset 내부에서 호출 |
| binary 변환 | MNIST: 홀수=1/짝수=0, FashionMNIST: 상의=1/하의=0, CIFAR-10: 동물=1/차량=0 — 데이터셋별로 다름 | |
| `load_mnist` 유지 | Phase 2.1 기존 코드·테스트 그대로 유지, public 함수로 존속 | |
| Stage 2 Phase 재정의 | Phase 2.1 mnist(완료), Phase 2.2  MNISTDataset(신규), Phase 2.3 Dataloader(신규) | |
| Trainer.fit 인터페이스 | `Trainer.fit(train_loader)` — Dataloader 수신 | Phase 4.2에서 구현 |
| Evaluator.evaluate 인터페이스 | `Evaluator.evaluate(test_loader)` — Dataloader 수신 | Phase 4.3에서 구현 |

## 4. 미결 사항

미결 사항 없음.

## 5. 다음 작업 목록

| 우선순위 | 작업 | 관련 파일 |
|---|---|---|
| 1 | ` MNISTDataset` 클래스 구현 (mnist.py에 추가) | `src/data/mnist.py` |
| 2 | `tests/stage2/test_dataset.py` 작성 및 실행 | `tests/stage2/test_dataset.py` |
| 3 | `docs/stage2/phase2.2_dataset.md` 작성 | `docs/stage2/phase2.2_dataset.md` |
| 4 | `Dataloader` 구현 | `src/data/dataloader.py` |
| 5 | `tests/stage2/test_dataloader.py` 작성 및 실행 | `tests/stage2/test_dataloader.py` |
| 6 | `docs/stage2/phase2.3_dataloader.md` 작성 | `docs/stage2/phase2.3_dataloader.md` |

## 6. 다음 세션 시작 지시문

아래는 이전 세션의 컨텍스트입니다.
이 내용을 기반으로 Phase 2.2 ` MNISTDataset` 구현부터 진행해 주세요.

작업 전 확인 파일:
- 프로젝트 명세: `_core/PROJECT-SPEC.md` §5.3, §6.2, §6.6
- 진행 현황: `_core/PROJECT-TODO.md` Phase 2.2, 2.3
- 핸드오프: `_core/sessions/260617-143421_session-handoff.md`

` MNISTDataset` 구현 규약:
- 위치: `src/data/mnist.py` (기존 `load_mnist` 아래에 추가)
- 입력: `split: str`, `task: str`, `dataset_dir=None`
- `__init__`: `load_mnist(split)` 호출 → reshape(-1, 784) + /255 정규화 → `_transform(labels, task)` 호출
- `__len__`: 샘플 수 반환
- `__getitem__(idx)`: `(image, target)` 단일 샘플 tuple 반환
- `_transform(labels, task)` task별 변환:
  - multiclass → one_hot(labels, num_classes=10), shape (N, 10)
  - binary → (labels % 2).reshape(-1, 1), 홀수=1/짝수=0
  - regression → (labels / 9.0).reshape(-1, 1)
- `self.task_spec`: `get_task_spec(task)` 결과 보관

`Dataloader` 구현 규약:
- 위치: `src/data/dataloader.py`
- 입력: `dataset`, `batch_size: int`, `shuffle: bool = False`
- `__iter__`: indices 배열로 슬라이싱하여 `(images_batch, targets_batch)` yield
- `__len__`: `ceil(len(dataset) / batch_size)` 반환

테스트 실행 환경: `conda run -n numpy_env pytest tests/stage2/ -v`
