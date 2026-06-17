# Stage 3 src/nn 재설계 및 MLP 재작성 세션 핸드오프

> 작성일시: 260617-172449
> 세션 목적: Stage 3 구조를 PyTorch 방식에 맞게 전면 재설계하고 구현 완료
> 이전 핸드오프: 없음

## 1. 세션 핵심 요약

`src/models/` 하위에 놓였던 `layers.py`, `activations.py`, `losses.py`를 `src/nn/` 패키지로 분리하여 `torch.nn` 대응 구조로 확정했다. `MLP`는 `Sequential(Linear, Sigmoid, ...)` 조합으로 재작성되었고 `forward()`는 raw logit을 반환한다. 손실 함수(`*_grad`)가 activation을 내부 처리하는 PyTorch 방식으로 통일되었다. Stage 3 테스트 69개 전체 통과.

## 2. 사용자 요청 및 의도

| 요청 내용 | 배경 목적 |
|---|---|
| 비어있지 않은 폴더의 .gitkeep 삭제 | 불필요한 placeholder 정리 |
| src 폴더 구조 재검토 | numpy 전용 구성요소와 공통 구조를 분리하여 후속 프레임워크와 일관성 확보 |
| Stage 3 재정의 및 MLP 재작성 | PyTorch 방식을 레퍼런스로, nn/ 모듈을 조립하는 방식으로 구현 |
| losses.py 인터페이스 통일 | logit 입력 방식(PyTorch CrossEntropyLoss 방식)으로 통일 |

## 3. 확정된 결정사항

| 항목 | 확정 내용 | 비고 |
|---|---|---|
| `src/nn/` 패키지 | `layers.py`, `activations.py`, `losses.py` — numpy-only, `torch.nn` 대응 | `models/`에서 분리 |
| `src/models/` | `mlp.py`, `cnn.py` 모델 정의만 — 모든 프레임워크 공통 | layers/activations 제거 |
| `MLP.forward()` 출력 | raw logit 반환 (`(N, output_dim)`) | activation은 losses.py에서 처리 |
| losses.py 인터페이스 | logit 입력, activation 내부 처리 | PyTorch `CrossEntropyLoss` 방식 |
| gradient 함수 | `cross_entropy_grad`, `binary_cross_entropy_grad`, `mse_grad` — losses.py에 포함 | trainer에서 backward 입력으로 사용 |
| `MLP.params` / `MLP.grads` | list 타입 (dict → list 변경) | `params[0]` = 첫 Linear의 `w` |
| Stage 3 Phase 구성 | 4개 Phase (3.1 activations, 3.2 layers, 3.3 losses, 3.4 mlp) | 기존 2개에서 확장 |

## 4. 미결 사항

| # | 항목 | 현재 상태 | 결정 필요 내용 |
|---|---|---|---|
| 1 | Stage 3 docs | 미작성 | phase3.1~3.4 문서 4개 작성 필요 |
| 2 | Stage 4 설계 검토 | 미시작 | trainer/evaluator가 logit 기반 losses.py를 사용하는 방식 설계 필요 |

## 5. 다음 작업 목록

| 우선순위 | 작업 | 관련 파일 |
|---|---|---|
| 1 | Stage 3 문서 4개 작성 | `docs/stage3/phase3.1_activations.md` ~ `phase3.4_mlp.md` |
| 2 | Phase 4.1 옵티마이저 구현 | `src/core/optimizers.py`, `tests/stage4/test_optimizers.py` |
| 3 | Phase 4.2 체크포인트 구현 | `src/core/checkpoints.py`, `tests/stage4/test_checkpoints.py` |

## 6. 현재 파일 구조 (Stage 3 완료 시점)

```
src/
├── nn/              ← 신규 (numpy-only)
│   ├── __init__.py
│   ├── activations.py   # sigmoid, softmax, relu, identity
│   ├── layers.py        # Module, Linear, Sigmoid, ReLU, Sequential
│   └── losses.py        # *_grad 포함, logit 입력
├── models/
│   └── mlp.py       ← 재작성 (Sequential 기반, logit 출력)
tests/stage3/
├── test_activations.py  ← 신규 (17개)
├── test_layers.py       ← 신규 (18개)
├── test_losses.py       ← 신규 (21개)
└── test_mlp.py          ← 재작성 (13개)
```

## 7. 다음 세션 시작 지시문

아래는 이전 세션의 컨텍스트입니다.
Stage 3 전체 코드·테스트가 완료되었고 문서만 미작성 상태입니다.
`session-start 실행 후 Stage 3 docs 작성 또는 Phase 4.1 옵티마이저 구현을 이어서 진행해 주세요.`

참고 파일:
- 핸드오프: `_core/sessions/260617-172449_session-handoff.md`
- 할일: `_core/PROJECT-TODO.md`
- 스펙: `_core/PROJECT-SPEC.md`
