---
tags: [stage4, core, checkpoints]
created: 2026-06-17
updated: 2026-06-17
---

# Phase 4.2 checkpoint 구현

## 1. 역할

`src/core/checkpoints.py`는 `model.params` 목록을 `.npz` 파일로 저장하고 복원한다.
저장 시 `param_0`, `param_1`, ... 형태의 인덱스 키를 사용하고, 복원 시 in-place로 배열 값을 덮어쓴다.

## 2. 구현

### 2.1. save(model, path)

`model.params` 목록을 `param_0`, `param_1`, ... 키로 `.npz` 파일에 저장한다.

```python
save(model, "outputs/model")   # → outputs/model.npz 생성
```

`numpy.savez`는 경로에 `.npz`가 없으면 자동으로 추가한다.

### 2.2. load(model, path)

`.npz` 파일에서 파라미터를 읽어 `model.params`를 in-place로 복원한다.

```python
load(model, "outputs/model")      # .npz 자동 보완
load(model, "outputs/model.npz")  # 확장자 포함도 허용
```

경로에 `.npz`가 없으면 자동으로 보완하여 `numpy.load`에 전달한다.
`param[...] = data[f"param_{i}"]` 방식으로 배열 값만 교체하므로 옵티마이저가 보유한 `params` 참조가 그대로 유지된다.

### 2.3. 인터페이스

```python
from src.core.checkpoints import save, load
from src.models.mlp import MLP

model = MLP(task="multiclass", seed=0)

save(model, "outputs/checkpoint")
# → outputs/checkpoint.npz

load(model, "outputs/checkpoint")
# model.params 배열이 저장된 값으로 in-place 복원
```

## 3. 테스트

테스트 파일: `tests/stage4/test_checkpoints.py`

| 클래스 | 항목 수 | 주요 검증 내용 |
|---|---|---|
| `TestSave` | 3 | .npz 파일 생성 확인, 키 존재 확인, 저장 값과 원본 일치 |
| `TestLoad` | 4 | 값 복원 정확성, in-place 복원(참조 불변), .npz 확장자 포함/미포함 경로 모두 허용 |

실행 명령:

```bash
conda run -n numpy_env pytest tests/stage4/test_checkpoints.py -v
```

## 4. 설계 결정

- `param[...] = ...` 방식으로 in-place 복원한다. `model.params[i] = ...`로 재할당하면 옵티마이저(`SGD`, `Adam`)가 생성 시 저장한 참조가 무효화된다.
- `src/utils/io.py`의 `save_params`/`load_params`는 dict 기반이므로 직접 사용하지 않는다. 체크포인트는 list 기반 `model.params`를 다루므로 인덱스 키 방식을 자체 구현한다.
- `.npz` 확장자 처리는 `load`에서만 필요하다. `np.savez`는 자동으로 확장자를 추가하지만, `np.load`는 정확한 경로를 요구하기 때문이다.
