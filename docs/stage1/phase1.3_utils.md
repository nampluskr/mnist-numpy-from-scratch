---
tags: [stage1, utils]
created: 2026-06-17
updated: 2026-06-17
---

# Phase 1.3 유틸리티 구현: 배치 처리, 난수 시드, 파일 I/O

## 1. 역할

`src/utils/`는 batching, 난수 시드, 파일 I/O를 담당하는 공통 보조 모듈을 제공한다.
후속 PyTorch, TensorFlow, JAX 프로젝트에서도 동일한 함수명과 사용법을 유지한다.

## 2. 구현

### 2.1. batching.py - get_batches()

`get_batches(*arrays, batch_size, shuffle=False)`는 하나 이상의 배열을 받아 mini-batch를 순서대로 yield하는 generator이다.

인자와 동작 규약은 다음과 같다.

| 인자 | 설명 |
|---|---|
| `*arrays` | 같은 길이의 numpy 배열 1개 이상 |
| `batch_size` | mini-batch 크기 |
| `shuffle` | `True`이면 매 호출마다 배열 전체를 무작위로 섞는다 |

반환 규약은 다음과 같다.

- 배열이 1개이면 단일 `np.ndarray`를 yield한다.
- 배열이 2개 이상이면 같은 인덱스로 슬라이싱한 `tuple`을 yield한다.
- 마지막 배치는 `batch_size`보다 작을 수 있다.
- `shuffle=True`이면 모든 배열에 동일한 인덱스를 적용하므로 쌍 관계가 유지된다.

### 2.2. random.py - set_seed()

`set_seed(seed)`는 numpy 난수 시드를 설정한다.
실험 재현성을 확보하기 위해 학습 시작 전에 호출한다.

### 2.3. io.py - save_params() / load_params()

`save_params(params, path)`는 문자열 키와 numpy 배열 값으로 구성된 dict를 `.npz` 파일로 저장한다.
`load_params(path)`는 `.npz` 파일을 읽어 동일한 구조의 dict로 반환한다.

저장 형식으로 `.npz`를 선택한 이유는 여러 파라미터를 단일 파일에 묶어 키 기반으로 접근할 수 있기 때문이다.

## 3. 테스트

`tests/stage1/`의 테스트 파일별 검증 항목은 다음과 같다.

`test_batching.py` 검증 항목은 다음과 같다.

- generator가 반복 가능한지
- 단일 배열 입력 시 `np.ndarray`를 yield하는지
- 복수 배열 입력 시 `tuple`을 yield하는지
- batch size가 `batch_size` 이하인지
- 마지막 배치 크기가 나머지인지
- 전체 샘플 수가 유지되는지
- `shuffle=False`일 때 원래 순서가 유지되는지
- `shuffle=True`일 때 복수 배열 간 인덱스가 일치하는지

`test_random.py` 검증 항목은 다음과 같다.

- 같은 시드로 두 번 호출하면 동일한 배열이 생성되는지
- 다른 시드로 호출하면 다른 배열이 생성되는지

`test_io.py` 검증 항목은 다음과 같다.

- 저장 후 파일이 존재하는지
- 로드 반환 타입이 `dict`인지
- 저장·로드 후 키 집합이 일치하는지
- 저장·로드 후 배열 값이 일치하는지

테스트 실행 명령은 다음과 같다.

```bash
conda run -n numpy_env pytest tests/stage1/ -q
```

```text
[Expected output]
38 passed in 0.47s
```
