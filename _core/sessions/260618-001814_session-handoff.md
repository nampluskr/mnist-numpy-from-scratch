---
tags: [session, handoff]
created: 260618-001814
---

# Phase 7.1 CLI 확장 + CuPy 환경 교체 세션 핸드오프

> 작성일시: 260618-001814
> 세션 목적: Phase 7.1 CLI --model 플래그 추가 및 CuPy 환경 안정화
> 이전 핸드오프: _core/sessions/260617-235801_session-handoff.md

## 1. 세션 핵심 요약

Phase 7.1을 완료하고 CuPy 환경을 안정화했다.
- scripts 4개에 `--model mlp|cnn` 플래그 추가, tests/stage5에 모델 선택 테스트 케이스 추가
- `cupy-cuda11x` → `cupy-cuda12x[ctk]` 교체 (CUDA 12.8 드라이버 호환)
- CuPy 14.x 호환 수정 (`np.asarray()` → `.get()`)
- Linear/Conv2d 초기화 dtype 버그 수정 (`float32 * float64` 업캐스트 방지)
- 최종: stage5+stage6 176 passed

## 2. 완료 항목

| 파일 | 내용 |
|---|---|
| `scripts/train.py` | --model 플래그 추가, build_config에 model 키 추가 |
| `scripts/evaluate.py` | --model 플래그 추가 |
| `scripts/predict.py` | --model 플래그 추가 |
| `scripts/visualize.py` | --model 플래그 추가 |
| `tests/stage5/test_train.py` | make_args model 파라미터, TestTrainModel 클래스 추가 |
| `tests/stage5/test_evaluate.py` | 동일 |
| `tests/stage5/test_predict.py` | 동일 |
| `tests/stage5/test_visualize.py` | 동일 |
| `docs/stage7/phase7.1_cli-extension.md` | Phase 7.1 문서 신규 작성 |
| `requirements.txt` | cupy-cuda11x → cupy-cuda12x[ctk] |
| `src/models/cnn.py` | np.asarray(x_xp) → x_xp.get() (CuPy 14.x 호환) |
| `src/nn/layers.py` | Linear init dtype 버그 수정 |
| `src/nn/conv.py` | Conv2d init dtype 버그 수정 |
| `tests/stage6/test_cnn.py` | to_np() 헬퍼 추가, np.asarray() 교체 |
| `_core/PROJECT-TODO.md` | Phase 7.1 체크박스 9개 완료 처리 |
| `_core/PROJECT-LOG.md` | 이번 세션 작업 이력 추가 |

## 3. 미결 사항

없음.

## 4. 다음 작업 목록

| 우선순위 | Phase | 작업 | 관련 파일 |
|---|---|---|---|
| 1 | Phase 7.2 | 6종 실험 실행 (3 task x 2 model) → outputs/ 저장 | conda run -n numpy_env, MPLBACKEND=Agg |
| 2 | Phase 7.3~7.5 | task별 튜토리얼 문서 (MLP + CNN 각 2개) | docs/stage7/{multiclass,binary,regression}/ |
| 3 | Phase 7.6 | 프레임워크 연계 체크리스트 | docs/stage7/phase7.6_framework-checklist.md |

## 5. 현재 진행 상태

```
Stage 0  레거시 분석 및 계획  [완료]
Stage 1  기본 설정 및 과제 규약  [완료]
Stage 2  MNIST 데이터 로더  [완료]
Stage 3  nn 모듈 및 MLP  [완료]
Stage 4  실행 객체  [완료]
Stage 5  클라이언트 코드  [완료]
Stage 6  CuPy CNN  [완료]
Stage 7  문서화 및 검증
  Phase 7.1 CLI 확장 (--model)  [완료]
  Phase 7.2 실험 실행 및 결과 수집  [미시작]  <- 다음 세션 시작 지점
  Phase 7.3 Multiclass 튜토리얼 (mlp + cnn)
  Phase 7.4 Binary 튜토리얼 (mlp + cnn)
  Phase 7.5 Regression 튜토리얼 (mlp + cnn)
  Phase 7.6 프레임워크 연계 체크리스트
```

## 6. 다음 세션 시작 지시문

`session-start 실행 후 Phase 7.2 실험 실행(6종: 3 task x 2 model)부터 순서대로 진행해 주세요.`

참고 파일:
- 핸드오프: `_core/sessions/260618-001814_session-handoff.md`
- 할일: `_core/PROJECT-TODO.md`
- 스펙: `_core/PROJECT-SPEC.md`
