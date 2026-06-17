# Stage 6~7 계획 재검토 세션 핸드오프

> 작성일시: 260617-235022
> 세션 목적: Stage 7 단계 재검토 — Phase 재정의, Stage 6 CuPy 환경 추가
> 이전 핸드오프: _core/sessions/260617-232624_session-handoff.md

## 1. 세션 핵심 요약

이번 세션에서는 코드 구현 없이 Stage 6~7 계획을 재정의했다. Stage 6에 Phase 6.0(CuPy 환경 구성)을 신규 추가하고, Stage 7을 기존 3 Phase에서 6 Phase로 세분화했다. PROJECT-SPEC.md §5.7~5.8과 PROJECT-TODO.md Stage 6~7 섹션을 갱신했다.

## 2. 완료 항목

| 파일 | 내용 |
|---|---|
| `_core/PROJECT-SPEC.md` | §5.7 Stage 6에 Phase 6.0 추가, §5.8 Stage 7을 6 Phase로 전면 교체 |
| `_core/PROJECT-TODO.md` | Stage 6에 Phase 6.0 Task 삽입, Stage 7 전면 교체 (3 Phase → 6 Phase) |
| `_core/sessions/260617-235022_session-handoff.md` | 이 문서 |

## 3. 미결 사항

없음 (계획 문서 갱신만 수행).

## 4. 다음 작업 목록

다음 세션에서 아래 순서로 진행한다.

| 우선순위 | Phase | 작업 | 관련 파일 |
|---|---|---|---|
| 1 | Phase 6.0 | CuPy 환경 구성 | requirements.txt, docs/stage6/phase6.0_cupy-setup.md |
| 2 | Phase 7.1 | scripts --model 플래그 추가 + stage5 테스트 업데이트 | scripts/\*.py, tests/stage5/\*.py, docs/stage7/phase7.1_cli-extension.md |
| 3 | Phase 7.2 | 6종 실험 실행 (3 task × 2 model) → outputs/ 저장 | conda run -n numpy_env, MPLBACKEND=Agg |
| 4 | Phase 7.3~7.5 | task별 튜토리얼 문서 (MLP + CNN 각 2개) | docs/stage7/{multiclass,binary,regression}/ |
| 5 | Phase 7.6 | 프레임워크 연계 체크리스트 | docs/stage7/phase7.6_framework-checklist.md |

## 5. 현재 진행 상태

```
Stage 0  레거시 분석 및 계획  [완료]
Stage 1  기본 설정 및 과제 규약  [완료]
Stage 2  MNIST 데이터 로더  [완료]
Stage 3  nn 모듈 및 MLP  [완료]
Stage 4  실행 객체  [완료]
Stage 5  클라이언트 코드  [완료]
Stage 6  CuPy CNN
  Phase 6.0 CuPy 환경 구성  [미시작] ← 다음 세션 시작 지점
  Phase 6.1 CNN 모델            [완료]
  Phase 6.2 통합 검증          [완료]
Stage 7  문서화 및 검증  [미시작]
  Phase 7.1 CLI 확장 (--model)
  Phase 7.2 실험 실행 및 결과 수집
  Phase 7.3 Multiclass 튜토리얼 (mlp + cnn)
  Phase 7.4 Binary 튜토리얼 (mlp + cnn)
  Phase 7.5 Regression 튜토리얼 (mlp + cnn)
  Phase 7.6 프레임워크 연계 체크리스트
```

## 6. Stage 7 계획 요약

### Phase 7.1 CLI 확장

scripts 4개에 `--model` 플래그 추가 (parse_args + build_config 패턴):
```python
parser.add_argument("--model", default="mlp", choices=["mlp", "cnn"])
# build_config에 "model": args.model 추가
```

### Phase 7.2 실험 실행

실행 환경: `conda run -n numpy_env` + `MPLBACKEND=Agg`
순서: visualize.py 먼저(디렉터리 생성) → train.py --checkpoint
6종 조합: multiclass/binary/regression × mlp/cnn

outputs/ 구조:
```
outputs/
├── multiclass/mlp/   training_log.png, predictions.png, model.npz
├── multiclass/cnn/
├── binary/mlp/
├── binary/cnn/
├── regression/mlp/
└── regression/cnn/
```

### Phase 7.3~7.5 튜토리얼 구조 (공통 8개 섹션)

```
## 1. 과제 개요    ## 2. 데이터 규약    ## 3. 모델 구성
## 4. 학습         ## 5. 평가           ## 6. 예측
## 7. 시각화       ## 8. 설계 결정
```

docs 파일 위치: `docs/stage7/{task}/phase7.{N}_tutorial-{model}.md`

### Phase 7.6 프레임워크 체크리스트

nn·data·optimizer·실행 객체 PyTorch 대응 표 + 주요 차이점 (autograd, zero_grad, no_grad) + 인터페이스 호환 체크박스

## 7. 다음 세션 시작 지시문

아래는 이전 세션의 컨텍스트입니다.
Stage 6 Phase 6.1~6.2는 완료되어 있고, Phase 6.0(CuPy 환경 구성)이 미시작 상태입니다.
Stage 7 Phase는 이번 세션에서 6개로 재정의했습니다.

`session-start 실행 후 Phase 6.0 CuPy 환경 구성부터 순서대로 진행해 주세요.`

참고 파일:
- 핸드오프: `_core/sessions/260617-235022_session-handoff.md`
- 할일: `_core/PROJECT-TODO.md`
- 스펙: `_core/PROJECT-SPEC.md`
- 계획 상세: `/home/nampl/.claude/plans/stage-7-iterative-hummingbird.md`
