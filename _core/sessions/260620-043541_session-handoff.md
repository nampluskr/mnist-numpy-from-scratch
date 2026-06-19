---
tags: [project, sessions]
created: 2026-06-20
updated: 2026-06-20
---

# 폴더 역할 재정의 및 experiments/ 구축 세션 핸드오프

> 작성일시: 260620-043541
> 세션 목적: 폴더 역할 체계 재정의 + experiments/ batch job 코드 신규 작성
> 이전 핸드오프: 260620-032534_session-handoff.md

## 1. 세션 핵심 요약

프로젝트 폴더 역할을 재정의하고 CLAUDE.md·PROJECT-SPEC.md를 현행화했다.
experiments/ 폴더에 subprocess 기반 batch job 스크립트 5개를 신규 작성했다.
tests/ 하위 잘못 생성된 `__init__.py` 3개를 삭제했다.

## 2. 확정된 폴더 역할 체계

| 폴더 | 역할 |
|---|---|
| `src/` | 재사용 가능한 소스 코드 (라이브러리) |
| `scripts/` | 사용자용 CLI 클라이언트 코드 — `src/core/` 실행 객체를 조립하는 진입점 |
| `tests/` | pytest 기반 TDD 테스트 코드 |
| `docs/` | 상세 매뉴얼 및 참조 문서 |
| `notebooks/` | 교육용 튜토리얼 노트북 |
| `experiments/` | CLI scripts 실행 예제 Python 스크립트 (batch job), 결과는 `outputs/{exp_name}/` |

## 3. experiments/ 구조 및 설계 결정

### 파일 구조

```text
experiments/
├── train_all.py
├── evaluate_all.py
├── predict_all.py
├── visualize_all.py
└── run_all.py
```

### 설계 결정 사항

| 항목 | 결정 |
|---|---|
| 실행 방식 | subprocess로 scripts/*.py 호출 |
| 조합 정의 | 명시적 CONFIGS 리스트 (itertools.product 미사용) |
| CONFIGS 관리 | run_all.py에 단일 정의, 각 main(configs, dataset_dir, seed)에 주입 |
| 단독 실행 | 각 *_all.py의 `if __name__ == "__main__"` 블록에 _CONFIGS 기본값 보유 |
| outputs 경로 | `outputs/{exp_name}/` — exp_name: `{task}_{model}_ep{epochs}_lr{lr}_bs{batch_size}` |
| 진행 표시 | `[i/total]`, 성공/실패: `[OK]`/`[FAIL]`, 종료: `[done] N/total success` |

### 실행 방법

```bash
# 전체 파이프라인
python experiments/run_all.py

# 단계별 단독 실행
python experiments/train_all.py
python experiments/evaluate_all.py
python experiments/predict_all.py
python experiments/visualize_all.py
```

## 4. 이번 세션 완료 항목

| 항목 | 내용 |
|---|---|
| CLAUDE.md 업데이트 | experiments/ 역할 설명 갱신, notebooks/ 설명 갱신 |
| PROJECT-SPEC.md §3 업데이트 | scripts/docs/notebooks/experiments/ 역할 불릿 추가·수정 |
| PROJECT-SPEC.md §6.3 업데이트 | 섹션 제목 변경, experiments/ 구조 및 설명 추가 |
| experiments/ 신규 작성 | train_all/evaluate_all/predict_all/visualize_all/run_all.py 5개 |
| tests/ __init__.py 삭제 | stage2/3/4 각 1개씩 총 3개 삭제 |
| PROJECT-LOG.md 갱신 | 세션 작업 내역 3줄 추가 |

## 5. 미결 사항

없음.

## 6. 다음 작업 목록

| 우선순위 | 작업 | 참고 |
|---|---|---|
| 1 | PyTorch 마이그레이션 프로젝트 시작 | `docs/stage7/phase7.5_framework-checklist.md` 체크리스트 참조 |

## 7. 다음 세션 시작 지시문

`session-start 실행` 후 PyTorch 마이그레이션 프로젝트 시작 여부를 확인해 주세요.

참고 파일:
- 핸드오프: `_core/sessions/260620-043541_session-handoff.md`
- 프레임워크 체크리스트: `docs/stage7/phase7.5_framework-checklist.md`
