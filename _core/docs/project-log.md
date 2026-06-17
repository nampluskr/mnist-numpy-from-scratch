---
tags: [project, docs]
created: 2026-06-15
updated: 2026-06-17
---

# project-log.md

이 프로젝트의 주요 작업 이력을 기록한다.
에이전트가 주요 변경 후 갱신한다.

| Date | 작업 내용 | 비고 |
|---|---|---|
| 2026-06-15 | 워크스페이스 초기화 - `_core/legacy/refs/`의 PROJECT.md, PROJECT-TODO.md 내용을 `_core/docs/project-spec.md`, `_core/docs/project-todo.md`에 반영 | project-todo.md는 전체 미완료 상태로 초기화 |
| 2026-06-15 | CLAUDE.md, project-guide.md 플레이스홀더 채움 (프로젝트명, 목적, 날짜) | |
| 2026-06-17 | Phase 1.1 완료 - requirements.txt, src/config.py, tests/stage1/test_config.py, docs/stage1/phase1.1_config.md | |
| 2026-06-17 | 환경 확정 - numpy_env (Python 3.11), jupyterlab, ipykernel 설치 및 커널 등록 | |
| 2026-06-17 | 구조 확정 - stage 폴더명 0패딩 제거, tests/__init__.py 금지, pyproject.toml 삭제, conftest.py 경로 설정 | coding-rules.md §8 반영 |
| 2026-06-17 | Phase 2.3 완료 - src/data/dataloader.py, tests/stage2/test_dataloader.py (13개), docs/stage2/phase2.3_dataloader.md | Stage 2 전체 54개 테스트 통과 |
| 2026-06-17 | session-end.md Step 6 추가 - 종료 브리핑 후 사용자 승인을 받아 커밋·푸시 진행하는 절차 추가 | |
| 2026-06-17 | 레거시 코드 전체 구조 추가 - task 스크립트 6개(manual·module 각 3종) + common 모듈 6개, numpy_env 실행 검증 완료 | 기존 단일 파일 3개 삭제 |
| 2026-06-17 | Stage 0 문서 전면 재작성 - "재검토" 프레임 제거, 레거시 분석·구현 계획·테스트 계획 수립 중심으로 재편 | phase0.1~0.3 |
| 2026-06-17 | project-spec.md 전면 업데이트 - src/models/ 하위 구성요소, core/optimizers.py 추가, tests stage 단위 구조, 인터페이스 규약 확장 | |
| 2026-06-17 | project-todo.md 재구성 - Stage 0 미완료 초기화(다음 세션 재진행), Stage 4 Phase 4.1 optimizers 추가 및 재번호 | |
| 2026-06-17 | Stage 0 Phase 0.1~0.3 전면 재작성 - phase0.1_legacy-analysis.md, phase0.2_implementation-plan.md, phase0.3_test-plan.md 신규 작성, 기존 3개 파일 삭제 | Stage 0 전체 완료 |
| 2026-06-17 | Phase 명칭 전면 개선 - project-todo.md, project-spec.md Phase 1.1~7.3 명칭을 "동사구: 항목 나열" 형식으로 개선, 10개 phase 문서 H1 반영 | |
| 2026-06-17 | em dash 전면 제거 및 문서 규칙 추가 - 19개 파일 `—` → ` - ` 치환, docs-rules.md에 키보드 입력 불가 문자 사용 금지 조항 추가 | |
| 2026-06-17 | Phase 명 구분자 변경 - phase 헤딩 행의 ` - `를 `: `로 변경 (project-todo.md, project-spec.md, docs/stage*/phase*.md) | |
| 2026-06-17 | src 구조 재설계 - models/ 하위 layers/activations/losses를 nn/ 패키지로 분리, torch.nn 대응 명시, project-spec.md §5.4·§6.2·§6.5·§6.6 갱신 | PyTorch 방식 통일 결정 |
| 2026-06-17 | Stage 3 전면 재구성 - Phase 2→4개 분리(activations/layers/losses/mlp), src/nn/ 4파일 신규, mlp.py Sequential 기반 재작성, tests/stage3 테스트 69개 통과 | logit 출력, *_grad 함수 도입 |
| 2026-06-17 | Stage 3 문서 4개 작성 - phase3.1_activations.md, phase3.2_layers.md, phase3.3_losses.md, phase3.4_mlp.md, 구 버전 phase3.1_mlp.md 삭제 | Stage 3 전체 완료 |
| 2026-06-17 | Phase 4.1 완료 - src/core/optimizers.py (SGD, Adam), tests/stage4/test_optimizers.py (12개), docs/stage4/phase4.1_optimizers.md | |
| 2026-06-17 | Phase 4.2 완료 - src/core/checkpoints.py (save/load), tests/stage4/test_checkpoints.py (7개), docs/stage4/phase4.2_checkpoints.md | |
| 2026-06-17 | Phase 4.3 완료 - src/core/trainer.py (Trainer.fit), tests/stage4/test_trainer.py (16개), docs/stage4/phase4.3_trainer.md | |
