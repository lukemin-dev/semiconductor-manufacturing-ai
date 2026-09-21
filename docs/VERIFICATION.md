# 실행 검증 기록

## v2 검증 완료

- macOS Python 3.12: pytest **9 passed**.
- Linux ARM64 Docker + 실제 PostgreSQL 16: pytest **9 passed**.
- Ruff 기본 오류 검사 및 포맷 검사 통과.
- 8개 후보 × 15 folds = **120개 fold 학습**. 기존 test를 모델 선택에 사용하지 않음.
- 시간순 split에 동일 timestamp가 겹치지 않음을 검증.
- 브라우저에서 예측 → TOP5 → 이력 저장 확인. SECOM-0170 경보의 `open → acknowledged` 변경 및 예측값 유지 확인.
- v1/v2 모델 버전의 과거 이력이 함께 보존됨을 확인.
- GitHub Actions: [실제 성공한 실행 기록](https://github.com/lukemin-dev/semiconductor-manufacturing-ai/actions/runs/35503597000). PostgreSQL 테스트, Ruff, Compose 검증, Docker 빌드 통과. 검증한 코드 commit: `fe52cbbc212344b11454ec5ca2266fdefe591180`.
- 마지막 문서·스크린샷 갱신은 기능 코드 변경을 포함하지 않음. 최신 실행 상태는 저장소 Actions에서 확인 가능.

## v1 기록

macOS 6 passed, PostgreSQL 통합 검사 3 passed, Linux Docker 6 passed. 대시보드 예측·TOP5·이력 표시를 실제 브라우저에서 확인.

테스트 도구의 deprecation 경고 2건은 실패가 아닙니다. 테스트 통과는 모델의 높은 성능 또는 현장 적용 가능성을 의미하지 않습니다. 평가 결과와 한계는 README를 함께 확인하세요.
