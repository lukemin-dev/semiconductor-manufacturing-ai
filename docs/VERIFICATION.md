# 실행 검증 기록

## v2 로컬 검증

- macOS Python 3.12: pytest **9 passed**.
- Ruff 기본 오류 검사 및 자동 포맷 확인.
- 8개 후보 × 15 folds = 120개 fold 학습. 기존 test를 모델 선택에 사용하지 않음.
- 시간순 split에 동일 timestamp가 겹치지 않음을 검증.
- 실제 Linux/PostgreSQL 및 GitHub CI 결과는 마지막 검증 후 이 문서에 기록.

## v1 기록

- macOS 6 passed, PostgreSQL 통합 검사 3 passed, Linux Docker 6 passed.
- 대시보드 예측·TOP5·이력 표시를 실제 브라우저에서 확인.

테스트 통과는 모델의 높은 성능 또는 현장 적용 가능성을 의미하지 않습니다. 평가 결과와 한계는 README를 함께 확인하세요.
