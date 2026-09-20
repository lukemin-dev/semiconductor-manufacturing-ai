# SECOM Manufacturing AI — Fault Detection & Root-Cause Candidates

공개 반도체 데이터의 결측·불균형을 검토하고, 예측부터 원인 후보·모델 버전·Alert 이력까지 연결한 개인 포트폴리오입니다. 실제 Fab 배포나 수율 개선 성과를 주장하지 않습니다.

## 빠른 실행

Python 3.12, Docker Engine과 Compose가 필요합니다. 데이터와 학습 모델을 ZIP에 포함했습니다.

```bash
docker compose up --build -d
# 이 Mac처럼 Compose가 별도 설치되었다면: docker-compose up --build -d
```

Dashboard: http://localhost:8501 · API 문서: http://localhost:8000/docs · 준비 확인: http://localhost:8000/health
첫 실행 중 API가 준비되기 전에는 화면을 새로고침하세요. 화면에서 Demo Lot과 샘플을 선택한 뒤 ‘예측 및 이력 저장’을 누릅니다. 결과와 동일 Lot의 이력이 표시됩니다. API 호출 한 번은 하나의 예측 이벤트이며, 반복 요청도 각각 기록합니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
python scripts/download_data.py  # ZIP에 이미 데이터가 있으면 생략
python scripts/train.py          # 실제 데이터 재학습; artifacts 덮어쓰기
pytest -q
docker compose up -d db
export DATABASE_URL=postgresql+psycopg2://secom:secom_local@localhost:55432/secom
uvicorn api.main:app --host 127.0.0.1 --port 8000
# 별도 터미널에서 같은 가상환경 사용
streamlit run dashboard/app.py
```

재학습 후 서비스가 새 모델을 읽도록 Docker 이미지를 다시 빌드하세요. 기본 DB 계정은 로컬 시연 전용이며, 포트는 localhost에만 열립니다. 종료는 `docker compose down`이며 DB 볼륨은 보존됩니다.

## 실제 데이터와 평가 설계

[UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom), McCann & Johnston (2008), [DOI](https://doi.org/10.24432/C54305), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
공식 설명은 591 feature로 표기하지만 다운로드한 `secom.data`를 파싱한 실제 입력 행렬은 **1,567 × 590**입니다. 이를 임의로 591개로 맞추지 않았습니다. Fail 104건(6.64%), 전체 셀 결측률 4.54%, 중복 행 0건입니다. 파일별 SHA-256은 `artifacts/data_manifest.json`에 있습니다.

- 고정 seed 42, 층화 train 783건(52 Fail) / validation 392건(26 Fail) / test 392건(26 Fail).
- 학습 데이터에서만 결측률 40% 초과 및 상수 feature 제거, 중앙값 대체, 표준화 수행. 442 feature 유지.
- Logistic Regression, Random Forest, XGBoost 각각 가중치 미적용/적용 비교. IsolationForest는 정상 학습 데이터만 사용하며, 검증 라벨로 임계값을 선택하므로 완전한 비지도 평가가 아닙니다.
- 모델은 validation AP 우선, 동률이면 F1로 선택. 각 모델의 임계값은 validation F2 최대값으로 고정합니다. F2는 Recall에 더 높은 비중을 주는 시연용 정책으로, 실제 비용 최적값을 의미하지 않습니다.
- 선택 후 재학습 없이 같은 test 392건에서 모든 모델을 한 번 비교합니다. 테스트 결과로 선정 모델 또는 임계값을 변경하지 않습니다.
- IsolationForest는 음수 decision function 원점수를 사용하며 테스트 범위 정규화를 하지 않습니다. 점수는 확률이 아닙니다.

## 실제 실행 결과

| 모델 | Recall | Precision | F1 | PR-AUC (AP) | TP/FN/FP |
|---|---:|---:|---:|---:|---|
| logistic_unweighted | 0.269 | 0.101 | 0.147 | 0.079 | 7/19/62 |
| random_forest_unweighted | 0.808 | 0.143 | 0.243 | 0.179 | 21/5/126 |
| logistic_balanced | 0.269 | 0.104 | 0.151 | 0.079 | 7/19/60 |
| random_forest_balanced | 0.923 | 0.096 | 0.174 | 0.170 | 24/2/226 |
| xgboost_unweighted | 0.385 | 0.135 | 0.200 | 0.145 | 10/16/64 |
| xgboost_balanced | 0.923 | 0.112 | 0.199 | 0.134 | 24/2/191 |
| isolation_forest | 0.962 | 0.066 | 0.123 | 0.132 | 25/1/354 |
| always_pass | 0.000 | 0.000 | 0.000 | 0.066 | 0/26/0 |

선정 모델: **logistic_balanced**, 버전 `secom-20260920-093213`, 임계값 0.136046. 검증 AP는 0.225이지만 테스트 AP는 0.079로 낮아졌습니다. 테스트에서 불량 26건 중 7건 탐지, 19건 누락, 정상 60건 오경보입니다. **현장 적용 가능한 모델이라는 결론을 내릴 수 없습니다.**

Random Forest 미가중 모델은 테스트에서 21건을 잡지만 정상 126건을 경보로 분류했습니다. 이는 비교 결과일 뿐, 테스트 결과를 이용해 운영 모델을 교체했다는 뜻이 아닙니다. 가중치가 Recall을 높여도 Precision·AP가 반드시 좋아지지는 않았습니다. 소수의 Fail과 단일 분할에 따른 불안정성을 보여줍니다.

Accuracy만 보면 전부 Pass인 모델도 93.4%입니다. Fail Recall은 불량 누락을, Precision은 경보 후 검사 부담을 나타냅니다. F1은 둘의 균형, PR-AUC는 불균형 데이터에서 임계값 전반의 Precision/Recall 품질을 비교하는 데 사용합니다. 여기서 PR-AUC는 사다리꼴 면적이 아니라 scikit-learn의 **Average Precision(AP)**입니다. Recall만 높이면 IsolationForest처럼 대부분을 경보로 만들 수 있으므로 FP/FN을 같이 봐야 합니다.

## 아키텍처와 기술 선택

```mermaid
flowchart LR
 A[UCI raw + SHA256] --> B[Train-only preprocessing]
 B --> C[Validation model / threshold selection]
 C --> D[Frozen held-out test]
 C --> E[Versioned model artifact]
 E --> F[FastAPI inference + local attribution]
 G[Streamlit sample / demo Lot] --> F
 F --> H[(PostgreSQL)]
 H --> I[process_data / prediction_result]
 H --> J[model_version / alert_history]
 H --> G
```

scikit-learn Pipeline으로 학습/추론 전처리를 공유하고, XGBoost로 비선형 boosting을 비교했습니다. 사용할 수 없으면 HistGradientBoosting으로 대체하고 이유를 metadata에 기록합니다. FastAPI는 입력 검증과 서비스 경계를, SQLAlchemy/PostgreSQL은 외래키와 트랜잭션 기반 추적을 제공합니다. Streamlit은 샘플 탐색·검토 화면에, Docker는 실행환경 재현에 사용합니다.

`POST /predict`는 관측된 유효 feature가 최소 하나 있어야 하며 오타 feature, 무한대, 빈 입력을 거절합니다. 부분 결측은 학습 중앙값으로 대체하고 결측 비율을 반환합니다. `GET /model`, `/health`, `/history?lot_id=DEMO-000`을 제공합니다. 원본 입력·예측·Alert는 한 트랜잭션으로 저장되며 모델 버전은 외래키로 연결됩니다. 실제 날짜 정보는 원자료 조회에만 사용하고 모델 입력에서 제외했습니다.

## 원인 후보 해석

각 샘플에서 feature 하나를 학습 중앙값으로 바꾼 전후 위험 점수 차이를 구해 절댓값 TOP5를 반환합니다. `contribution > 0`은 관측값이 점수를 높였다는 뜻입니다. **SHAP이 아닌 local median replacement attribution**이며 기여도 합은 예측값과 일치하지 않습니다. global importance는 테스트 샘플의 절대 변화량 평균으로, 사후 설명에만 쓰고 모델 선정에 쓰지 않습니다.

익명 feature, 상관관계, 상호작용 때문에 후보가 물리적 원인이라는 보장은 없습니다. 항상 **root-cause candidate**로 표현합니다. 원인 확정에는 설비·공정 의미 매핑, 엔지니어 검토, 통제된 추가 검증이 필요합니다. 위험 점수도 보정된 불량 확률이 아닙니다.

## 화면

![Dashboard](docs/dashboard.png)

![원인 후보와 저장 이력](docs/dashboard-detail.png)

## 검증과 산출물

- `artifacts/metrics.json`, `validation_metrics.json`: 실제 모델별 결과와 선택 근거.
- `splits.json`, `data_audit.json`, `feature_audit.csv`: 분할·전처리 감사 기록.
- `all_model_predictions.csv`, `test_predictions.csv`: 동일 test의 모델별 점수/예측/정답.
- `local_attributions.csv`, `feature_importance.csv`: 샘플 TOP5와 전체 feature 중요도.
- `model.joblib`, `metadata.json`: 학습 파이프라인과 모델 버전. 신뢰하는 프로젝트 artifact만 로드하세요.
- `sql/schema.sql`: 실제 ORM에서 생성한 PostgreSQL DDL.
- `tests/`: 전처리 격리, 선정 규칙, attribution 일치, 잘못된 입력, API/DB/Alert 검증.
- `.github/workflows/ci.yml`: PostgreSQL 서비스에서 pytest 및 Docker 빌드. 원격 GitHub 실행은 저장소 push 후 수행됩니다.
- `docs/PORTFOLIO.md`, `APPLICATIONS.md`, `INTERVIEW.md`: 경험 연결, 회사별 지원 문구, 면접 답변.

## 한계와 다음 실험

테스트 Fail이 26건뿐이고 단일 층화 분할이므로 성능 추정 변동이 큽니다. 시계열 순방향 검증·그룹 분할은 아직 수행하지 않았으며, 실제 장비/Lot 정보도 없습니다. 랜덤 분할에서 동일 행 누수는 막았지만 시간·설비 의존성까지 배제할 수는 없습니다. 원자료의 측정 시점이 명확하지 않아 실제 사전 예측 가능성도 검증 대상입니다.

후속 실험은 학습 영역 안의 반복/중첩 교차검증, 시간순 홀드아웃, 확률 보정, 오경보 처리 용량을 고려한 임계값 선택입니다. 이미 확인한 test를 계속 튜닝 목표로 사용하지 않습니다. 인증·접근제어·모델 레지스트리·drift 감시·실제 설비 연동·대량 요청 성능은 범위 밖이며 운영 성과를 주장하지 않습니다. 제공된 시스템은 검증과 추적을 시연하는 개인 프로젝트입니다.

## 검증 완료 기록

2026-09-20: macOS pytest 6개 통과, 실제 PostgreSQL 통합 검사 3개 통과, Linux Docker 내 전체 pytest 6개 통과. Compose 서비스 3개 실행과 브라우저의 샘플 예측 → 후보 TOP5 → 이력 표시를 확인했습니다. 두 건의 테스트 도구 deprecation 경고는 실패가 아닙니다. GitHub Actions 파일은 작성했으나 원격 실행은 하지 않았습니다. 로그는 artifacts에 포함했습니다.

2페이지 요약본: `output/pdf/SECOM_Portfolio_Summary.pdf`. 재생성은 `pip install reportlab` 후 `python scripts/create_summary.py`이며 동봉된 Nanum Gothic/OFL 라이선스를 사용합니다.
