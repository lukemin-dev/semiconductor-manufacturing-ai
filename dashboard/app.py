import sys, os, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import requests
import streamlit as st
from src.data import load_secom

st.set_page_config(page_title="SECOM · Manufacturing AI", layout="wide")
st.title("SECOM · Manufacturing AI")
st.caption("공개 반도체 데이터로 검증한 불량 탐지와 추적 시스템 | 개인 포트폴리오")
API = os.environ.get("API_URL", "http://localhost:8000")


@st.cache_data
def data():
    return load_secom()


ds = data()
try:
    response = requests.get(API + "/model", timeout=10)
    response.raise_for_status()
    meta = response.json()
except requests.RequestException:
    st.error(
        "예측 API에 연결할 수 없습니다. docker compose up --build 실행 상태를 확인하세요."
    )
    st.stop()
st.info(
    "Lot은 시연을 위해 25개 행씩 묶은 가상 식별자입니다. 위험 점수는 보정된 불량 확률이 아닙니다. 원인 후보는 실제 공정 원인으로 확정할 수 없습니다."
)
metrics = json.loads((ROOT / "artifacts/metrics.json").read_text())
selected = metrics[meta["selected_model"]]
cols = st.columns(4)
for c, label, key in zip(
    cols,
    ["Fail Recall", "Precision", "F1", "PR-AUC (AP)"],
    ["fail_recall", "fail_precision", "f1", "pr_auc"],
):
    c.metric(label, f"{selected[key]:.3f}")
st.caption(
    f"선정 모델: {meta['selected_model']} | 버전: {meta['model_version']} | 판정 임계값: {meta['threshold']:.4f}"
)
lot = st.selectbox("Demo Lot", sorted({f"DEMO-{i // 25:03d}" for i in ds.X.index}))
indices = [i for i in ds.X.index if f"DEMO-{i // 25:03d}" == lot]
idx = st.selectbox("샘플", indices, format_func=lambda i: f"SECOM-{i:04d}")
splits = json.loads((ROOT / "artifacts/splits.json").read_text())
partition = next(name for name, ids in splits.items() if idx in ids)
st.caption(f"데이터 분할: {partition} | 상단 성능 지표는 test 샘플에서만 계산했습니다.")
st.write(
    f"실제 라벨: {'Fail' if ds.y.loc[idx] else 'Pass'} · 기록 시각: {ds.timestamps.loc[idx]}"
)
if st.button("예측 및 이력 저장", type="primary"):
    features = {k: None if pd.isna(v) else float(v) for k, v in ds.X.loc[idx].items()}
    try:
        r = requests.post(
            API + "/predict",
            json={"sample_id": f"SECOM-{idx:04d}", "lot_id": lot, "features": features},
            timeout=60,
        )
        r.raise_for_status()
        st.session_state["prediction"] = r.json()
    except requests.RequestException as exc:
        st.error(str(exc))
r = st.session_state.get("prediction")
if r and r["sample_id"] == f"SECOM-{idx:04d}":
    st.subheader("불량 위험도 · root-cause candidates")
    st.metric(
        "위험 점수",
        f"{r['risk_score']:.4f}",
        delta="Alert" if r["predicted_fail"] else "Pass candidate",
        delta_color="off",
    )
    st.dataframe(pd.DataFrame(r["top_features"]), hide_index=True)
    st.caption(
        "contribution = 현재 점수 − 해당 feature를 학습 중앙값으로 바꾼 점수. 양수는 위험 점수를 높인 후보, 음수는 낮춘 후보입니다. SHAP 값이 아니며 합산할 수 없습니다."
    )
    st.caption(f"결측 비율 {r['missing_fraction']:.1%} · 예측 ID {r['prediction_id']}")
st.subheader("Lot 예측 · Alert history")
try:
    h = requests.get(API + "/history", params={"lot_id": lot}, timeout=10)
    h.raise_for_status()
    rows = h.json()
    st.dataframe(
        pd.DataFrame(
            [
                {
                    k: v
                    for k, v in row.items()
                    if k not in ["top_features", "attribution_method"]
                }
                for row in rows
            ]
        ),
        hide_index=True,
    )
except requests.RequestException as exc:
    st.error(str(exc))
with st.expander("동일 테스트 데이터에서 모델 비교"):
    st.dataframe(pd.DataFrame(metrics).T)

open_alerts = (
    [row["prediction_id"] for row in rows if row["alert_status"] == "open"]
    if "rows" in locals()
    else []
)
if open_alerts:
    alert_id = st.selectbox("확인할 경보 ID", open_alerts)
    if st.button("경보 확인 완료"):
        try:
            response = requests.post(f"{API}/alerts/{alert_id}/acknowledge", timeout=10)
            response.raise_for_status()
            st.success("경보를 확인 처리했습니다. 예측 결과는 변경되지 않습니다.")
            st.rerun()
        except requests.RequestException as exc:
            st.error(str(exc))

with st.expander("검증 방식과 불확실성"):
    st.write(
        "모델은 development 데이터의 반복 교차검증 평균 AP로 선택했습니다. 기존에 확인한 test를 재사용한 탐색 결과이며 독립된 신규 검증이 아닙니다."
    )
    cv_path = ROOT / "artifacts/cv_summary.csv"
    if cv_path.exists():
        st.dataframe(pd.read_csv(cv_path), hide_index=True)
    ci_path = ROOT / "artifacts/confidence_intervals.json"
    if ci_path.exists():
        st.json(json.loads(ci_path.read_text()))
    st.caption(
        "95% 구간은 고정 모델·테스트 데이터에서 계산한 층화 bootstrap 구간입니다. 학습·모델 선택 불확실성을 모두 포함하지 않습니다."
    )
