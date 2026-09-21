"""Summarize v3 evidence, including negative results, without selecting on test."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/v3"


def main():
    comparison = pd.read_csv(OUT / "comparison.csv").set_index("model")
    nested = pd.read_csv(OUT / "nested_selected_predictions.csv")
    rows = []
    from sklearn.metrics import average_precision_score

    for fold, group in nested.groupby("outer_fold"):
        pred = group.risk_score >= group.threshold
        tp = int(((group.actual_fail == 1) & pred).sum())
        rows.append(
            {
                "outer_fold": int(fold),
                "selected_model": group.model.iloc[0],
                "ap": float(
                    average_precision_score(group.actual_fail, group.risk_score)
                ),
                "recall": tp / int(group.actual_fail.sum()),
                "precision": tp / int(pred.sum()) if pred.sum() else 0.0,
                "alert_rate": float(pred.mean()),
            }
        )
    nested_metrics = pd.DataFrame(rows)
    nested_metrics.to_csv(OUT / "nested_selected_metrics.csv", index=False)
    audit = json.loads((OUT / "audit.json").read_text())
    baseline = comparison.loc["rf_weighted_baseline"]
    table = [
        "| Candidate | Outer AP | Recall@30% | Temporal AP | Temporal Recall@30% | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, row in comparison.iterrows():
        table.append(
            f"| {name} | {row.outer_ap_mean:.3f} | {row.outer_recall_at_30:.3f} | {row.temporal_ap_mean:.3f} | {row.temporal_recall_at_30:.3f} | {'pass' if row.passes_research_gate else 'fail'} |"
        )
    passed = list(comparison.index[comparison.passes_research_gate])
    decision = {
        "research_gate_passed": passed,
        "deployment_changed": False,
        "reason": "No candidate passes predefined outer AP + temporal AP + temporal recall gates"
        if not passed
        else "Research gate passed; independent evaluation still required before promotion",
        "historical_test_evaluated": False,
        "nested_selected_outer_ap_mean": float(nested_metrics.ap.mean()),
        "nested_selected_outer_ap_std": float(nested_metrics.ap.std()),
        "v2_artifacts_unchanged": audit["v2_artifacts_unchanged"],
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2))
    report = (
        f"""# v3 연구 결과: 결측 정보와 균형 표본추출

## 결론

**배포 모델은 v2를 유지합니다.** 비교한 6개 후보 중 사전에 정한 개선 조건을 모두 통과한 후보가 없습니다. 교차검증의 작은 개선을 시간순 일반화 개선으로 과장하지 않았습니다. 기존 test 392건은 v3에서 평가하지 않았으며 v2 모델·metadata·metrics·test predictions 파일의 SHA-256이 그대로임을 확인했습니다.

## 1. 놓친 불량 분석

v2의 반복 OOF 점수를 행별로 평균한 후 기존 임계값을 적용한 설명용 분석입니다. 평균 점수의 분포는 개별 OOF 점수와 다르므로 기존 지표와 같은 평가로 취급하지 않습니다. development 1,175건 중 불량 78건에서 37건이 누락되었습니다.

- 누락 불량의 평균 결측률은 4.27%, 탐지 불량은 4.10%로 차이는 약 0.17%p입니다. 결측률 하나로 누락을 설명할 근거는 부족합니다.
- 월별 누락은 7월 0/8건, 8월 16/39건, 9월 8/13건, 10월 13/18건입니다. 시기에 따른 차이는 보이지만 작은 표본과 무작위 교차검증 결과로 실제 drift나 원인을 확정할 수 없습니다.
- 이 분석에는 historical test를 사용하지 않았습니다. 월별·행별 근거는 `v2_failure_by_month.csv`, `v2_failure_rows.csv`에 있습니다.

## 2. 실험 설계

기존 development 1,175건만 사용했습니다. 5개 outer fold 각각의 학습 부분에서 3-fold inner CV로 후보 모델과 threshold를 선택하고, 바깥 validation에서 평가했습니다. 각 후보별 outer 결과는 비교용이며 그중 최고 후보의 점수는 모델 선택 편향에서 자유롭지 않습니다. 선택 절차 자체의 outer 평균 AP는 **{nested_metrics.ap.mean():.3f} ± {nested_metrics.ap.std():.3f}**입니다. fold 편차는 독립 표본 표준오차가 아닙니다.

모든 feature 필터, 중앙값, 결측 indicator 스키마, 상관 변수 제거, 균형 표본추출은 해당 학습 부분에서만 fit했습니다. tree 후보들에는 표준화를 생략했으므로 baseline은 v2 계열의 비교 기준이며 v2 실행 전체와 동일한 재현 실험은 아닙니다.

추가로 development 내부의 연속된 시간 블록에서 과거 학습 → 다음 구간 threshold 설정 → 그 다음 구간 평가를 3회 수행했습니다. 모델 종류를 이 평가로 튜닝하지 않았습니다. 동일 시각은 분할 경계를 넘지 않습니다. 각 구간의 불량 수가 적고, 개발 데이터 자체는 이미 탐색했으므로 외부 독립 검증이 아닙니다.

학습 횟수: 6개 후보 × 5 outer × (3 inner + 1 outer fit) + 6개 후보 × 3 시간 구간 = **138회**.

## 3. 같은 분할에서 비교

"""
        + "\n".join(table)
        + f"""

`Recall@30%`는 각 평가 배치에서 점수가 높은 상위 floor(0.3 × 표본 수)개만 검토했을 때 탐지하는 불량 비율입니다. 정답을 보지 않고 점수로 정렬하며 동점은 입력 순서를 따릅니다. 고정 점수 임계값의 Recall과 다릅니다. 10%·20% 결과도 comparison.csv에 저장했습니다.

- 결측 indicator 추가는 outer AP를 {baseline.outer_ap_mean:.3f}에서 {comparison.loc["rf_missing", "outer_ap_mean"]:.3f}로 높였으나, 시간순 AP는 {baseline.temporal_ap_mean:.3f}에서 {comparison.loc["rf_missing", "temporal_ap_mean"]:.3f}로 낮췄습니다.
- 균형 표본추출은 이번 비교에서 일관된 개선을 보여주지 못했습니다. 복잡한 기법의 추가 자체를 개선으로 간주하지 않습니다.
- 시간순 baseline의 평균 Recall@30%는 {baseline.temporal_recall_at_30:.3f}로, 검사량 30%에서 Recall 90%라는 도전 목표에 미달합니다.
- 이 결과와 기존 v2 test 수치는 표본·분할이 다르므로 직접적인 전후 성능 비교에 사용하지 않습니다.

## 4. 배포 결정

사전에 정한 기준은 baseline 대비 outer 평균 AP +0.02 이상, 시간순 평균 AP +0.02 이상, 시간순 Recall@30% 비감소였습니다. 통과 후보: **{", ".join(passed) if passed else "없음"}**. 이 기준은 개인 프로젝트의 연구 판단 규칙이며 실제 공장의 요구조건이나 통계적 유의성 검정이 아닙니다.

따라서 **실행 중인 v2 모델과 threshold를 변경하지 않습니다.** 새로운 기법이 별 효과가 없었다는 결과도 코드·분할·예측과 함께 공개합니다. 판정은 `decision.json`, artifact 보존 근거는 `audit.json`에 있습니다.

## 5. 다음 판단

SECOM에서 같은 데이터로 후보를 계속 늘리는 것만으로 목표 달성을 보장할 수 없습니다. 다음 우선순위는 독립된 새 시간대 데이터, 실제 공정/장비 키, feature 측정 시점과 의미를 확보하는 것입니다. 추가 데이터가 없다면 모델 성능 주장을 키우기보다 검증 절차와 검사 우선순위의 한계를 설명하는 데 집중합니다.

참고: [BalancedRandomForest 공식 문서](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html), [scikit-learn 교차검증](https://scikit-learn.org/stable/modules/cross_validation.html).
"""
    )
    (ROOT / "docs/V3_RESULTS.md").write_text(report)
    labels = list(comparison.index)
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.barh(
        positions - 0.17,
        comparison.outer_ap_mean,
        height=0.32,
        label="Outer CV AP",
        color="#147D92",
    )
    ax.barh(
        positions + 0.17,
        comparison.temporal_ap_mean,
        height=0.32,
        label="Chronological AP",
        color="#DD9144",
    )
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.3)
    ax.set_xlabel("Average precision (AP)")
    ax.set_title(
        "V3: small CV gains do not establish temporal improvement", loc="left", pad=16
    )
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "docs/v3-comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
