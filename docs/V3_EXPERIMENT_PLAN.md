# v3: failure analysis and imbalance experiments

## Scope fixed before fitting

Use only the v2 development partition (1,175 rows); do not score the historical 392-row test. All results are exploratory development evidence, not a new independent test.

1. Diagnose v2 OOF false negatives by missingness and calendar period. Average repeated OOF scores per row for descriptions, not for selecting deployment thresholds. Use the frozen v2 threshold for the descriptions. Report group sizes; differences are associations, not physical causes.
2. Compare six predefined candidates with the same 5 outer stratified folds and 3 inner folds:
   - v2 class-weighted Random Forest baseline;
   - baseline + missingness indicators, including training-variable missingness of dropped raw features;
   - baseline + missingness indicators + fold-local correlation pruning at |r| >= 0.98;
   - Balanced Random Forest with leaf size 2;
   - Balanced Random Forest + missingness indicators with leaf size 2;
   - Balanced Random Forest + missingness indicators with leaf size 5.
3. Within each outer training fold, select model by mean inner AP. Learn threshold from its inner OOF predictions only (F2, review-rate cap 30%). Evaluate once on the outer validation fold. Also report each candidate's outer AP and Recall-at-review-budget 10/20/30% as ranking diagnostics; budget selection uses scores, not labels.
4. Independently run three expanding chronological development windows. For each candidate, train on earlier data, tune threshold on the next time block, and assess the subsequent time block. Fit every feature transform and resampler on training rows only. Do not select models on temporal assessment labels.
5. Publish failures as well as gains. Do not replace the running v2 model unless a candidate exceeds baseline mean outer AP by 0.02 AND mean chronological AP by 0.02, without reducing chronological Recall@30%. These are project acceptance rules, not industry requirements. Promotion would still require a new evaluation; this experiment alone does not establish field readiness.

## Artifacts and checks

Persist the plan, candidate settings, fold IDs, inner selection decisions, outer scores, review-budget diagnostics, temporal predictions, data-quality diagnostics and a generated report. Add tests for fold-local missingness, correlation filtering, review-budget accounting and partition separation. Keep v2 inference artifacts intact.
