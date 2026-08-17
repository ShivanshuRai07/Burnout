# Burnout Model Evaluation

## Outcome

The production candidate is **HistGradientBoostingRegressor** for the continuous `burnout_score` (0–10). It achieved the best held-out accuracy while using only operational and work-context variables.

## Evaluation protocol

- Dataset: 150,000 rows from `tech_mental_health_burnout.csv`
- Split: random 80/20 hold-out, seed 42 (30,000 unseen test rows)
- Regression target: `burnout_score`
- Risk target: score at or above the training split's 90th percentile (3.70); a transparent statistical flag, not a diagnosis
- Excluded by default: age, gender, anxiety/depression, therapy, and help-seeking fields

| Regression model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| Ridge | 0.4953 | 0.6226 | 0.6876 |
| Random Forest | 0.4298 | 0.5742 | 0.7343 |
| HistGradientBoosting **(recommended)** | **0.4221** | **0.5630** | **0.7446** |
| XGBoost | 0.4216 | 0.5649 | 0.7428 |

The calibrated elevated-risk classifier achieved ROC-AUC **0.9508**, PR-AUC **0.7323**, and F1 **0.6560** at its 0.50 threshold. Its held-out positive rate was 11.01%.

## Explainability

XGBoost was retained as the explainability companion because SHAP natively supports its tree ensemble. The leading influences were stress level, weekly work hours, work–life balance, manager support, overtime, and social support. The SHAP chart confirms the expected directions: higher stress/work hours/overtime increase predicted burnout, while higher work–life balance and manager support reduce it.

## Deployment boundary

This dataset is an external snapshot and does not establish longitudinal workplace causality. Before any real deployment, validate on consented organization data using time-based splits, review fairness and calibration by approved cohorts, set intervention policy with HR/legal, and monitor drift. Use predictions to offer supportive workload and manager interventions—never for adverse employment actions.
