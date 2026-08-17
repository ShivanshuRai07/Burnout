# Burnout ML Pipeline

This project trains two separate models from `tech_mental_health_burnout.csv`:

- A continuous burnout-score regressor (`burnout_score`, 0–10), compared across Ridge, Random Forest, HistGradientBoosting, and XGBoost.
- A calibrated elevated-risk classifier. “Elevated” is defined from the training split’s 90th percentile of burnout score, preventing test-set leakage.

## Run

```powershell
python train.py --data "C:\path\tech_mental_health_burnout.csv" --output artifacts
```

## Artifacts

- `metrics.json` — held-out comparison and risk metrics.
- `burnout_score_model.joblib` — selected regression pipeline, including preprocessing.
- `elevated_risk_model.joblib` — calibrated classification pipeline.
- `feature_importance.csv`, `shap_summary.png`, `prediction_scatter.png` — global explainability and evaluation assets.

The default model deliberately excludes demographic and direct health/treatment fields (`age`, `gender`, therapy, help-seeking, anxiety, depression). This is a safer default for workplace decision support. Any production deployment requires consented internal validation, privacy review, fairness testing, monitoring, and human oversight; it must not make adverse employment decisions.

## Dashboard

Run `run_dashboard.ps1` from PowerShell, then open `http://127.0.0.1:8501`. The dashboard provides organization-level summaries, burnout-by-role and risk-band views, plus a scenario explorer backed by the trained model artifacts.

## Company CSV upload

The dashboard's **Company file analysis** section accepts a CSV with up to 1,000,000 rows (500 MB). It streams the file in 100,000-row batches to keep large analyses manageable. The required columns are:

`work_hours_per_week`, `overtime_hours`, `meetings_per_day`, `deadlines_missed`, `experience_years`, `job_role`, and `work_mode`.

Optional workplace fields include `stress_level`, `work_life_balance`, `manager_support`, `job_satisfaction`, and `social_support_score`. Missing optional values use the established reference baseline; personal lifestyle and health data is not required. Results include predicted risk bands, role comparisons, optional location comparisons, data-coverage feedback, and aggregate risk metrics. Use `company_upload_template.csv` as the starting schema.
