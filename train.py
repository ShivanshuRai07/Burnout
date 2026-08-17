"""Train reproducible burnout-score and elevated-risk models.

Run with:
  python train.py --data "C:\\path\\tech_mental_health_burnout.csv" --output artifacts
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

RANDOM_STATE = 42
TARGET = "burnout_score"
RISK_QUANTILE = 0.90
# Excluded from default deployment features: personal demographics and direct health/treatment signals.
SENSITIVE_OR_NON_ACTIONABLE = {
    "age", "gender", "has_therapy", "seeks_professional_help", "anxiety_score", "depression_score",
}


def validate_data(df: pd.DataFrame) -> None:
    required = {TARGET, "burnout_level", "stress_level", "work_hours_per_week"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns absent: {sorted(missing)}")
    if df.empty or df[TARGET].isna().any():
        raise ValueError("Dataset is empty or has missing burnout_score values.")
    if not df[TARGET].between(0, 10).all():
        raise ValueError("burnout_score is outside the expected 0–10 range.")
    if df.duplicated().any():
        raise ValueError("Exact duplicate rows found; resolve source quality before training.")


def make_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)),
    ])
    return ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)], remainder="drop"), numeric, categorical


def regression_metrics(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    return {"mae": round(float(mean_absolute_error(y_true, pred)), 4),
            "rmse": round(float(mean_squared_error(y_true, pred) ** 0.5), 4),
            "r2": round(float(r2_score(y_true, pred)), 4)}


def classification_metrics(y_true: pd.Series, prob: np.ndarray) -> dict[str, float]:
    label = (prob >= 0.5).astype(int)
    return {"roc_auc": round(float(roc_auc_score(y_true, prob)), 4),
            "pr_auc": round(float(average_precision_score(y_true, prob)), 4),
            "f1_at_0_5": round(float(f1_score(y_true, label, zero_division=0)), 4),
            "positive_rate": round(float(y_true.mean()), 4)}


def save_plots(y_test: pd.Series, pred: np.ndarray, transformed_names: np.ndarray, model, X_test_t: np.ndarray, output: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred, s=5, alpha=0.15)
    plt.plot([0, 10], [0, 10], "r--", linewidth=1)
    plt.xlabel("Observed burnout score")
    plt.ylabel("Predicted burnout score")
    plt.title("Held-out predictions")
    plt.tight_layout()
    plt.savefig(output / "prediction_scatter.png", dpi=180)
    plt.close()

    sample_n = min(600, len(X_test_t))
    sample = X_test_t[:sample_n]
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    shap.summary_plot(values, sample, feature_names=transformed_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(output / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()


def main(data_path: str, output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    # Drop columns that are not available in the provided dataset
    df = df.drop(columns=[col for col in ['wellness_score', 'support_index', 'workload_index', 'meeting_load'] if col in df.columns])
    validate_data(df)

    # Regression estimates continuous burnout.  The provided categorical level is excluded to prevent target leakage.
    feature_columns = sorted(set(df.columns) - {TARGET, "burnout_level"} - SENSITIVE_OR_NON_ACTIONABLE)
    X, y = df[feature_columns], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
    preprocessor, _, _ = make_preprocessor(X_train)

    models = {
        "ridge": Ridge(alpha=5.0),
        "random_forest": RandomForestRegressor(n_estimators=160, max_depth=16, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.07, max_leaf_nodes=31, l2_regularization=1.0, random_state=RANDOM_STATE),
        "xgboost": XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.05, subsample=0.85, colsample_bytree=0.9, objective="reg:squarederror", n_jobs=-1, random_state=RANDOM_STATE),
    }
    comparison, fitted = {}, {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        comparison[name] = regression_metrics(y_test, pipe.predict(X_test))
        fitted[name] = pipe
    best_name = min(comparison, key=lambda n: comparison[n]["rmse"])
    best_regressor = fitted[best_name]

    # Elevated risk is a transparent source-specific definition, not a clinical diagnosis.
    threshold = float(y_train.quantile(RISK_QUANTILE))
    risk_train, risk_test = (y_train >= threshold).astype(int), (y_test >= threshold).astype(int)
    risk_preprocessor, _, _ = make_preprocessor(X_train)
    risk_base = XGBClassifier(n_estimators=350, max_depth=6, learning_rate=0.05, subsample=0.85,
                              colsample_bytree=0.9, eval_metric="logloss", n_jobs=-1,
                              scale_pos_weight=float((risk_train == 0).sum() / max((risk_train == 1).sum(), 1)),
                              random_state=RANDOM_STATE)
    risk_pipeline = Pipeline([("preprocessor", risk_preprocessor), ("model", risk_base)])
    calibrated_risk = CalibratedClassifierCV(risk_pipeline, method="sigmoid", cv=3)
    calibrated_risk.fit(X_train, risk_train)
    risk_prob = calibrated_risk.predict_proba(X_test)[:, 1]

    # Export transparent transformed-feature importance and SHAP for the selected tree regressor.
    chosen_xgb = fitted["xgboost"]
    X_test_t = chosen_xgb.named_steps["preprocessor"].transform(X_test)
    names = chosen_xgb.named_steps["preprocessor"].get_feature_names_out()
    importance = pd.DataFrame({"feature": names, "gain_importance": chosen_xgb.named_steps["model"].feature_importances_}).sort_values("gain_importance", ascending=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    save_plots(y_test, best_regressor.predict(X_test), names, chosen_xgb.named_steps["model"], X_test_t, output)

    metrics = {"dataset": {"rows": int(len(df)), "features_used": feature_columns, "excluded_sensitive_or_non_actionable": sorted(SENSITIVE_OR_NON_ACTIONABLE), "test_rows": int(len(X_test))},
               "regression_target": {"name": TARGET, "range": [0, 10], "models": comparison, "recommended_model": best_name},
               "risk_model": {"definition": f"burnout_score >= training {int(RISK_QUANTILE * 100)}th percentile", "threshold": round(threshold, 4), "metrics": classification_metrics(risk_test, risk_prob)},
               "limitations": ["External synthetic/proxy data; validate on consented company data before deployment.", "Risk band is statistical, not clinical; do not use for adverse employment decisions.", "No time field: this is a snapshot model, not a longitudinal forecast."]}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(best_regressor, output / "burnout_score_model.joblib")
    joblib.dump(calibrated_risk, output / "elevated_risk_model.joblib")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(Path(args.output).resolve() / ".mpl"))
    main(args.data, args.output)
