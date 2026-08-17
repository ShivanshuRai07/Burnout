"""Main execution pipeline to run the enterprise Workforce Burnout model training and clustering.

Usage:
  python train_pipeline.py --data "C:\\Users\\Manis\\Downloads\\DataSets\\DataSets" --output artifacts
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from burnout_platform.ingestion import load_all_sources
from burnout_platform.preprocessing import preprocess_data
from burnout_platform.features import engineer_features
from burnout_platform.models import (
    train_and_compare_models,
    fit_calibrated_risk_model,
    perform_segmentation
)
from burnout_platform.explainability import generate_shap_explanations

TARGET = "burnout_score"
RISK_QUANTILE = 0.90
SENSITIVE_OR_NON_ACTIONABLE = {
    "employee_id", "source_dataset"
}

def main(data_dir: str, output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading raw datasets from: {data_dir}")
    raw_df = load_all_sources(data_dir)
    print(f"Ingested {len(raw_df)} records across datasets.")
    
    print("Preprocessing data and applying group-aware median imputation...")
    preprocessed_df = preprocess_data(raw_df)
    
    print("Engineering features (Workload Index, Wellness Score, Support Index)...")
    feature_df = engineer_features(preprocessed_df)
    
    print("Performing employee K-Means clustering segmentation...")
    segmented_df = perform_segmentation(feature_df)
    
    # Save the master unified dataset for dashboard analytics
    segmented_df.to_csv(output / "master_workforce_dataset.csv", index=False)
    print(f"Saved master unified dataset of size {len(segmented_df)} rows.")
    
    # Prepare features for ML modeling.
    # Exclude target-derived features or high-missingness variables from prediction features.
    exclude_cols = {
        TARGET, "employee_segment", "source_dataset", "employee_id", 
        "mental_fatigue_score", "stress_level"
    }
    # Exclude engineered columns that are not available or desired
    engineered_to_drop = {"workload_index", "wellness_score", "support_index", "meeting_load"}
    feature_columns = sorted(
        set(segmented_df.columns)
        - exclude_cols
        - engineered_to_drop
        - {c for c in segmented_df.columns if c.endswith("_is_missing")}
    )
    
    # We train models on rows that have ground truth burnout_score values.
    labeled_df = segmented_df[segmented_df[TARGET].notna()]
    print(f"Training models using {len(labeled_df)} labeled records.")
    
    X = labeled_df[feature_columns]
    y = labeled_df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    print("Comparing models (Ridge, RandomForest, HistGradientBoosting, XGBoost)...")
    comparison_metrics, fitted_models, best_name = train_and_compare_models(X_train, y_train, X_test, y_test)
    
    print(f"Best Regressor: {best_name} (RMSE: {comparison_metrics[best_name]['rmse']})")
    best_pipeline = fitted_models[best_name]
    
    # Fit risk classification
    threshold = float(y_train.quantile(RISK_QUANTILE))
    y_train_binary = (y_train >= threshold).astype(int)
    y_test_binary = (y_test >= threshold).astype(int)
    
    print(f"Fitting calibrated risk model with threshold={threshold:.4f}...")
    calibrated_risk = fit_calibrated_risk_model(X_train, y_train_binary, best_pipeline.named_steps["preprocessor"])
    
    # Save best models
    joblib.dump(best_pipeline, output / "burnout_score_model.joblib")
    joblib.dump(calibrated_risk, output / "elevated_risk_model.joblib")
    print("Saved pipeline model artifacts.")
    
    # Run SHAP Explainability on the chosen model (XGBoost or random_forest)
    try:
        model_pipeline = fitted_models["xgboost"]
        X_test_trans = model_pipeline.named_steps["preprocessor"].transform(X_test)
        feature_names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
        generate_shap_explanations(model_pipeline.named_steps["model"], X_test_trans, feature_names, output)
        print("Generated SHAP explainability plots.")
    except Exception as e:
        print(f"Skipped SHAP explainability generation: {e}")
        
    # Write metrics summary
    metrics_summary = {
        "dataset": {
            "total_rows": int(len(segmented_df)),
            "labeled_rows": int(len(labeled_df)),
            "features_used": feature_columns
        },
        "regression_models": comparison_metrics,
        "recommended_model": best_name,
        "risk_model": {
            "threshold": threshold,
            "risk_quantile": RISK_QUANTILE
        }
    }
    
    (output / "metrics.json").write_text(json.dumps(metrics_summary, indent=2))
    print("Saved metrics summary file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to DataSets directory")
    parser.add_argument("--output", required=True, help="Path to artifacts output directory")
    args = parser.parse_args()
    main(args.data, args.output)
