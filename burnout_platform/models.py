"""Model training, evaluation, and clustering module for the Burnout Platform.

Compares Ridge, RandomForest, HistGradientBoosting, and XGBoost.
Runs K-Means employee segmentation.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor, XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

RANDOM_STATE = 42

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ], remainder="drop")

def train_and_compare_models(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> tuple[dict[str, dict[str, float]], dict[str, Pipeline], str]:
    preprocessor = make_preprocessor(X_train)
    
    models = {
        "ridge": Ridge(alpha=5.0),
        "random_forest": RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=150, learning_rate=0.07, max_leaf_nodes=31, random_state=RANDOM_STATE),
        "xgboost": XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.85, objective="reg:squarederror", n_jobs=-1, random_state=RANDOM_STATE)
    }
    
    comparison = {}
    fitted = {}
    
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        
        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        r2 = r2_score(y_test, pred)
        
        comparison[name] = {
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2), 4)
        }
        fitted[name] = pipe
        
    best_name = min(comparison, key=lambda n: comparison[n]["rmse"])
    return comparison, fitted, best_name

def fit_calibrated_risk_model(
    X_train: pd.DataFrame, 
    y_train_binary: pd.Series, 
    preprocessor: ColumnTransformer
) -> Pipeline:
    from sklearn.ensemble import HistGradientBoostingClassifier
    risk_base = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.07, max_leaf_nodes=31, random_state=RANDOM_STATE
    )
    risk_pipeline = Pipeline([("preprocessor", preprocessor), ("model", risk_base)])
    risk_pipeline.fit(X_train, y_train_binary)
    return risk_pipeline

def perform_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """Segment employees based on Workload Index, Wellness Score, and Burnout Score."""
    df = df.copy()
    
    features = ["burnout_score"]
    # Drop rows with NaN targets for clustering, or fill them
    X_cluster = df[features].fillna(df[features].median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Map clusters to meaningful names based on centroids
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Find cluster labels
    # We want:
    # - Overloaded / High Risk: High workload, low wellness, high burnout
    # - Healthy / Balanced: Low workload, high wellness, low burnout
    # - Disengaged: Low workload, low wellness, low burnout
    # - High Performers: High workload, high wellness, moderate/low burnout
    
    cluster_names = {}
    for i, center in enumerate(centroids):
        wl, wel, bo = center
        if wl > 55 and bo > 5.0:
            cluster_names[i] = "Overloaded (At Risk)"
        elif wel > 65 and bo < 4.0:
            cluster_names[i] = "Healthy & Balanced"
        elif wl < 45 and wel < 50:
            cluster_names[i] = "Disengaged"
        else:
            cluster_names[i] = "Moderate / Transitioning"
            
    # Guarantee unique tags
    df["employee_segment"] = [cluster_names.get(c, f"Segment {c}") for c in clusters]
    return df
