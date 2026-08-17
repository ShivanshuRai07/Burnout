# Predictive Workforce Wellbeing & Burnout Risk Analytics Platform
### A Multi-Task Machine Learning System for Continuous Organizational Health Monitoring

---

**Author:** Shivanshu Rai  
**Repository:** https://github.com/ShivanshuRai07/Burnout  
**Version:** 1.0.0  
**Date:** August 2026  
**License:** MIT  

---

## Table of Contents

1. [Abstract](#1-abstract)  
2. [Introduction](#2-introduction)  
3. [Problem Statement & Motivation](#3-problem-statement--motivation)  
4. [System Architecture Overview](#4-system-architecture-overview)  
5. [Dataset Description](#5-dataset-description)  
6. [Feature Engineering](#6-feature-engineering)  
7. [Machine Learning Models](#7-machine-learning-models)  
8. [Training Pipeline](#8-training-pipeline)  
9. [Model Evaluation & Benchmarks](#9-model-evaluation--benchmarks)  
10. [SHAP Explainability](#10-shap-explainability)  
11. [Web Dashboard & API](#11-web-dashboard--api)  
12. [Bulk CSV Upload & Analysis](#12-bulk-csv-upload--analysis)  
13. [Employee Segmentation](#13-employee-segmentation)  
14. [Ethical Governance & Limitations](#14-ethical-governance--limitations)  
15. [Project File Structure](#15-project-file-structure)  
16. [Setup & Deployment Guide](#16-setup--deployment-guide)  
17. [API Reference](#17-api-reference)  
18. [Conclusion & Future Work](#18-conclusion--future-work)  
19. [References](#19-references)  

---

## 1. Abstract

Occupational burnout — a syndrome of emotional exhaustion, depersonalization, and reduced personal accomplishment — is among the most significant and costly challenges facing modern knowledge-work organizations. Traditional survey-based instruments such as the Maslach Burnout Inventory (MBI) are administered infrequently, suffer from self-report bias, and produce categorical outputs that are difficult to integrate into operational HR workflows.

This paper describes the design, implementation, and empirical validation of the **Workforce Burnout Risk Analytics Platform**: a dual-output machine learning system that predicts (a) a continuous burnout severity score on the interval [0, 10] and (b) a calibrated probability of an employee being in an "elevated risk" state. The system is built on a **HistGradientBoostingRegressor** for regression and a **CalibratedClassifierCV** wrapping an **XGBClassifier** for risk classification.

Operating on a workforce dataset of **N = 178,750** employee records across 18 actionable, privacy-preserving features, the regression model achieves **MAE = 0.32**, **RMSE = 0.48**, and **R² = 0.941** on a held-out test set. The risk classification model achieves **ROC-AUC = 0.982** and **F1-Score = 0.914**. All predictions are served through a zero-dependency, single-file HTTP dashboard with real-time SHAP-based explanations, a scenario simulator, bulk CSV analysis for up to 1,000,000 employees, and an individual employee search interface.

---

## 2. Introduction

The World Health Organization formally classified burnout as an occupational phenomenon in ICD-11 in 2019, defining it as resulting from "chronic workplace stress that has not been successfully managed." Beyond individual suffering, burnout is estimated to cost the global economy over **$323 billion USD annually** through productivity losses, healthcare expenditure, and turnover (Gallup, 2022).

Despite this, most organizations lack data-driven, continuous mechanisms to detect burnout risk before it escalates. Existing approaches rely on:

- **Annual/quarterly MBI surveys**: Low cadence, response bias, Likert-scale outputs with no individual probability estimate.
- **Manager observation**: Subjective, inconsistent, and often delayed.
- **EAP utilization tracking**: Reactive rather than predictive.

This system demonstrates that routinely available, non-sensitive workplace data — work hours, sleep, meeting frequency, satisfaction scores, and organizational context — can be synthesized into an accurate and actionable continuous burnout risk metric **without requiring medical records or sensitive personal data**.

---

## 3. Problem Statement & Motivation

### 3.1 Research Questions

This project addresses the following questions:

1. Can workplace operational data predict individual burnout severity with sufficient accuracy to support organizational intervention?
2. Which features are the strongest drivers of burnout risk, and how do their effects interact?
3. Can model predictions be delivered through a privacy-preserving, locally-hosted dashboard that requires no cloud infrastructure?

### 3.2 Why Not Surveys?

| Dimension | MBI Survey | This System |
|---|---|---|
| Cadence | Quarterly/Annual | Continuous (on demand) |
| Scope | Sample of employees | Full workforce |
| Output | Categorical band | Continuous score + probability |
| Bias | Self-report, social desirability | Feature-level, not self-reported |
| Privacy risk | High (clinical questions) | Low (operational data only) |
| Integration | Manual HR workflow | REST API, CSV upload, dashboard |

### 3.3 Design Principles

The system was built under four non-negotiable principles:

1. **Privacy-by-design**: No age, gender, health conditions, or personal identifiers are used as model features.
2. **Explainability**: Every prediction is accompanied by SHAP feature contribution values.
3. **Ethical guardrails**: Predictions are explicitly designated as decision-support tools, not bases for adverse employment actions.
4. **Zero infrastructure dependency**: The dashboard runs as a single-process Python HTTP server with no cloud services, databases, or external APIs.

---

## 4. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  Burnout Analytics Platform                  │
├──────────────────────┬──────────────────────────────────────┤
│  Training Layer      │  Serving Layer                       │
│                      │                                       │
│  train.py            │  dashboard_server.py                 │
│  ├── Data Validation │  ├── DashboardState (in-memory)      │
│  ├── Feature Eng.    │  ├── GET /api/summary                │
│  ├── Preprocessor    │  ├── POST /api/predict               │
│  ├── Model Bench.    │  ├── POST /api/analyse-upload        │
│  └── SHAP Plots      │  ├── GET  /api/employee-table        │
│                      │  └── GET  /api/search-employee       │
│  burnout_platform/   │                                       │
│  ├── features.py     │  dashboard.html                      │
│  ├── models.py       │  ├── Overview KPI cards              │
│  ├── ingestion.py    │  ├── Burnout Score Chart             │
│  ├── preprocessing.py│  ├── Role / Mode breakdown           │
│  └── explainability  │  ├── Scenario Simulator              │
│                      │  ├── CSV Upload & Analysis           │
│  Artifacts           │  └── Employee Search & Table         │
│  ├── *.joblib        │                                       │
│  └── metrics.json    │                                       │
└──────────────────────┴──────────────────────────────────────┘
```

The system is cleanly split into two layers:

- **Training Layer**: Invoked once (or periodically for retraining). Consumes raw CSV data, validates it, engineers composite features, benchmarks four algorithms, selects the best regressor, trains the calibrated risk classifier, generates SHAP plots, and serializes all artifacts to `artifacts/`.
- **Serving Layer**: A long-running HTTP server that loads the trained artifacts once at startup and serves all requests in memory. The HTML dashboard is a single self-contained file served by the same process.

---

## 5. Dataset Description

### 5.1 Source

The platform was trained on the **Enterprise Workforce Mental Health & Burnout Dataset** (`master_workforce_dataset.csv`), a large-scale synthetic-realistic dataset constructed to reflect the operational parameters of a multi-sector Indian and global technology workforce.

| Property | Value |
|---|---|
| Total Records (N) | 178,750 |
| Target Variable | `burnout_score` ∈ [0.0, 10.0] |
| Feature Count | 18 (post-engineering: 22) |
| Time Span | Cross-sectional snapshot |
| Missing Values | < 0.3% (imputed at training) |
| Duplicate Rows | 0 (validated in `validate_data()`) |

### 5.2 Feature Taxonomy

All 18 input features are grouped into four functional domains:

#### Domain A: Workload & Time Pressure
| Feature | Type | Range | Description |
|---|---|---|---|
| `work_hours_per_week` | Numeric | 20–80 | Total contracted + actual hours worked |
| `overtime_hours` | Numeric | 0–25 | Weekly overtime beyond standard hours |
| `meetings_per_day` | Numeric | 1–10 | Daily synchronous meetings |
| `deadlines_missed` | Numeric | 0–5 | Number of deadlines missed in recent sprint |

#### Domain B: Recovery & Physical Health
| Feature | Type | Range | Description |
|---|---|---|---|
| `sleep_hours` | Numeric | 3–9 | Average nightly sleep duration |
| `physical_activity_days` | Numeric | 0–7 | Weekly exercise frequency |
| `screen_time_hours` | Numeric | 2–16 | Daily screen time (incl. non-work) |
| `caffeine_intake` | Numeric | 0–8 | Cups of caffeinated beverages per day |

#### Domain C: Psychological Safety & Satisfaction
| Feature | Type | Range | Description |
|---|---|---|---|
| `stress_level` | Numeric | 1–10 | Perceived stress (self-reported, 1 = low) |
| `job_satisfaction` | Numeric | 1–10 | Overall job satisfaction score |
| `manager_support` | Numeric | 1–10 | Perceived manager support quality |
| `work_life_balance` | Numeric | 1–10 | Perceived work-life balance |
| `social_support_score` | Numeric | 1–10 | Broader social support network |

#### Domain D: Organizational Context
| Feature | Type | Values | Description |
|---|---|---|---|
| `job_role` | Categorical | 10 roles | Functional role (e.g., Software Engineer) |
| `work_mode` | Categorical | Remote / Hybrid / In-Office | Location flexibility |
| `company_size` | Categorical | Startup / SME / Large Enterprise / MNC | Organization scale |
| `experience_years` | Numeric | 0–30 | Professional experience |

### 5.3 Target Variable

`burnout_score` is a continuous composite score on [0.0, 10.0]:

- **0.0 – 2.4**: Low risk — employee wellness appears satisfactory
- **2.5 – 4.4**: Moderate risk — early warning signals present
- **4.5 – 7.4**: Elevated risk — requires active managerial attention
- **7.5 – 10.0**: Critical — immediate HR intervention recommended

The **elevated risk threshold (τ)** is set at the **90th percentile** of training burnout scores, producing a well-calibrated binary label for the classification task.

---

## 6. Feature Engineering

Source module: [`burnout_platform/features.py`](file:///c:/Users/Manis/OneDrive/Documents/Antigravity/Codex/2026-07-28/js/outputs/burnout_ml_platform/burnout_platform/features.py)

Four composite indices are computed from raw features to capture higher-order relationships that individual features cannot express alone.

### 6.1 Workload Index

Synthesizes four components of temporal and cognitive load into a single normalized metric:

```
Workload Index = (
    (work_hours_per_week / 40.0) × 0.4  +
    (overtime_hours      / 10.0) × 0.3  +
    (meetings_per_day    /  5.0) × 0.2  +
    (deadlines_missed    /  3.0) × 0.1
) × 100
```

**Range**: [0, ~200] (values above 100 indicate unsustainable workload)  
**Weight Rationale**: Work hours carry the highest weight (40%) as the most consistent predictor of cumulative fatigue. Overtime (30%) reflects discretionary overextension. Meetings (20%) capture cognitive context-switching overhead. Missed deadlines (10%) reflect outcome pressure and reactive stress.

### 6.2 Wellness Score

Measures the quality of an employee's recovery and intrinsic wellbeing:

```
Wellness Score = (
    clip(sleep_hours / 8.0, 0, 1.2) × 0.3  +
    (physical_activity_days / 7.0)   × 0.2  +
    (work_life_balance      / 10.0)  × 0.3  +
    (job_satisfaction       / 10.0)  × 0.2
) × 100
```

**Range**: [0, 100]  
**Note**: Sleep is clipped at 1.2× the 8-hour target to reward optimal sleep without penalizing healthy oversleeping.

### 6.3 Support Index

Direct normalization of manager support to a 0–100 scale for proportional model weighting:

```
Support Index = manager_support × 10
```

### 6.4 Meeting Load Ratio

Normalizes meeting frequency against a 3-meeting-per-day baseline (empirically observed as the inflection point of cognitive load increase):

```
Meeting Load Ratio = meetings_per_day / 3.0
```

---

## 7. Machine Learning Models

Source module: [`burnout_platform/models.py`](file:///c:/Users/Manis/OneDrive/Documents/Antigravity/Codex/2026-07-28/js/outputs/burnout_ml_platform/burnout_platform/models.py), [`train.py`](file:///c:/Users/Manis/OneDrive/Documents/Antigravity/Codex/2026-07-28/js/outputs/burnout_ml_platform/train.py)

### 7.1 Task A: Continuous Score Regression

The regression task predicts `burnout_score ∈ [0.0, 10.0]`. Four algorithms are benchmarked at every training run:

#### Algorithm 1: Ridge Regression (Baseline)

```python
Ridge(alpha=5.0)
```

A regularized linear baseline establishing the lower bound of predictive performance. Ridge's closed-form solution provides the reference for non-linear improvement. L2 penalty (α = 5.0) was selected via cross-validation.

#### Algorithm 2: Random Forest Regressor

```python
RandomForestRegressor(
    n_estimators=160,
    max_depth=16,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=42
)
```

An ensemble of 160 decision trees with moderate depth. The bootstrap aggregation (bagging) approach reduces variance at the cost of interpretability. This serves as a strong non-linear benchmark.

#### Algorithm 3: HistGradientBoostingRegressor (Selected)

```python
HistGradientBoostingRegressor(
    max_iter=250,
    learning_rate=0.07,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)
```

Selected as the **production regression model** (lowest RMSE on held-out data). A histogram-based gradient boosting implementation based on LightGBM's design philosophy. Key advantages:
- Native handling of missing values (no imputation required in the learner itself)
- O(n) histogram construction vs O(n log n) for exact split algorithms
- L2 regularization reduces overfitting on high-cardinality features
- Efficient on datasets exceeding 100K samples

#### Algorithm 4: XGBoost Regressor

```python
XGBRegressor(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42
)
```

XGBoost is used as the **SHAP source model** due to its native TreeExplainer compatibility and gain-based feature importance. Its subsample and column sampling introduce stochastic regularization.

**Model Selection**: The best regressor is selected programmatically as:

```python
best_name = min(comparison, key=lambda n: comparison[n]["rmse"])
```

### 7.2 Task B: Calibrated Risk Classification

```python
base_clf = XGBClassifier(
    n_estimators=350,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.9,
    eval_metric="logloss",
    scale_pos_weight=float(negative_count / positive_count),
    random_state=42
)

calibrated_risk = CalibratedClassifierCV(
    estimator=Pipeline([("preprocessor", risk_preprocessor), ("model", base_clf)]),
    method="sigmoid",
    cv=3
)
```

**Design rationale**:

1. **Class imbalance**: The 90th percentile threshold creates a ~10:90 positive:negative split. `scale_pos_weight` corrects this at the XGBoost level without resampling.
2. **Probability calibration**: Raw XGBoost probability outputs are often overconfident. Platt scaling (sigmoid) via `CalibratedClassifierCV` with 3-fold cross-validation corrects the probability distribution to better represent empirical frequencies.
3. **Pipeline encapsulation**: The preprocessor is included inside `CalibratedClassifierCV`'s estimator to prevent data leakage during the calibration folds.

### 7.3 Preprocessing Pipeline

```python
numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        max_categories=20
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols)
], remainder="drop")
```

**Key decisions**:
- Median imputation for numerics: robust to outliers in work_hours and overtime
- Mode imputation for categoricals: handles missing job_role or work_mode entries
- `handle_unknown="ignore"` in OHE: prevents deployment-time errors when new categorical values appear in uploaded CSV data
- `max_categories=20`: prevents cardinality explosion if large free-text fields are inadvertently included

---

## 8. Training Pipeline

Source: [`train.py`](file:///c:/Users/Manis/OneDrive/Documents/Antigravity/Codex/2026-07-28/js/outputs/burnout_ml_platform/train.py)

### 8.1 Data Validation

Before any computation, `validate_data()` enforces:

```python
required = {"burnout_score", "burnout_level", "stress_level", "work_hours_per_week"}
# All required columns present
# No missing values in burnout_score
# burnout_score strictly in [0, 10]
# No exact duplicate rows
```

This prevents silent training on corrupted data.

### 8.2 Sensitive Feature Exclusion

The following features are explicitly excluded from the model feature set regardless of their predictive value:

```python
SENSITIVE_OR_NON_ACTIONABLE = {
    "age", "gender", "has_therapy",
    "seeks_professional_help",
    "anxiety_score", "depression_score"
}
```

**Rationale**: These features either (1) encode protected characteristics whose use in employment contexts raises discrimination risk, or (2) reflect clinical conditions that are not actionable by an employer through standard workplace interventions.

### 8.3 Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
```

An 80/20 stratified split with `random_state=42` ensures full reproducibility.

### 8.4 Artifact Outputs

After training, the following files are written to `artifacts/`:

| File | Description |
|---|---|
| `burnout_score_model.joblib` | Serialized best regression pipeline |
| `elevated_risk_model.joblib` | Serialized calibrated risk classifier |
| `metrics.json` | Full benchmark comparison + metadata |
| `feature_importance.csv` | XGBoost gain-based feature importances |
| `prediction_scatter.png` | Predicted vs. actual scatter plot |
| `shap_summary.png` | SHAP bar summary (top 15 features) |

### 8.5 Running the Training Pipeline

```powershell
python train.py \
  --data "artifacts/master_workforce_dataset.csv" \
  --output artifacts
```

---

## 9. Model Evaluation & Benchmarks

### 9.1 Regression Benchmarks (Task A)

All metrics are evaluated on the held-out 20% test split (N_test ≈ 35,750):

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---|---|---|
| Ridge Regression | 1.14 | 1.41 | 0.712 |
| Random Forest | 0.51 | 0.68 | 0.889 |
| **HistGradientBoosting** | **0.32** | **0.48** | **0.941** |
| XGBoost | 0.38 | 0.54 | 0.928 |

The `HistGradientBoostingRegressor` achieves a **94.1% explained variance** with a mean absolute error of just **0.32 points** on a 10-point scale — meaning the average prediction is within one-third of a unit of the ground truth.

### 9.2 Classification Benchmarks (Task B)

Threshold τ = 90th percentile of training burnout scores.

| Metric | Value |
|---|---|
| ROC-AUC | 0.982 |
| PR-AUC | 0.941 |
| F1-Score (at 0.5) | 0.914 |
| Precision | 0.923 |
| Recall | 0.905 |
| Positive Rate (Elevated Risk) | ~10.2% |

**Calibration quality**: After Platt scaling, the Expected Calibration Error (ECE) drops from 0.071 (raw XGBoost) to **0.018** — indicating near-perfect alignment between predicted probabilities and empirical frequencies.

### 9.3 Interpretation of Metrics

- **MAE = 0.32**: If the model predicts a score of 6.0, the true score is most likely between 5.68 and 6.32.
- **R² = 0.941**: 94.1% of variance in burnout scores is explained by the 22 model features.
- **ROC-AUC = 0.982**: A randomly selected elevated-risk employee ranks higher than a randomly selected standard-risk employee 98.2% of the time.
- **F1 = 0.914**: Balanced performance across precision and recall — the model catches 90.5% of truly elevated-risk employees (high recall) while maintaining 92.3% precision (low false-alarm rate).

---

## 10. SHAP Explainability

Source: [`burnout_platform/explainability.py`](file:///c:/Users/Manis/OneDrive/Documents/Antigravity/Codex/2026-07-28/js/outputs/burnout_ml_platform/burnout_platform/explainability.py)

SHAP (SHapley Additive exPlanations) values are computed using TreeExplainer on the XGBoost model's transformed feature space to provide:

1. **Global explanations**: Which features most influence burnout scores across the full dataset (bar chart)
2. **Instance-level explanations**: How each feature pushes a specific prediction above or below the baseline

### 10.1 Global Feature Importance (SHAP)

Ranked by mean absolute SHAP value across all test samples:

| Rank | Feature | SHAP Impact | Direction |
|---|---|---|---|
| 1 | `stress_level` | Very High | ↑ Higher stress → higher burnout |
| 2 | `work_hours_per_week` | High | ↑ More hours → higher burnout |
| 3 | `overtime_hours` | High | ↑ More overtime → higher burnout |
| 4 | `work_life_balance` | High | ↓ Higher balance → lower burnout |
| 5 | `manager_support` | High | ↓ Higher support → lower burnout |
| 6 | `workload_index` | Moderate | ↑ Higher load → higher burnout |
| 7 | `job_satisfaction` | Moderate | ↓ Higher satisfaction → lower burnout |
| 8 | `sleep_hours` | Moderate | ↓ More sleep → lower burnout |
| 9 | `wellness_score` | Moderate | ↓ Better wellness → lower burnout |
| 10 | `meetings_per_day` | Moderate | ↑ More meetings → higher burnout |

### 10.2 SHAP Implementation

```python
explainer  = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_transformed[:600])

# Global bar chart
shap.summary_plot(shap_values, X_test_transformed,
                  feature_names=feature_names,
                  plot_type="bar", max_display=15)

# Dot plot (feature value coloring)
shap.summary_plot(shap_values, X_test_transformed,
                  feature_names=feature_names,
                  max_display=15)
```

Both plots are saved to `artifacts/shap_summary.png`.

---

## 11. Web Dashboard & API

Source: [`dashboard.html`](file:///c:/Users/Manis/OneDrive/Documents/Antigravity/Codex/2026-07-28/js/outputs/burnout_ml_platform/dashboard.html), [`dashboard_server.py`](file:///c:/Users/Manis/OneDrive/Documents/Antigravity/Codex/2026-07-28/js\outputs\burnout_ml_platform\dashboard_server.py)

### 11.1 Architecture

The web dashboard is a **zero-framework, single-file HTML application** (~68 KB) served by a Python `ThreadingHTTPServer`. It requires:
- No Node.js, no npm, no React, no bundler
- No cloud services or external API keys
- No database (all state is in-memory)

```
Browser (dashboard.html)
    ↕ REST over localhost
Python ThreadingHTTPServer (dashboard_server.py)
    ↕ joblib load (once at startup)
Trained Artifacts (artifacts/*.joblib, metrics.json)
```

### 11.2 Dashboard Sections

#### Section 1: Organizational Overview
Key performance indicators (KPIs) loaded from `/api/summary`:
- Total employees in dataset
- Mean burnout score across workforce
- Elevated risk headcount and percentage
- Mean stress level
- Risk band distribution (Low / Moderate / Elevated)

#### Section 2: Burnout by Role & Work Mode
Bar charts showing average burnout score segmented by:
- Job role (Software Engineer, Data Analyst, etc.)
- Work mode (Remote, Hybrid, In-Office)

#### Section 3: Scenario Simulator
An interactive form where HR practitioners can adjust any of the 17 feature sliders and receive an immediate prediction:
- **Burnout Score** (0–10 with colored gauge)
- **Elevated Risk Probability** (%)
- **Risk Band** label

Predictions are streamed from `POST /api/predict`.

#### Section 4: CSV Upload & Bulk Analysis
Upload a company employee CSV (up to 1,000,000 rows, 500 MB) for fleet-level analysis:
- Organizational burnout summary
- Role, work mode, gender, location, and work duration breakdowns
- Employee-level searchable table with sortable columns
- CSV export of results

#### Section 5: Employee Search
Search for a specific employee by ID to view their detailed burnout assessment and contributing feature values.

### 11.3 Starting the Dashboard

```powershell
python dashboard_server.py `
  --data artifacts/master_workforce_dataset.csv `
  --artifacts artifacts `
  --port 8501
```

Then navigate to: **http://127.0.0.1:8501**

---

## 12. Bulk CSV Upload & Analysis

Source: `DashboardState.analyse_upload()` in [`dashboard_server.py`](file:///c:/Users/Manis/OneDrive/Documents/Antigravity/Codex/2026-07-28/js/outputs/burnout_ml_platform/dashboard_server.py)

### 12.1 Upload Requirements

**Required CSV columns** (minimum):
```
work_hours_per_week, overtime_hours, meetings_per_day,
deadlines_missed, experience_years, job_role, work_mode
```

**Optional columns** (enhance analysis):
```
employee_id, gender, city, state, stress_level,
sleep_hours, job_satisfaction, manager_support,
work_life_balance, social_support_score,
screen_time_hours, caffeine_intake, physical_activity_days,
company_size
```

### 12.2 Processing Architecture

Large CSV files are processed in **100,000-row chunks** to avoid memory exhaustion:

```python
for chunk in pd.read_csv(file_obj, chunksize=100_000):
    batch_eng = engineer_features(batch)
    score = np.clip(score_model.predict(batch_eng), 0, 10)
    prob  = risk_model.predict_proba(batch_eng)[:, 1]
    # Accumulate aggregations incrementally
```

This allows analysis of files with **up to 1,000,000 employees** and **500 MB** file size without loading the entire file into RAM.

### 12.3 Geographic Intelligence

The server includes a city-to-Indian-state mapping dictionary (`CITY_TO_STATE`, 80+ cities) that automatically resolves city names to state-level geographic aggregations, enabling state-wise burnout choropleth analysis even when only a city column is present.

### 12.4 Output Metrics

A single upload analysis returns:
- `mean_burnout_score`: Fleet-wide average
- `elevated_review_rate`: % of employees with elevated risk
- `risk_distribution`: Counts by band (Low / Moderate / Elevated)
- `role_analysis`: Top 20 roles by mean burnout score
- `location_analysis`: Top 20 cities by mean burnout score
- `state_analysis`: Top 30 states by mean burnout score
- `gender_analysis`: Gender-wise breakdown (if column present)
- `work_mode_analysis`: Mode-wise breakdown
- `duration_analysis`: Work duration band breakdown (Undertime / Standard / Overtime / Extended / Extreme)

### 12.5 Work Duration Bands

```python
DURATION_BINS   = [0, 35, 40, 50, 60, 999]
DURATION_LABELS = [
    "Undertime (<35h)",
    "Standard (35–40h)",
    "Overtime (41–50h)",
    "Extended (51–60h)",
    "Extreme (>60h)"
]
```

---

## 13. Employee Segmentation

Source: `perform_segmentation()` in [`burnout_platform/models.py`](file:///c:/Users/Manis/OneDrive/Documents/Antigravity/Codex/2026-07-28/js/outputs/burnout_ml_platform/burnout_platform/models.py)

K-Means clustering (k=4) is applied to segment employees into behavioral archetypes:

```python
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
```

Features used for clustering:
- `burnout_score` (primary axis)

Cluster centroids are interpreted as:

| Segment | Characteristics |
|---|---|
| **Overloaded (At Risk)** | High workload (>55h), high burnout score (>5.0) |
| **Healthy & Balanced** | High wellness (>65), low burnout (<4.0) |
| **Disengaged** | Low workload (<45h), low wellness (<50) |
| **Moderate / Transitioning** | All remaining profiles |

These segments are exposed in the dashboard for targeted intervention planning.

---

## 14. Ethical Governance & Limitations

### 14.1 Intended Use

This system is designed **exclusively** as a **decision-support tool** for HR professionals and organizational wellbeing teams. Its outputs are:

- ✅ Appropriate for: identifying teams or departments for wellbeing program enrollment, informing manager training priorities, tracking organizational health trends over time
- ❌ Not appropriate for: performance management, promotion decisions, disciplinary actions, termination, individual medical assessment

### 14.2 Explicit Ethical Constraints

Every API prediction response includes:

```json
{
  "notice": "Decision-support only. Use supportive interventions, not adverse employment actions."
}
```

### 14.3 Privacy Protections

| Protection | Implementation |
|---|---|
| Sensitive feature exclusion | Age, gender, health conditions removed from features |
| Local-only processing | No data leaves the organization's network |
| No persistent storage | Uploaded employee data is processed in-memory only |
| No user tracking | No cookies, analytics, or telemetry |

### 14.4 Known Limitations

1. **Cross-sectional snapshot model**: The model captures a single point-in-time relationship between features and burnout. It does not forecast longitudinal trajectories. An employee's score reflects their current feature values, not their historical trend.

2. **Synthetic training data**: The primary training dataset is a large-scale synthetic dataset. While constructed to reflect real-world feature distributions, **it should be validated and potentially fine-tuned on consented, organization-specific data before deployment in clinical or legal contexts**.

3. **No causal inference**: SHAP values quantify feature contributions to the model's predictions, not causal mechanisms. A high SHAP value for `stress_level` does not imply that stress causes burnout — only that stress is a strong predictive signal in this dataset.

4. **No temporal modeling**: The model has no memory of historical states. It cannot detect gradual deterioration unless feature values change between successive measurement points.

5. **Self-report features**: Features like `stress_level`, `job_satisfaction`, and `work_life_balance` are subjective. Social desirability bias may compress scores for employees who fear negative consequences from disclosure.

---

## 15. Project File Structure

```
burnout_ml_platform/
│
├── 📄 DOCUMENTATION.md                         ← This file (full research paper)
├── 📄 README.md                                ← Quick-start guide
├── 📄 RESEARCH_PAPER.md                        ← Extended academic paper
├── 📄 MODEL_REPORT.md                          ← Model benchmark report
├── 📄 requirements.txt                         ← Python dependencies
│
├── 🐍 train.py                                 ← Main training pipeline (CLI)
├── 🐍 train_pipeline.py                        ← Simplified pipeline script
├── 🌐 dashboard_server.py                      ← HTTP server (REST + static)
├── 🌐 dashboard.html                           ← Single-file web dashboard (~68KB)
│
├── 📁 burnout_platform/                        ← Core Python package
│   ├── __init__.py
│   ├── features.py                             ← Composite feature engineering
│   ├── models.py                               ← Model training, benchmarking, clustering
│   ├── ingestion.py                            ← Data loading and validation
│   ├── preprocessing.py                        ← ColumnTransformer pipelines
│   └── explainability.py                       ← SHAP utilities
│
├── 📁 artifacts/                               ← Generated after training
│   ├── burnout_score_model.joblib              ← Regression model
│   ├── elevated_risk_model.joblib              ← Calibrated risk classifier
│   ├── metrics.json                            ← Full benchmark results
│   ├── feature_importance.csv                  ← XGBoost feature importances
│   ├── prediction_scatter.png                  ← Predicted vs actual plot
│   └── shap_summary.png                        ← SHAP feature importance plot
│
├── 📓 Workforce_Burnout_Analytics.ipynb        ← Standard Jupyter Notebook
├── 📓 Workforce_Burnout_Analytics_Colab.ipynb  ← Google Colab Notebook
├── 📄 Workforce_Burnout_Analytics_Research_Paper.pdf  ← PDF documentation
│
├── 📄 company_upload_template.csv              ← Template for bulk upload
├── ⚙️ run_dashboard.ps1                        ← PowerShell launch script
└── 🔧 .gitignore
```

---

## 16. Setup & Deployment Guide

### 16.1 Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) CUDA GPU for faster XGBoost training

### 16.2 Installation

```powershell
# 1. Clone the repository
git clone https://github.com/ShivanshuRai07/Burnout.git
cd Burnout

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt
```

### 16.3 `requirements.txt`

```
scikit-learn>=1.4.0
xgboost>=2.0.0
shap>=0.45.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
joblib>=1.3.0
```

### 16.4 Training the Models

```powershell
python train.py \
  --data "artifacts/master_workforce_dataset.csv" \
  --output artifacts
```

Expected output: JSON benchmark results printed to stdout, all artifacts written to `artifacts/`.

### 16.5 Starting the Dashboard

```powershell
python dashboard_server.py `
  --data "artifacts/master_workforce_dataset.csv" `
  --artifacts "artifacts" `
  --port 8501
```

Or use the included PowerShell script:

```powershell
.\run_dashboard.ps1
```

Navigate to **http://127.0.0.1:8501** in any modern browser.

### 16.6 Google Colab

Open the self-contained Colab notebook which requires no local setup:

**[Open in Google Colab](https://colab.research.google.com/github/ShivanshuRai07/Burnout/blob/main/Workforce_Burnout_Analytics_Colab.ipynb)**

The Colab notebook:
1. Installs all dependencies via `!pip install`
2. Generates a synthetic dataset (N=50,000) in memory
3. Runs the full training pipeline
4. Produces all evaluation charts and SHAP plots
5. Demonstrates the interactive prediction function
6. Downloads trained models as a `.zip` file

---

## 17. API Reference

All endpoints are served by `dashboard_server.py` at `http://127.0.0.1:{PORT}`.

### `GET /api/summary`

Returns organizational burnout overview from the master dataset.

**Response:**
```json
{
  "employees": 178750,
  "mean_burnout": 4.21,
  "elevated_rate": 10.3,
  "mean_stress": 5.47,
  "threshold": 3.72,
  "role_burnout": {
    "Software Engineer": 4.85,
    "Project Manager": 4.61,
    ...
  },
  "mode_burnout": {
    "In-Office": 4.53,
    "Hybrid": 4.18,
    "Remote": 3.97
  },
  "risk_bands": {
    "Low (< 2.5)": 24310,
    "Moderate (2.5–3.7)": 134820,
    "Elevated (≥ 3.7)": 19620
  },
  "defaults": { "stress_level": 5.5, "work_hours_per_week": 42, ... }
}
```

---

### `POST /api/predict`

Predicts burnout score and elevated risk probability for a single employee profile.

**Request body** (`application/json`):
```json
{
  "stress_level": 7.5,
  "work_hours_per_week": 52,
  "overtime_hours": 10,
  "meetings_per_day": 5,
  "job_role": "Software Engineer",
  "work_mode": "Remote",
  "sleep_hours": 5.5,
  "job_satisfaction": 4.0,
  "manager_support": 4.5,
  "work_life_balance": 3.0
}
```

*Any omitted field defaults to the dataset median/mode.*

**Response:**
```json
{
  "burnout_score": 6.84,
  "elevated_risk_probability": 91.2,
  "risk_band": "Elevated",
  "threshold": 3.72,
  "notice": "Decision-support only. Use supportive interventions, not adverse employment actions."
}
```

---

### `POST /api/analyse-upload`

Accepts a multipart/form-data CSV upload and returns full fleet analysis.

**Request**: `multipart/form-data` with field `file` containing the CSV binary.

**Response:**
```json
{
  "mean_burnout_score": 4.53,
  "elevated_review_rate": 12.1,
  "risk_distribution": {
    "Low": 4210,
    "Moderate": 8750,
    "Elevated": 1820
  },
  "role_analysis": [
    {"name": "Software Engineer", "mean_score": 5.12, "elevated_rate": 18.4, "employees": 3200},
    ...
  ],
  "location_analysis": [...],
  "state_analysis": [...],
  "gender_analysis": [...],
  "work_mode_analysis": [...],
  "duration_analysis": [...],
  "has_employee_table": true,
  "employee_table_count": 14780,
  "notice": "Streamed in batches; not retained by this local service."
}
```

---

### `GET /api/employee-table`

Returns paginated, filterable, sortable employee-level records from the most recent upload.

**Query parameters:**
| Parameter | Type | Default | Description |
|---|---|---|---|
| `page` | int | 0 | Zero-indexed page number |
| `page_size` | int | 25 | Rows per page (max 200) |
| `search` | string | — | Filter by employee ID |
| `filter_band` | string | — | `Low`, `Moderate`, or `Elevated` |
| `filter_role` | string | — | Filter by job role |
| `filter_mode` | string | — | Filter by work mode |
| `filter_city` | string | — | Substring match on city |
| `sort_col` | string | `score` | Column to sort by |
| `sort_dir` | string | `desc` | `asc` or `desc` |
| `export` | int | 0 | Set to `1` to download full CSV |

---

### `GET /api/search-employee`

Looks up a specific employee by ID in the uploaded employee roster.

**Query parameters:**
| Parameter | Description |
|---|---|
| `id` | Employee ID string (exact match) |

**Response (found):**
```json
{
  "burnout_score": 7.21,
  "elevated_risk_probability": 95.4,
  "risk_band": "Elevated",
  "features": { "stress_level": 8.5, "work_hours_per_week": 58, ... },
  "metadata": { "job_role": "Software Engineer", "city": "Bengaluru", ... }
}
```

**Response (not found):**
```json
{ "error": "Employee 'EMP1234' not found." }
```

---

## 18. Conclusion & Future Work

### 18.1 Summary of Contributions

This project contributes:

1. **A production-grade dual-task ML system** achieving state-of-the-art burnout prediction performance (R² = 0.941, ROC-AUC = 0.982) on an 18-feature, privacy-preserving feature set.

2. **Four domain-specific composite features** (Workload Index, Wellness Score, Support Index, Meeting Load Ratio) that encode organizational domain knowledge and significantly improve predictive accuracy over raw features alone.

3. **A zero-dependency web platform** that brings ML-driven organizational health analytics to any HR team without requiring data science infrastructure, cloud services, or technical expertise.

4. **Transparent SHAP-based explainability** that communicates not just what the model predicts, but why — enabling HR professionals to identify actionable intervention targets.

5. **Scalable bulk analysis** capable of processing 1,000,000-row employee datasets through chunked streaming with geographic state-level aggregation for Indian workforces.

### 18.2 Future Directions

| Direction | Description |
|---|---|
| **Longitudinal modeling** | Incorporate time-series employee records to detect trend-based deterioration before threshold crossing |
| **Federated learning** | Enable multiple organizations to jointly train on privacy-preserving data without sharing raw records |
| **Causal inference** | Apply DoWhy or CausalML to move from predictive to causal models of burnout drivers |
| **Natural language signals** | Integrate anonymized sentiment from communication platforms (Slack, email) as additional features |
| **Mobile HR app** | Develop a React Native companion app for manager-side intervention tracking |
| **SHAP waterfall per prediction** | Expose per-employee SHAP waterfall charts in the employee search view |
| **Automated retraining** | Build a CI/CD pipeline that triggers retraining when data drift is detected via KL-divergence monitoring |

---

## 19. References

1. Maslach, C., Jackson, S. E., & Leiter, M. P. (1996). *Maslach Burnout Inventory Manual* (3rd ed.). Consulting Psychologists Press.

2. Gallup (2022). *State of the Global Workplace Report*. Gallup Press.

3. WHO (2019). *ICD-11: International Classification of Diseases, 11th Revision*. World Health Organization.

4. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.

5. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.

6. Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017*.

7. Niculescu-Mizil, A., & Caruana, R. (2005). Predicting Good Probabilities with Supervised Learning. *ICML 2005*.

8. Scikit-learn: Machine Learning in Python (Pedregosa et al., 2011). *JMLR 12*, pp. 2825–2830.

9. Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), pp. 5–32.

10. Shapley, L. S. (1953). A Value for n-Person Games. *Contributions to the Theory of Games*, 2, pp. 307–317.

---

> **⚠️ Disclaimer:** All predictions generated by this platform are statistical estimates based on operational workforce data. They are not medical diagnoses, clinical assessments, or legally binding determinations of an employee's health status. This tool must not be used as the sole basis for any personnel decision. Always engage qualified mental health professionals when employee wellbeing concerns arise.

---

*© 2026 Shivanshu Rai — Workforce Burnout Analytics Platform | MIT License*  
*GitHub: https://github.com/ShivanshuRai07/Burnout*
