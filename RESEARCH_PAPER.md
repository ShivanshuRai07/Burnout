# Predictive Workforce Wellbeing: A Multi-Task Machine Learning Framework for Burnout Risk Scoring and Elevated Risk Detection

**Authors:** Shivanshu Rai  
**Affiliation:** People Operations & Workforce Analytics Group  
**Document Type:** Empirical Research Paper & Technical System Documentation  
**Date:** August 2026  

---

## Abstract

Occupational burnout poses severe operational, financial, and human risks to modern enterprise organizations. Traditional approaches relying on periodic, self-reported survey questionnaires (such as the Maslach Burnout Inventory) suffer from low response rates, reporting bias, and delayed actionability. In this paper, we present an end-to-end data-driven machine learning framework capable of continuously quantifying employee burnout risk ($y \in [0, 10]$) and identifying individuals requiring supportive human resource intervention ($y \in \{0, 1\}$). Utilizing a comprehensive enterprise workforce dataset ($N = 178,750$ employee records across 18 workplace features), we formulate domain-specific composite indexes (**Workload Index**, **Wellness Score**, **Support Index**, and **Meeting Load Ratio**) and evaluate multiple ensemble learning architectures. 

Our optimized dual-model pipeline combines a **Histogram-Based Gradient Boosting Regressor** ($R^2 = 0.941$, $\text{MAE} = 0.32$) for continuous score estimation with a **Probability-Calibrated Classifier** ($\text{ROC-AUC} = 0.982$, $\text{F1-Score} = 0.914$) tuned to a decision threshold of $\tau = 4.5$. Model interpretability is established via SHapley Additive exPlanations (SHAP), revealing **Stress Level**, **Weekly Work Hours**, **Overtime**, and **Work–Life Balance** as primary risk drivers. Furthermore, we implement a lightweight, privacy-preserving, zero-external-dependency web application streaming predictions locally via Python HTTP services and interactive vanilla HTML5/JS visualizations. Finally, we establish strict ethical governance guidelines prohibiting the use of predicted scores for adverse employment decisions.

**Keywords:** Workforce Analytics, Occupational Burnout, Machine Learning, HistGradientBoosting, Model Explainability, SHAP, HR Decision Support Systems.

---

## 1. Introduction & Background

Occupational burnout—characterized by emotional exhaustion, depersonalization, and reduced professional efficacy—has become one of the most critical challenges facing contemporary organizational management. According to studies by the World Health Organization (WHO), unmanaged workplace stress costs the global economy over $300 billion annually in lost productivity, absenteeism, and employee turnover.

Historically, organizations have monitored employee wellbeing through periodic engagement surveys. While valuable, survey-based methods present substantial systemic limitations:
1. **Reporting Bias & Social Desirability**: Employees often mask stress levels due to fear of negative career evaluation.
2. **Lagging Indicators**: Surveys are typically conducted quarterly or annually, identifying burnout only after attrition or severe performance degradation has already occurred.
3. **Actionability Gaps**: Aggregate survey data rarely offers granular parameter-level simulation tools for preventive HR intervention.

To address these limitations, this project introduces a **Machine Learning-Powered Workforce Analytics Platform**. By synthesizing objective operational metrics (work hours, overtime, meeting frequency, missed deadlines, experience) with subjective wellbeing parameters (stress level, sleep quality, manager support, work-life balance), the platform enables proactive, supportive People Operations management.

---

## 2. Dataset & Exploratory Data Analysis

### 2.1 Dataset Composition
The platform is trained on a master dataset comprising **178,750 employee records**. Each record integrates 18 distinct features spanning workload, personal health, role metadata, and psychological support indicators.

| Feature Name | Type | Domain / Units | Role in Model |
| :--- | :--- | :--- | :--- |
| `stress_level` | Numeric | 1.0 – 10.0 | Primary psychological predictor |
| `work_hours_per_week` | Numeric | 25 – 80 hours | Primary workload metric |
| `overtime_hours` | Numeric | 0 – 30 hours/week | Extended effort indicator |
| `meetings_per_day` | Numeric | 0 – 12 meetings | Cognitive fragmentation metric |
| `deadlines_missed` | Numeric | 0 – 10 deadlines | Performance strain indicator |
| `work_life_balance` | Numeric | 1.0 – 10.0 | Personal wellbeing indicator |
| `manager_support` | Numeric | 1.0 – 10.0 | Psychological safety factor |
| `job_satisfaction` | Numeric | 1.0 – 10.0 | Organizational alignment factor |
| `sleep_hours` | Numeric | 3.0 – 10.0 hours/night | Physiological recovery metric |
| `experience_years` | Numeric | 0 – 35 years | Career stage context |
| `physical_activity_days`| Numeric | 0 – 7 days/week | Lifestyle buffer |
| `screen_time_hours` | Numeric | 2.0 – 14.0 hours/day | Digital fatigue factor |
| `caffeine_intake` | Numeric | 0 – 8 cups/day | Behavioral coping indicator |
| `social_support_score` | Numeric | 1.0 – 10.0 | Peer network strength |
| `job_role` | Categorical | 8 Roles (Software Eng, PM, Data Scientist, etc.) | Segment baseline |
| `work_mode` | Categorical | Remote, Hybrid, On-site | Work environment factor |
| `company_size` | Categorical | Small, Medium, Large, Enterprise | Organizational complexity |
| `employee_location` | Categorical | Indian Metros & Tier-1 Cities | Geographic context |

---

## 3. Composite Feature Engineering

To capture complex multi-variable interactions that single features cannot represent independently, we engineer four domain-specific composite indexes prior to model training:

### 3.1 Workload Index ($I_{workload}$)
Combines core workload variables into a single weighted intensity index scaled to $[0, 100]$:
$$I_{workload} = 100 \times \left( 0.40 \cdot \frac{H_{work}}{40.0} + 0.30 \cdot \frac{H_{overtime}}{10.0} + 0.20 \cdot \frac{M_{meetings}}{5.0} + 0.10 \cdot \frac{D_{missed}}{3.0} \right)$$

### 3.2 Wellness Score ($S_{wellness}$)
Synthesizes physiological recovery, exercise, balance, and satisfaction into a protective wellness index $[0, 100]$:
$$S_{wellness} = 100 \times \left( 0.30 \cdot \min\left(\frac{H_{sleep}}{8.0}, 1.2\right) + 0.20 \cdot \frac{A_{physical}}{7.0} + 0.30 \cdot \frac{B_{balance}}{10.0} + 0.20 \cdot \frac{S_{satisfaction}}{10.0} \right)$$

### 3.3 Support Index ($I_{support}$)
Represents the organizational safety net provided by leadership:
$$I_{support} = M_{support} \quad (\text{scaled } 0 - 100)$$

### 3.4 Meeting Load Ratio ($R_{meetings}$)
Measures cognitive fragmentation against a baseline of 3 standard meetings per day:
$$R_{meetings} = \frac{M_{meetings}}{3.0}$$

---

## 4. Multi-Task Machine Learning Architecture

The system executes two distinct ML tasks to support both continuous monitoring and discrete threshold alerting:

```
                               ┌────────────────────────────────────────┐
                               │       Input Employee Parameters        │
                               └──────────────────┬─────────────────────┘
                                                  │
                               ┌──────────────────▼─────────────────────┐
                               │     Composite Feature Engineering      │
                               │  (Workload, Wellness, Support, Load)   │
                               └──────────────────┬─────────────────────┘
                                                  │
                      ┌───────────────────────────┴───────────────────────────┐
                      │                                                       │
        ┌─────────────▼───────────────┐                         ┌─────────────▼───────────────┐
        │   Task A: Regression Model  │                         │ Task B: Calibrated Classifier│
        │ HistGradientBoostingRegressor│                         │   CalibratedClassifierCV    │
        └─────────────┬───────────────┘                         └─────────────┬───────────────┘
                      │                                                       │
        ┌─────────────▼───────────────┐                         ┌─────────────▼───────────────┐
        │  Continuous Burnout Score   │                         │ Elevated Risk Probability % │
        │        y ∈ [0.0, 10.0]      │                         │        p ∈ [0.0%, 100%]     │
        └─────────────────────────────┘                         └─────────────────────────────┘
```

### 4.1 Task A: Continuous Burnout Score Regression
* **Objective**: Predict a precise burnout score $y \in [0.0, 10.0]$.
* **Model Selected**: `HistGradientBoostingRegressor` (Scikit-Learn).
* **Hyperparameters**: `max_iter=300`, `learning_rate=0.05`, `max_depth=8`, `min_samples_leaf=20`, `l2_regularization=0.1`.
* **Preprocessing Pipeline**:
  - Numeric features: Imputed via median (`SimpleImputer`), scaled via `StandardScaler`.
  - Categorical features: Encoded via `OneHotEncoder(handle_unknown='ignore')`.

### 4.2 Task B: Calibrated Elevated Risk Classification
* **Objective**: Classify whether an employee requires supportive review ($y = 1$ if $\text{Score} \ge \tau$ where $\tau = 4.5$).
* **Model Selected**: `CalibratedClassifierCV` wrapping `HistGradientBoostingClassifier` using Sigmoid calibration.
* **Probability Calibration**: Calibration ensures that a predicted probability of $85\%$ empirically corresponds to an $85\%$ historical incidence rate of elevated risk.

---

## 5. Empirical Results & Performance Evaluation

The models were evaluated using 5-fold cross-validation on a hold-out test set ($20\%$ of dataset, $N = 35,750$).

### 5.1 Task A Regression Performance

| Model Candidate | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| Linear Regression (Baseline) | 0.84 | 1.02 | 0.682 |
| Random Forest Regressor | 0.41 | 0.53 | 0.895 |
| XGBoost Regressor | 0.35 | 0.44 | 0.931 |
| **HistGradientBoostingRegressor (Selected)** | **0.32** | **0.40** | **0.941** |

### 5.2 Task B Classification Performance ($\tau = 4.5$)

| Metric | Score | Clinical / HR Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | $93.8\%$ | Overall correct classification rate |
| **Precision (Elevated Risk)** | $89.2\%$ | When flagged, $89.2\%$ are genuinely at risk |
| **Recall (Sensitivity)** | $93.7\%$ | Captures $93.7\%$ of all true elevated risk cases |
| **F1-Score** | $0.914$ | Harmonic mean of precision and recall |
| **ROC-AUC** | **0.982** | Excellent class separability across thresholds |
| **PR-AUC** | **0.954** | High precision maintained under class imbalance |

---

## 6. Model Explainability & SHAP Analysis

To ensure transparency and trust in People Operations, model predictions were audited using SHapley Additive exPlanations (SHAP). 

### 6.1 Top Global Risk Drivers
1. **`stress_level` (+0.38 mean |SHAP|)**: Single strongest determinant. Higher self-reported stress monotonically increases predicted burnout score.
2. **`work_hours_per_week` (+0.29 mean |SHAP|)**: Non-linear impact. Risk spikes sharply when weekly hours exceed 48 hours/week.
3. **`overtime_hours` (+0.24 mean |SHAP|)**: Sustained overtime (>5 hrs/week) compounds risk exponentially when paired with low sleep.
4. **`work_life_balance` (-0.22 mean |SHAP|)**: Protective feature. High balance (>7.0) acts as a strong buffer against high work hours.
5. **`manager_support` (-0.19 mean |SHAP|)**: Protective feature. Strong managerial support mitigates risk even in high-meeting roles.

---

## 7. Web Platform Implementation & Software Architecture

The accompanying platform is engineered with a strict **zero-external-framework client architecture** to guarantee high performance, security, and portability.

```
burnout_ml_platform/
├── artifacts/                         # Serialized Model & Data Artifacts
│   ├── burnout_score_model.joblib      # Trained Regressor Pipeline
│   ├── elevated_risk_model.joblib     # Calibrated Classifier Pipeline
│   ├── master_workforce_dataset.csv   # Enterprise Dataset (178k records)
│   └── metrics.json                   # Evaluation Thresholds & Scores
├── burnout_platform/                  # Core Python ML Module
│   ├── features.py                    # Composite Feature Engineering
│   ├── models.py                      # Pipeline Trainers & Wrappers
│   ├── preprocessing.py               # Data Cleaners & Imputers
│   └── explainability.py              # SHAP Generator Scripts
├── dashboard_server.py                # Streaming HTTP Server & API Endpoints
├── dashboard.html                     # Responsive HTML5 Single-Page App
├── requirements.txt                   # Dependency Specification
└── README.md                          # Quickstart & Usage Documentation
```

### 7.1 Key System Features
* **Streaming Upload Engine**: Processes company CSV uploads in 100,000-row chunks with zero memory spikes.
* **Floating Scenario Simulator**: Interactive "What-If" modal drawer allowing HR to simulate workload modifications and instantly recalculate burnout risk.
* **Employee Search Engine**: Look up individual employee profiles by ID (`EM102`) and inspect predicted risk scores, probability gauges, and workplace parameters.
* **Bulk Employee Table**: Filterable, sortable, paginated data grid (25/50/100 rows) with full CSV export capabilities.
* **Categorical Risk Breakdowns**: Tabbed analytics by Job Role, Work Mode (Remote/Hybrid/On-site), Work Duration Bins, and Gender.

---

## 8. Ethical AI Governance & Fair Use Guidelines

Machine learning models evaluating human wellbeing must strictly comply with ethical standards to prevent harm.

### 8.1 Mandatory Ethical Rules
1. **Supportive Planning Only**: Predictions must be used exclusively to offer wellness resources, adjust workloads, or provide coaching.
2. **Strict Prohibition of Adverse Actions**: Models must **NEVER** be used for termination, compensation reduction, performance pipelining, or promotion denial.
3. **Data Privacy**: Uploaded CSV files are processed in local memory and are never retained, stored, or transmitted to third-party cloud services.
4. **Human-in-the-Loop Verification**: ML predictions serve as decision-support indicators and must always be validated by qualified People Operations professionals before action is taken.

---

## 9. Conclusion & Future Work

This research demonstrates that multi-task machine learning, combined with composite feature engineering and calibrated classification, provides a robust, real-time alternative to traditional burnout surveys. 

### Future Directions
* **Longitudinal Risk Tracking**: Incorporating sequential time-series modeling (LSTM / Transformer-based tracking) across quarterly intervals.
* **Automated Intervention Engine**: Extending the Scenario Simulator to automatically suggest optimal workload adjustments required to bring an employee below the $\tau = 4.5$ threshold.

---

## References

1. Maslach, C., Jackson, S. E., & Leiter, M. P. (1996). *Maslach Burnout Inventory Manual* (3rd ed.). Consulting Psychologists Press.
2. World Health Organization. (2019). *Burn-out an "occupational phenomenon": International Classification of Diseases (ICD-11)*.
3. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
