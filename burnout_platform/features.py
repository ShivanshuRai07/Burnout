"""Feature engineering module for the Burnout Platform.

Computes composite indexes: Workload Index, Wellness Score, Support Index, and Meeting Load.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Workload Index: combined weight of work hours, overtime, meetings, and deadlines missed.
    # We normalize each component, then compute a weighted sum.
    hours_norm = df["work_hours_per_week"] / 40.0
    overtime_norm = df["overtime_hours"] / 10.0
    meetings_norm = df["meetings_per_day"] / 5.0
    deadlines_norm = df["deadlines_missed"] / 3.0
    
    df["workload_index"] = (
        hours_norm * 0.4 +
        overtime_norm * 0.3 +
        meetings_norm * 0.2 +
        deadlines_norm * 0.1
    ) * 100.0
    
    # 2. Wellness Score: combines sleep, physical activity, work life balance, and satisfaction.
    sleep_norm = (df["sleep_hours"] / 8.0).clip(0.0, 1.2)
    activity_norm = df["physical_activity_days"] / 7.0
    balance_norm = df["work_life_balance"] / 100.0
    satisfaction_norm = df["job_satisfaction"] / 100.0
    
    df["wellness_score"] = (
        sleep_norm * 0.3 +
        activity_norm * 0.2 +
        balance_norm * 0.3 +
        satisfaction_norm * 0.2
    ) * 100.0
    
    # 3. Support Index
    # Combines manager support and general social support if available, otherwise scales manager support.
    df["support_index"] = df["manager_support"] # normalized 0-100
    
    # 4. Meeting Load Ratio
    # ratio of meetings to standard 3 meetings per day
    df["meeting_load"] = df["meetings_per_day"] / 3.0
    
    return df
