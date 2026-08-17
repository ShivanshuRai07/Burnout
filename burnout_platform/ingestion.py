"""Ingestion module for the Burnout Platform.

Handles loading of multiple datasets from different sources and mapping them
to a raw unified staging schema.
"""
from __future__ import annotations
import os
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

# Canonical feature names matching all possible features in the dashboard and datasets
CANONICAL_COLUMNS = [
    "source_dataset",
    "employee_id",
    "work_hours_per_week",
    "overtime_hours",
    "meetings_per_day",
    "deadlines_missed",
    "experience_years",
    "job_role",
    "work_mode",
    "job_satisfaction",
    "manager_support",
    "work_life_balance",
    "sleep_hours",
    "physical_activity_days",
    "stress_level",
    "mental_fatigue_score",
    "burnout_score",
    "caffeine_intake",
    "company_size",
    "screen_time_hours",
    "social_support_score"
]

def load_tech_burnout(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "tech_mental_health_burnout.csv"
    if not path.exists():
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    df = pd.read_csv(path)
    
    # Direct mapping
    mapped = pd.DataFrame(index=df.index)
    mapped["source_dataset"] = "tech_mental_health_burnout"
    mapped["employee_id"] = df.index.map(lambda x: f"tech_{x}")
    mapped["work_hours_per_week"] = df["work_hours_per_week"]
    mapped["overtime_hours"] = df["overtime_hours"]
    mapped["meetings_per_day"] = df["meetings_per_day"]
    mapped["deadlines_missed"] = df["deadlines_missed"]
    mapped["experience_years"] = df["experience_years"]
    mapped["job_role"] = df["job_role"]
    mapped["work_mode"] = df["work_mode"]
    mapped["job_satisfaction"] = df["job_satisfaction"] * 20.0  # Scale 1-5 to 0-100
    mapped["manager_support"] = df["manager_support"] * 20.0  # Scale 1-5 to 0-100
    mapped["work_life_balance"] = df["work_life_balance"] * 20.0  # Scale 1-5 to 0-100
    mapped["sleep_hours"] = df["sleep_hours"]
    mapped["physical_activity_days"] = df["physical_activity_days"]
    mapped["stress_level"] = df["stress_level"]
    mapped["mental_fatigue_score"] = np.nan
    mapped["burnout_score"] = df["burnout_score"] # scale is 0-10
    mapped["caffeine_intake"] = df["caffeine_intake"]
    mapped["company_size"] = df["company_size"]
    mapped["screen_time_hours"] = df["screen_time_hours"]
    mapped["social_support_score"] = df["social_support_score"]
    
    return mapped

def load_are_employees_burning_out(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "Are Your Employees Burning Out.csv"
    if not path.exists():
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    df = pd.read_csv(path)
    
    mapped = pd.DataFrame(index=df.index)
    mapped["source_dataset"] = "are_employees_burning_out"
    mapped["employee_id"] = df["Employee ID"]
    mapped["work_hours_per_week"] = df["Resource Allocation"] * 8.0  # Proxy resource allocation to hours
    mapped["overtime_hours"] = np.nan
    mapped["meetings_per_day"] = np.nan
    mapped["deadlines_missed"] = np.nan
    mapped["experience_years"] = df["Designation"] * 2.0  # Designation (0-5) proxy for experience
    mapped["job_role"] = "Unknown"
    mapped["work_mode"] = df["WFH Setup Available"].map({"Yes": "Remote", "No": "Onsite"})
    mapped["job_satisfaction"] = np.nan
    mapped["manager_support"] = np.nan
    mapped["work_life_balance"] = np.nan
    mapped["sleep_hours"] = np.nan
    mapped["physical_activity_days"] = np.nan
    mapped["stress_level"] = np.nan
    mapped["mental_fatigue_score"] = df["Mental Fatigue Score"] * 10.0  # Scale 0-10 to 0-100
    mapped["burnout_score"] = df["Burn Rate"] * 10.0  # Scale 0-1.0 to 0-10
    mapped["caffeine_intake"] = np.nan
    mapped["company_size"] = "Unknown"
    mapped["screen_time_hours"] = np.nan
    mapped["social_support_score"] = np.nan
    
    return mapped

def load_mental_health_workplace_survey(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "mental_health_workplace_survey.csv"
    if not path.exists():
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    df = pd.read_csv(path)
    
    mapped = pd.DataFrame(index=df.index)
    mapped["source_dataset"] = "mental_health_workplace_survey"
    mapped["employee_id"] = df["EmployeeID"].astype(str)
    mapped["work_hours_per_week"] = df["WorkHoursPerWeek"]
    mapped["overtime_hours"] = np.nan
    mapped["meetings_per_day"] = np.nan
    mapped["deadlines_missed"] = np.nan
    mapped["experience_years"] = df["YearsAtCompany"]
    mapped["job_role"] = df["JobRole"]
    mapped["work_mode"] = df["RemoteWork"].map({"Yes": "Remote", "No": "Onsite"})
    mapped["job_satisfaction"] = df["JobSatisfaction"] * 10.0  # Scale 1-10 to 0-100
    mapped["manager_support"] = df["ManagerSupportScore"] * 20.0  # Scale 1-5 to 0-100
    mapped["work_life_balance"] = df["WorkLifeBalanceScore"] * 20.0  # Scale 1-5 to 0-100
    mapped["sleep_hours"] = df["SleepHours"]
    mapped["physical_activity_days"] = df["PhysicalActivityHrs"] / 2.0  # Approx days
    mapped["stress_level"] = df["StressLevel"]
    mapped["mental_fatigue_score"] = np.nan
    mapped["burnout_score"] = df["BurnoutLevel"]  # scale is 1-10
    mapped["caffeine_intake"] = np.nan
    mapped["company_size"] = df["TeamSize"].map(lambda x: "Large" if x > 15 else "Medium")
    mapped["screen_time_hours"] = np.nan
    mapped["social_support_score"] = np.nan
    
    return mapped

def load_train_csv(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "train.csv"
    if not path.exists():
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    df = pd.read_csv(path)
    
    mapped = pd.DataFrame(index=df.index)
    mapped["source_dataset"] = "train_csv_small"
    mapped["employee_id"] = df["Employee_Id"]
    mapped["work_hours_per_week"] = df["Avg_Working_Hours_Per_Day"] * 5.0
    mapped["overtime_hours"] = np.nan
    mapped["meetings_per_day"] = np.nan
    mapped["deadlines_missed"] = np.nan
    mapped["experience_years"] = np.nan
    mapped["job_role"] = "Unknown"
    mapped["work_mode"] = df["Work_From"]
    mapped["job_satisfaction"] = df["Job_Satisfaction"].map({"Low": 25, "Medium": 50, "High": 75, "Very High": 100})
    mapped["manager_support"] = df["Manager_Support"].map({"Low": 25, "Medium": 50, "High": 75, "Very High": 100})
    mapped["work_life_balance"] = df["Work_Life_Balance"].map({"Poor": 25, "Fair": 50, "Good": 75, "Excellent": 100})
    mapped["sleep_hours"] = df["Sleeping_Habit"].map({"Poor": 5, "Fair": 7, "Good": 8})
    mapped["physical_activity_days"] = df["Exercise_Habit"].map({"No": 0, "Yes": 3})
    mapped["stress_level"] = df["Stress_Level"]
    mapped["mental_fatigue_score"] = np.nan
    mapped["burnout_score"] = np.nan
    mapped["caffeine_intake"] = np.nan
    mapped["company_size"] = "Unknown"
    mapped["screen_time_hours"] = np.nan
    mapped["social_support_score"] = np.nan
    
    return mapped

def load_all_sources(data_dir_str: str) -> pd.DataFrame:
    data_dir = Path(data_dir_str)
    dfs = [
        load_tech_burnout(data_dir),
        load_are_employees_burning_out(data_dir),
        load_mental_health_workplace_survey(data_dir),
        load_train_csv(data_dir)
    ]
    # Filter empty frames
    dfs = [d for d in dfs if not d.empty]
    if not dfs:
        raise ValueError("No valid raw data sources found or loaded.")
    
    master_df = pd.concat(dfs, ignore_index=True)
    return master_df[CANONICAL_COLUMNS]
