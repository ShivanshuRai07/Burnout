"""Preprocessing module for the Burnout Platform.

Handles cleaning, imputation (group-aware medians), normalization, and standardizing categories.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Standardize work_mode
    if "work_mode" in df.columns:
        df["work_mode"] = df["work_mode"].str.strip().str.capitalize()
        df["work_mode"] = df["work_mode"].replace({
            "Hybrid": "Hybrid",
            "Remote": "Remote",
            "Onsite": "Onsite",
            "On-site": "Onsite",
            "Home": "Remote",
            "Office": "Onsite"
        })
        df["work_mode"] = df["work_mode"].fillna("Unknown")
        
    # Standardize job_role
    if "job_role" in df.columns:
        df["job_role"] = df["job_role"].str.strip().str.title()
        df["job_role"] = df["job_role"].fillna("Unknown")
        
    # Standardize company_size
    if "company_size" in df.columns:
        df["company_size"] = df["company_size"].fillna("Medium")
        
    return df

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Track missingness for key columns
    cols_to_track = [
        "work_hours_per_week", 
        "overtime_hours", 
        "meetings_per_day", 
        "deadlines_missed", 
        "experience_years",
        "job_satisfaction",
        "manager_support",
        "work_life_balance",
        "sleep_hours",
        "physical_activity_days",
        "mental_fatigue_score",
        "caffeine_intake",
        "screen_time_hours",
        "social_support_score"
    ]
    
    for col in cols_to_track:
        if col in df.columns:
            df[f"{col}_is_missing"] = df[col].isna().astype(int)
            
            # Group-aware median imputation (by job_role)
            # If the group median is NaN, fall back to global median
            global_median = df[col].median()
            if pd.isna(global_median):
                global_median = 0.0
            
            group_medians = df.groupby("job_role")[col].transform("median")
            df[col] = df[col].fillna(group_medians).fillna(global_median)
            
    return df

def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Normalize burnout_score to 0-10 scale
    if "burnout_score" in df.columns:
        df["burnout_score"] = df["burnout_score"].clip(0.0, 10.0)
        
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_categories(df)
    df = impute_missing_values(df)
    df = normalize_metrics(df)
    
    # Remove duplicates based on employee_id and source_dataset
    df = df.drop_duplicates(subset=["employee_id", "source_dataset"], keep="first")
    
    return df
