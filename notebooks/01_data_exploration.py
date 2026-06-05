# ============================================
# 01_data_exploration.py
# Customer Churn & Retention Dashboard Project
# ============================================

import pandas as pd
from src.data_cleaning import clean_raw_data

# 1. Load + clean raw dataset
df = clean_raw_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe(include="all"))

print("\nMissing Values:")
print(df.isnull().sum())

# 2. Save cleaned dataset
df.to_csv("data/processed/cleaned_churn.csv", index=False)
print("\nCleaned dataset saved to data/processed/cleaned_churn.csv")
