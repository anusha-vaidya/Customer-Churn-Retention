# ============================================
# 02_feature_engineering.py
# Customer Churn & Retention Dashboard Project
# ============================================

import pandas as pd
from src.feature_engineering import engineer_features

# 1. Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_churn.csv")

# 2. Feature engineering + train/test split
X_train, X_test, y_train, y_test = engineer_features(df)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# 3. Save processed datasets
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Feature engineering complete. Files saved in data/processed/")
