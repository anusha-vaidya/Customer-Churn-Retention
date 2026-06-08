# ============================================
# 03_modeling.py
# Customer Churn & Retention Dashboard Project
# ============================================

import pandas as pd
from src.model_training import train_logistic_model

# 1. Load processed datasets
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# 2. Train model + get predictions
model, y_pred, y_prob = train_logistic_model(X_train, y_train, X_test)

# 3. Load cleaned dataset to retrieve customerID for the test rows
clean_df = pd.read_csv("data/processed/cleaned_churn.csv")

# IMPORTANT:
# X_test.index corresponds to the same row positions in cleaned_churn.csv
customer_ids = clean_df.loc[X_test.index, "customerID"].values

# 4. Save predictions for Power BI
pred_df = X_test.copy()
pred_df["customerID"] = customer_ids
pred_df["ActualChurn"] = y_test.values
pred_df["PredictedChurn"] = y_pred
pred_df["ChurnProbability"] = y_prob

pred_df.to_csv("data/processed/model_predictions.csv", index=False)
print("Model predictions saved to data/processed/model_predictions.csv")
