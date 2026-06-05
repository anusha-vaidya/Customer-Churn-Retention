# ============================================
# 04_feature_importance.py
# Customer Churn & Retention Dashboard Project
# ============================================

import pandas as pd
from src.model_training import train_logistic_model

# 1. Load training data
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

# 2. Train model (again) to extract coefficients
model, _, _ = train_logistic_model(X_train, y_train, X_train)

# 3. Extract feature importance
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.coef_[0]
}).sort_values(by="Importance", ascending=False)

# 4. Save to CSV
importance_df.to_csv("data/processed/feature_importance.csv", index=False)
print("Feature importance saved to data/processed/feature_importance.csv")
