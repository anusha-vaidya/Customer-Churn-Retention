# ============================================
# 03_model_training.py
# Customer Churn & Retention Dashboard Project
# ============================================

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. Load processed data
X_train = pd.read_csv("../data/processed/X_train.csv")
X_test = pd.read_csv("../data/processed/X_test.csv")
y_train = pd.read_csv("../data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("../data/processed/y_test.csv").values.ravel()

# ============================================
# Logistic Regression (Baseline Model)
# ============================================

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

log_preds = log_model.predict(X_test)

print("\n===== Logistic Regression Results =====")
print("Accuracy:", accuracy_score(y_test, log_preds))
print("Precision:", precision_score(y_test, log_preds))
print("Recall:", recall_score(y_test, log_preds))
print("F1 Score:", f1_score(y_test, log_preds))
print("\nClassification Report:\n", classification_report(y_test, log_preds))

# ============================================
# Random Forest (Stronger Model)
# ============================================

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("\n===== Random Forest Results =====")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("Precision:", precision_score(y_test, rf_preds))
print("Recall:", recall_score(y_test, rf_preds))
print("F1 Score:", f1_score(y_test, rf_preds))
print("\nClassification Report:\n", classification_report(y_test, rf_preds))

# ============================================
# Save predictions for dashboard
# ============================================

pred_df = pd.DataFrame({
    "Actual": y_test,
    "Logistic_Pred": log_preds,
    "RF_Pred": rf_preds
})

pred_df.to_csv("../data/processed/model_predictions.csv", index=False)

print("\nModel predictions saved to data/processed/model_predictions.csv")

