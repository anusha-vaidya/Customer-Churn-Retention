# ============================================
# 04_feature_importance.py
# Customer Churn & Retention Dashboard Project
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# 1. Load processed data
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

# 2. Train Random Forest (same config as before)
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

# 3. Extract feature importance
importance = rf_model.feature_importances_
feature_names = X_train.columns

feat_imp = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nTop Features Driving Churn:")
print(feat_imp.head(10))

# 4. Plot feature importance
plt.figure(figsize=(10,6))
sns.barplot(data=feat_imp.head(10), x="Importance", y="Feature", palette="viridis")
plt.title("Top 10 Features Driving Customer Churn")
plt.tight_layout()
plt.show()

# 5. Save feature importance for dashboard
feat_imp.to_csv("data/processed/feature_importance.csv", index=False)

print("\nFeature importance saved to data/processed/feature_importance.csv")
