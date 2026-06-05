# ============================================
# 01_data_exploration.ipynb
# Customer Churn & Retention Dashboard Project
# ============================================

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

# 2. Load dataset
df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preview
df.head()

# 3. Basic dataset info
print("Dataset Info:")
df.info()

print("\nSummary Statistics:")
df.describe(include='all')

# 4. Check missing values
print("\nMissing Values:")
df.isnull().sum()

# 5. Convert TotalCharges to numeric (professional step)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# 6. Churn distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Churn')
plt.title("Churn Distribution")
plt.show()

# 7. Churn by Contract Type
plt.figure(figsize=(7,4))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title("Churn by Contract Type")
plt.xticks(rotation=45)
plt.show()

# 8. Save cleaned version for modeling
df.to_csv("data/processed/cleaned_churn.csv", index=False)

print("\nCleaned dataset saved to data/processed/cleaned_churn.csv")

